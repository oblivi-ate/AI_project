from ortools.linear_solver import pywraplp
from itertools import combinations
import os
import numpy as np
import random
import time  # 添加time模块导入
import pickle  # 添加pickle模块导入
import hashlib  # 添加hashlib模块导入
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import math
from functools import partial

def is_dominated(set1, set2):
    """检查set1是否被set2支配（set2包含set1的所有元素且规模更小或相等）"""
    return set1.issubset(set2) and len(set1) >= len(set2)

def print_combination(comb):
    """格式化打印组合"""
    return ','.join(comb)

def check_dominated(subsets):
    """检查并移除被支配的子集"""
    non_dominated = []
    for i, s1 in enumerate(subsets):
        dominated = False
        for j, s2 in enumerate(subsets):
            if i != j and s2.issubset(s1):
                dominated = True
                break
        if not dominated:
            non_dominated.append(s1)
    return non_dominated

def calculate_overlap(set1, set2):
    """计算两个集合的重叠度"""
    return len(set1.intersection(set2))

def construct_ilp_problem(n, k, j, s, strict_coverage=True, min_cover=1):
    """
    构建ILP问题
    
    参数:
    n: 样本数量
    k: k元素子集大小
    j: j元素子集大小
    s: s元素子集大小
    strict_coverage: 是否使用严格覆盖模式
    min_cover: 宽松覆盖模式下，每个j元素子集至少需要被覆盖的s元素子集数量
    """
    # 生成样本集合
    samples = set([chr(ord('A') + i) for i in range(n)])
    print(f"生成的样本集合: {samples}")
    
    # 生成k元素子集
    k_subsets = [set(comb) for comb in combinations(samples, k)]
    print(f"生成的{k}元素子集数量: {len(k_subsets)}")
    
    # 预处理：移除被支配的子集
    k_subsets = check_dominated(k_subsets)
    print(f"预处理后的{k}元素子集数量: {len(k_subsets)}")
    
    # 生成j元素子集
    j_subsets = [set(comb) for comb in combinations(samples, j)]
    print(f"生成的{j}元素子集数量: {len(j_subsets)}")
    
    # 生成s元素子集
    s_subsets = []
    for j_subset in j_subsets:
        s_subsets.append([set(comb) for comb in combinations(j_subset, s)])
    print(f"每个{j}元素子集包含的{s}元素子集数量: {len(s_subsets[0])}")
    
    # 创建求解器
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception('SCIP求解器不可用')
    
    # 创建变量
    x = {}
    for i in range(len(k_subsets)):
        x[i] = solver.IntVar(0, 1, f'x_{i}')
    
    # 目标函数：最小化选择的集合数量
    objective = solver.Objective()
    for i in range(len(k_subsets)):
        objective.SetCoefficient(x[i], 1)
    objective.SetMinimization()
    
    if strict_coverage:
        # 严格覆盖模式：每个j元素子集的所有s元素子集都必须被至少一个k元素子集覆盖
        if min_cover > 1:
            print("注意：在严格覆盖模式下，min_cover参数无效")
        for j_idx, j_subset in enumerate(j_subsets):
            for s_subset in s_subsets[j_idx]:
                covering_sets = [i for i, k_subset in enumerate(k_subsets) if s_subset.issubset(k_subset)]
                if covering_sets:
                    constraint = solver.Constraint(1, solver.infinity())
                    for i in covering_sets:
                        constraint.SetCoefficient(x[i], 1)
    else:
        # 宽松覆盖模式：每个j元素子集至少需要min_cover个s元素子集被覆盖
        for j_idx, j_subset in enumerate(j_subsets):
            s_covered = []
            for s_subset in s_subsets[j_idx]:
                covering_sets = [i for i, k_subset in enumerate(k_subsets) if s_subset.issubset(k_subset)]
                if covering_sets:
                    y = solver.IntVar(0, 1, f'y_{j_idx}_{len(s_covered)}')
                    constraint = solver.Constraint(0, solver.infinity())
                    for i in covering_sets:
                        constraint.SetCoefficient(x[i], 1)
                    constraint.SetCoefficient(y, -1)
                    s_covered.append(y)
            
            if s_covered:
                constraint = solver.Constraint(min_cover, solver.infinity())
                for y in s_covered:
                    constraint.SetCoefficient(y, 1)
    
    return solver, x, k_subsets, j_subsets, s_subsets

def get_cache_filename(n, k, j, s):
    """根据参数生成缓存文件名"""
    # 使用MD5哈希值作为文件名，保证唯一性
    key = f"n{n}_k{k}_j{j}_s{s}"
    hash_obj = hashlib.md5(key.encode())
    hash_str = hash_obj.hexdigest()
    return f"cache_coverage_{hash_str}.pkl"

def save_coverage_cache(cache_data, filename):
    """将覆盖关系缓存保存到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"缓存已保存到: {filename}")

def load_coverage_cache(filename):
    """从文件加载覆盖关系缓存"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            cache_data = pickle.load(f)
        print(f"从缓存文件加载: {filename}")
        return cache_data
    return None

def process_k_set_chunk(args):
    """处理一个k集合块的覆盖关系"""
    chunk_idx, chunk, all_j_combinations, n, s = args
    chunk_size = len(chunk)
    result = {}
    j_sets_bits = np.zeros((len(all_j_combinations), n), dtype=bool)
    for i, j_comb in enumerate(all_j_combinations):
        j_sets_bits[i, list(j_comb)] = True
    
    for k_idx_local, k_set in enumerate(chunk):
        k_idx = chunk_idx * chunk_size + k_idx_local
        k_bit = np.zeros(n, dtype=bool)
        k_bit[list(k_set)] = True
        
        result[k_idx] = {}
        
        # 使用向量化操作计算交集
        intersections = np.logical_and(k_bit, j_sets_bits).sum(axis=1)
        
        # 处理每个j组合
        for j_idx, intersection_size in enumerate(intersections):
            j_comb = all_j_combinations[j_idx]
            
            # 只有当交集大小满足条件时才计算s子集
            if intersection_size >= s:
                # 计算该k集合覆盖的s子集
                s_combs = combinations(j_comb, s)
                covered_s = set()
                
                # 只考虑那些可能被k_set覆盖的s组合
                for s_comb in s_combs:
                    # 检查s_comb是否是k_set的子集
                    if all(item in k_set for item in s_comb):
                        covered_s.add(tuple(sorted(s_comb)))
                
                if covered_s:  # 只保存非空结果
                    result[k_idx][j_idx] = covered_s
    
    return result

def is_subset_covered(j_comb, k_set, s):
    """检查j组合与k集合的交集是否大于等于s
    
    优化版本：使用集合运算而不是逐个元素检查
    """
    # 将输入转为集合，确保操作正确
    j_set = set(j_comb) if not isinstance(j_comb, set) else j_comb
    k_set = set(k_set) if not isinstance(k_set, set) else k_set
    
    # 计算交集大小
    common = j_set & k_set
    return len(common) >= s

def compute_coverage_for_chunk(k_chunk, j_combinations, n, total_j_combinations, s=4):
    """计算一个数据块的覆盖关系"""
    result = {}
    chunk_size = len(k_chunk)
    
    for idx, k_set in enumerate(k_chunk):
        if idx % max(1, chunk_size // 5) == 0:
            print(f"并行进程: 完成 {idx}/{chunk_size} 计算")
            
        covered_j_combinations = set()
        # 将k_set转换为frozenset以便作为字典键
        k_set_frozen = frozenset(k_set)
        
        for j_idx, j_comb in enumerate(j_combinations):
            if is_subset_covered(j_comb, k_set, s):
                covered_j_combinations.add(j_idx)
                
        if covered_j_combinations:  # 只保存有覆盖的结果，节省内存
            result[k_set_frozen] = covered_j_combinations
        
    return result

def calculate_fitness(solution, all_coverage, j_combinations):
    """计算单个解的适应度"""
    covered_j_indices = set()
    for k_set in solution:
        k_set_frozen = frozenset(k_set)
        if k_set_frozen in all_coverage:
            covered_j_indices.update(all_coverage[k_set_frozen])
    
    # 返回覆盖率和解大小的加权组合
    coverage_ratio = len(covered_j_indices) / len(j_combinations) if j_combinations else 0
    solution_size_penalty = 0.05 * len(solution)  # 每个额外的集合有轻微惩罚
    
    # 返回最终得分 (0-100分)
    return coverage_ratio * 100 - solution_size_penalty

def calculate_fitness_for_chunk(solution_chunk, all_coverage, j_combinations):
    """计算一组解的适应度"""
    result = {}
    for solution in solution_chunk:
        solution_frozen = frozenset(map(frozenset, solution))
        result[solution_frozen] = calculate_fitness(solution, all_coverage, j_combinations)
    return result

def precompute_coverage_relations(n, k, j, s, use_cache=True, n_jobs=4):
    """预计算k集合对j组合的覆盖关系"""
    cache_filename = f"coverage_cache_n{n}_k{k}_j{j}_s{s}.pkl"
    
    # 检查缓存
    if use_cache and os.path.exists(cache_filename):
        print(f"从缓存加载覆盖关系: {cache_filename}")
        try:
            with open(cache_filename, 'rb') as f:
                cached_data = pickle.load(f)
                
            # 处理不同格式的缓存数据
            if isinstance(cached_data, tuple) and len(cached_data) == 3:
                return cached_data
            elif isinstance(cached_data, dict):
                # 如果缓存只有覆盖关系，重新计算组合
                print("缓存格式不完整，重新生成组合数据...")
                import itertools
                k_combinations = list(itertools.combinations(range(n), k))
                j_combinations = list(itertools.combinations(range(n), j))
                return cached_data, k_combinations, j_combinations
            else:
                print(f"未知缓存格式，重新计算...")
        except Exception as e:
            print(f"读取缓存出错: {e}，重新计算...")
    
    print(f"开始计算覆盖关系，参数: n={n}, k={k}, j={j}, s={s}")
    
    # 获取所有可能的k集合和j组合
    import itertools
    k_combinations = list(itertools.combinations(range(n), k))
    j_combinations = list(itertools.combinations(range(n), j))
    
    # 当问题规模较大时进行并行计算
    use_parallel = should_use_parallel(n, k, j)
    
    # 获取最佳并行进程数
    if use_parallel:
        import multiprocessing
        available_cpus = multiprocessing.cpu_count()
        
        # 对于n=13这种中等规模问题，限制进程数以减少开销
        if 10 <= n <= 13:
            # 限制最大进程数为4或可用CPU数的1/3，取较小值 (进一步减少进程数)
            optimal_jobs = min(4, max(2, available_cpus // 3))
        else:
            # 大规模问题，使用更多进程
            optimal_jobs = min(n_jobs, available_cpus)
        
        min_chunk_size = 500  # 确保每个进程有足够的工作量
        max_processes = max(1, min(optimal_jobs, len(k_combinations) // min_chunk_size))
        
        print(f"使用并行计算，进程数: {max_processes}，CPU核心数: {available_cpus}")
        n_jobs = max_processes
    else:
        n_jobs = 1
        print("问题规模较小，使用串行计算")
    
    all_coverage = {}
    
    # 使用改进的并行计算逻辑
    if n_jobs > 1:
        # 根据进程数划分k集合
        chunk_size = len(k_combinations) // n_jobs
        if chunk_size < 1:
            chunk_size = 1
        
        k_chunks = [k_combinations[i:i+chunk_size] for i in range(0, len(k_combinations), chunk_size)]
        
        # 合并最后两个块如果最后一个块太小
        if len(k_chunks) > 1 and len(k_chunks[-1]) < chunk_size // 2:
            k_chunks[-2].extend(k_chunks[-1])
            k_chunks.pop()
        
        print(f"并行计算：将{len(k_combinations)}个组合分为{len(k_chunks)}个任务块")
        
        # 使用multiprocessing并行计算
        with multiprocessing.Pool(processes=min(n_jobs, len(k_chunks))) as pool:
            results = []
            for chunk in k_chunks:
                results.append(pool.apply_async(compute_coverage_for_chunk, 
                                              args=(chunk, j_combinations, n, len(j_combinations), s)))
            
            # 收集结果
            for result in results:
                chunk_result = result.get()
                all_coverage.update(chunk_result)
    else:
        # 串行计算
        all_coverage = compute_coverage_for_chunk(k_combinations, j_combinations, n, len(j_combinations), s)
    
    print(f"覆盖关系计算完成，共{len(all_coverage)}个k集合，{len(j_combinations)}个j组合")
    
    # 保存到缓存
    if use_cache:
        with open(cache_filename, 'wb') as f:
            pickle.dump((all_coverage, k_combinations, j_combinations), f)
        print(f"覆盖关系已缓存到: {cache_filename}")
    
    # 返回三个值以保持与原函数兼容
    return all_coverage, k_combinations, j_combinations

def parallel_fitness_calculation(population, all_coverage, j_combinations, n=None, k=None, j=None, n_jobs=None):
    """并行计算适应度，针对不同规模问题优化"""
    # 调整并行度
    actual_n_jobs = 1  # 默认串行
    
    # 判断是否应该使用并行
    use_parallel = should_use_parallel(n, k, j) and len(population) > 50
    
    # 根据问题规模调整并行度
    if use_parallel:
        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()
            
        if 12 <= n <= 14:
            # 中等规模问题优化
            actual_n_jobs = min(4, n_jobs)
            print(f"适应度计算: 中等规模问题 (n={n})，使用 {actual_n_jobs} 个进程")
        else:
            actual_n_jobs = n_jobs
    
    # 执行适应度计算
    if use_parallel and actual_n_jobs > 1:
        # 确保每个进程有足够工作量
        chunk_size = max(10, len(population) // actual_n_jobs)
        population_chunks = [population[i:i + chunk_size] for i in range(0, len(population), chunk_size)]
        
        # 使用线程池替代进程池，减少开销
        with ThreadPoolExecutor(max_workers=actual_n_jobs) as executor:
            results = list(executor.map(
                lambda chunk: calculate_fitness_for_chunk(chunk, all_coverage, j_combinations),
                population_chunks
            ))
        
        # 合并结果
        all_solutions_fitness = {}
        for chunk_result in results:
            all_solutions_fitness.update(chunk_result)
    else:
        # 串行计算
        all_solutions_fitness = {}
        for solution in population:
            solution_frozen = frozenset(map(frozenset, solution))
            all_solutions_fitness[solution_frozen] = calculate_fitness(solution, all_coverage, j_combinations)
    
    return all_solutions_fitness

def fitness_with_lookup(solution, coverage_lookup, all_k_sets, all_j_combinations, s, strict_coverage=True, min_cover=1):
    """使用优化的查找表计算适应度"""
    # 将solution中的k集合转换为索引
    solution_indices = []
    for k_set in solution:
        try:
            k_idx = all_k_sets.index(k_set)
            solution_indices.append(k_idx)
        except ValueError:
            # 如果k_set不在all_k_sets中，则跳过
            continue
    
    # 计算每个j组合被覆盖的s子集
    covered_j_count = 0
    total_coverage_quality = 0
    
    for j_idx in range(len(all_j_combinations)):
        covered_s = set()
        # 只考虑有覆盖关系的k集合
        for k_idx in solution_indices:
            if k_idx in coverage_lookup and j_idx in coverage_lookup[k_idx]:
                covered_s.update(coverage_lookup[k_idx][j_idx])
        
        # 根据覆盖模式计算质量分数
        j_comb = all_j_combinations[j_idx]
        s_combinations = list(combinations(j_comb, s))
        
        if strict_coverage:
            # 严格覆盖模式
            if len(covered_s) == len(s_combinations):
                covered_j_count += 1
                total_coverage_quality += 1
        else:
            # 宽松覆盖模式
            s_covered = len(covered_s)
            if s_covered >= min_cover:
                covered_j_count += 1
                # 增加对超额覆盖的奖励，但降低权重
                extra_coverage = (s_covered - min_cover) / len(s_combinations)
                total_coverage_quality += 1 + extra_coverage * 0.1
            else:
                # 对接近目标的解给予部分奖励
                coverage_ratio = s_covered / min_cover
                total_coverage_quality += coverage_ratio * 0.5
    
    # 计算综合得分
    coverage_ratio = covered_j_count / len(all_j_combinations)
    coverage_score = total_coverage_quality / len(all_j_combinations) * 100
    
    # 增加对解大小的惩罚，使用平方函数惩罚
    size_penalty = len(solution) * len(solution) * 0.08  # 使用与fitness相同的惩罚
    
    coverage_bonus = 0 if strict_coverage else coverage_ratio * 3.0 # 减少覆盖奖励
    low_coverage_penalty = 50.0 if coverage_ratio < 0.5 else 0
    
    return coverage_score - size_penalty + coverage_bonus - low_coverage_penalty

def generate_initial_solution_ga(n, k, j, s, population_size=50, generations=100, use_cache=True, n_jobs=-1):
    """使用遗传算法生成初始解，支持缓存和并行化"""
    # 使用GA_version_2中的遗传算法
    solution = genetic_algorithm(n, k, j, s, 
                               population_size=population_size,
                               generations=generations,
                               base_mutation_rate=0.1,
                               base_crossover_rate=0.9,
                               elitism=True,
                               strict_coverage=False,
                               min_cover=1,
                               use_cache=use_cache,
                               n_jobs=n_jobs)
    
    # 检查解的覆盖情况
    satisfies = satisfies_condition(solution, n, k, j, s, strict_coverage=False, min_cover=1)
    
    # 计算并输出最终解的适应度
    final_fitness = fitness(solution, n, k, j, s, strict_coverage=False, min_cover=1)
    print(f"遗传算法找到的解的适应度: {final_fitness:.2f}")
    print(f"覆盖状态: {'完全覆盖' if satisfies else '部分覆盖'}")
    
    return solution

def generate_combinations(items, k):
    return list(combinations(items, k))

def check_coverage(group, j_combination, s):
    # Check if the group contains at least s items from the j_combination
    common_elements = set(group) & set(j_combination)
    return len(common_elements) >= s

def satisfies_condition(solution, n, k, j, s, strict_coverage=True, min_cover=1, coverage_cache=None):
    """检查解是否满足约束条件，支持宽松和严格覆盖模式，使用缓存优化性能"""
    # 使用缓存加速重复计算
    if coverage_cache is not None:
        solution_key = tuple(sorted(solution))
        if solution_key in coverage_cache:
            return coverage_cache[solution_key]
    
    solution_sets = [set(group) for group in solution]
    all_items = list(range(n))
    j_combinations = list(combinations(all_items, j))
    
    # 检查是否所有j组合都被覆盖
    for j_comb in j_combinations:
        if strict_coverage:
            # 严格覆盖模式：检查所有s元素子集是否都被覆盖
            s_combinations = list(combinations(j_comb, s))
            for s_comb in s_combinations:
                s_covered = False
                for group in solution_sets:
                    if set(s_comb).issubset(group):
                        s_covered = True
                        break
                if not s_covered:
                    if coverage_cache is not None:
                        coverage_cache[tuple(sorted(solution))] = False
                    return False
        else:
            # 宽松覆盖模式：至少一个k集合包含指定数量的s组合
            s_combinations = list(combinations(j_comb, s))
            # 检查至少有1个k集合包含至少min_cover的s组合
            covered = False
            count = 0
            for s_comb in s_combinations:
                for group in solution_sets:
                    if set(s_comb).issubset(group):
                        count += 1
                        if count >= min_cover:
                            covered = True
                            break
                if covered:
                    break
            if not covered:
                if coverage_cache is not None:
                    coverage_cache[tuple(sorted(solution))] = False
                return False
    
    if coverage_cache is not None:
        coverage_cache[tuple(sorted(solution))] = True
    return True

def fitness(solution, n, k, j, s, strict_coverage=True, min_cover=1):
    """计算解的适应度，支持宽松和严格覆盖模式"""
    solution_sets = [set(group) for group in solution]
    all_items = list(range(n))
    j_combinations = list(combinations(all_items, j))
    
    # 计算覆盖的j组合数量和总的覆盖质量
    total_coverage_quality = 0
    covered_j_count = 0
    
    for j_comb in j_combinations:
        j_set = set(j_comb)
        if strict_coverage:
            # 严格覆盖模式：检查所有s元素子集是否都被覆盖
            s_combinations = list(combinations(j_comb, s))
            all_covered = True
            for s_comb in s_combinations:
                s_covered = False
                for group in solution_sets:
                    if set(s_comb).issubset(group):
                        s_covered = True
                        break
                if not s_covered:
                    all_covered = False
                    break
            if all_covered:
                covered_j_count += 1
                total_coverage_quality += 1
        else:
            # 宽松覆盖模式：计算每个j组合中被覆盖的s组合数量
            s_combinations = list(combinations(j_comb, s))
            covered_s_combinations = set()
            
            # 对于每个解集合，找出它能覆盖的s组合
            for group in solution_sets:
                # 只考虑与当前j组合有足够交集的集合
                if len(group & j_set) >= s:
                    # 找出这个集合能覆盖的所有s组合
                    for s_comb in s_combinations:
                        if set(s_comb).issubset(group):
                            covered_s_combinations.add(tuple(sorted(s_comb)))
            
            # 计算覆盖质量
            s_covered = len(covered_s_combinations)
            if s_covered >= min_cover:
                covered_j_count += 1
                # 增加对超额覆盖的奖励，但降低权重
                extra_coverage = (s_covered - min_cover) / len(s_combinations)
                total_coverage_quality += 1 + extra_coverage * 0.1
            else:
                # 对接近目标的解给予部分奖励
                coverage_ratio = s_covered / min_cover
                total_coverage_quality += coverage_ratio * 0.5
    
    # 计算覆盖率
    coverage_ratio = covered_j_count / len(j_combinations)
    
    # 综合评分系统
    coverage_score = total_coverage_quality / len(j_combinations) * 100  # 基础覆盖分数
    
    # 增加对解大小的惩罚，使用平方函数惩罚
    size_penalty = len(solution) * len(solution) * 0.08  # 显著增加惩罚
    
    coverage_bonus = 0 if strict_coverage else coverage_ratio * 3.0  # 减少覆盖奖励
    low_coverage_penalty = 50.0 if coverage_ratio < 0.5 else 0  # 低覆盖惩罚
    
    return coverage_score - size_penalty + coverage_bonus - low_coverage_penalty

def create_population_chunk(args):
    """创建单个种群块的函数"""
    n, k, size, min_groups, max_groups, j, s, chunk_id = args
    population_chunk = []
    
    # 预计算所有组合
    all_items = list(range(n))
    local_all_k_combinations = list(combinations(all_items, k))
    local_all_j_combinations = list(combinations(all_items, j))
    
    # 计算每个k组合的覆盖能力
    coverage_scores = []
    for k_set in local_all_k_combinations:
        k_set = set(k_set)
        score = 0
        for j_comb in local_all_j_combinations:
            if len(set(j_comb) & k_set) >= s:
                score += 1
        coverage_scores.append(score)
    
    # 选择覆盖能力最强的前50%组合
    sorted_indices = sorted(range(len(coverage_scores)), key=lambda i: coverage_scores[i], reverse=True)
    top_indices = sorted_indices[:len(sorted_indices)//2]
    top_combinations = [local_all_k_combinations[i] for i in top_indices]
    
    # 创建指定数量的解
    for _ in range(size):
        solution = []
        uncovered = set(range(len(local_all_j_combinations)))
        
        # 贪心地添加覆盖最多未覆盖j组合的k集合
        while uncovered and len(solution) < max_groups:
            best_combination = None
            best_coverage = -1
            
            # 随机选择一些候选集合来减少计算量
            candidates = random.sample(top_combinations, min(20, len(top_combinations)))
            
            for k_set in candidates:
                k_set = set(k_set)
                coverage = 0
                for j_idx in list(uncovered):
                    j_comb = local_all_j_combinations[j_idx]
                    if len(set(j_comb) & k_set) >= s:
                        coverage += 1
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_combination = tuple(sorted(k_set))
            
            if best_combination and best_coverage > 0:
                solution.append(best_combination)
                # 更新未覆盖列表
                for j_idx in list(uncovered):
                    j_comb = local_all_j_combinations[j_idx]
                    if len(set(j_comb) & set(best_combination)) >= s:
                        uncovered.remove(j_idx)
            else:
                # 如果没有找到有效覆盖，添加一个随机组合
                random_group = tuple(sorted(random.sample(all_items, k)))
                solution.append(random_group)
        
        # 移除冗余集合
        if len(solution) > min_groups:
            # 随机尝试删除一些集合
            for _ in range(min(5, len(solution) - min_groups)):
                if len(solution) <= min_groups:
                    break
                idx = random.randint(0, len(solution) - 1)
                removed = solution.pop(idx)
                if not satisfies_condition(solution, n, k, j, s):
                    solution.append(removed)  # 如果删除后不满足条件，恢复
        
        population_chunk.append(solution)
    
    return population_chunk

def create_initial_population_parallel(n, k, population_size, min_groups, max_groups, j, s, n_jobs=-1):
    """并行优化的种群初始化函数"""
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    # 预计算所有组合
    all_k_combinations = list(combinations(range(n), k))
    all_j_combinations = list(combinations(range(n), j))
    
    # 分割种群为多个块，每个进程创建一部分
    chunk_size = max(1, population_size // n_jobs)
    chunks = [(n, k, chunk_size, min_groups, max_groups, j, s, i) 
              for i in range(math.ceil(population_size / chunk_size))]
    
    # 并行创建种群
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        chunks_results = list(executor.map(create_population_chunk, chunks))
    
    # 合并结果
    population = []
    for chunk in chunks_results:
        population.extend(chunk)
    
    # 截断到所需大小
    return population[:population_size]

def crossover(parent1, parent2):
    # Multi-point crossover
    child1, child2 = [], []
    for group1, group2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(group1)
            child2.append(group2)
        else:
            child1.append(group2)
            child2.append(group1)
    
    # Remove duplicates and limit size
    child1 = list(set(child1))[:len(parent1)]
    child2 = list(set(child2))[:len(parent2)]
    return child1, child2

def mutate(solution, n, k, mutation_rate, j, s):
    """变异操作，尝试优化解的覆盖能力"""
    if random.random() < mutation_rate:
        all_items = list(range(n))
        
        # 确定最小组数要求
        min_groups = max(6, j)  # 保持与genetic_algorithm中一致
        
        # 如果当前解小于最小组数，强制增加集合
        if len(solution) < min_groups:
            operation = 'add'
        else:
            # 根据当前解的大小调整操作概率
            if len(solution) <= min_groups + 1:
                # 如果解大小接近最小值，增加添加和替换的概率，降低移除的概率
                operation = random.choice(['add', 'replace', 'replace', 'optimize', 'remove'])
            elif len(solution) >= min_groups + 4:
                # 如果解太大，大幅增加移除和优化的概率
                operation = random.choice(['add', 'remove', 'remove', 'remove', 'replace', 'optimize', 'optimize'])
            else:
                # 正常情况
                operation = random.choice(['add', 'remove', 'replace', 'optimize'])
        
        if operation == 'add' and len(solution) < n:
            # 生成一个新的k元素集合，优先选择覆盖更多j组合的集合
            j_combinations = list(combinations(range(n), j))
            
            # 快速生成几个候选集合
            candidates = []
            for _ in range(5):  # 增加候选集合数量
                new_group = tuple(sorted(random.sample(all_items, k)))
                # 计算简单得分 - 与现有集合的平均重叠度
                overlap_score = 0
                for existing in solution:
                    overlap_score += len(set(new_group) & set(existing))
                overlap_score = overlap_score / (len(solution) + 1) if solution else 0
                # 希望新集合与现有集合有一定重叠但不完全重叠
                candidates.append((new_group, abs(overlap_score - k/2)))
            
            # 选择评分最好的
            solution.append(min(candidates, key=lambda x: x[1])[0])
            
        elif operation == 'remove' and len(solution) > min_groups:
            # 移除多个集合，但要保证满足最小集合数要求
            removals = min(random.randint(1, 2), len(solution) - min_groups)
            
            # 首先尝试找出对覆盖贡献最小的集合
            orig_solution = solution.copy()
            for _ in range(removals):
                if len(solution) <= min_groups:
                    break
                    
                worst_idx = None
                min_impact = float('inf')
                
                # 随机选择几个索引进行评估，而不是遍历所有
                indices = random.sample(range(len(solution)), min(5, len(solution)))
                
                for idx in indices:
                    # 尝试移除这个集合
                    temp_solution = solution.copy()
                    temp_solution.pop(idx)
                    
                    # 检查是否还满足条件
                    if satisfies_condition(temp_solution, n, k, j, s):
                        impact = len(solution) - len(temp_solution)
                        if impact < min_impact:
                            min_impact = impact
                            worst_idx = idx
                
                if worst_idx is not None:
                    solution.pop(worst_idx)
                else:
                    # 如果没有找到可以安全移除的集合，回退到原始解
                    solution = orig_solution
                    break
                
        elif operation == 'replace':
            # 替换一个集合
            idx = random.randint(0, len(solution) - 1)
            new_group = tuple(sorted(random.sample(all_items, k)))
            solution[idx] = new_group
        elif operation == 'optimize':
            # 尝试优化一个现有集合
            if len(solution) > 0:
                idx = random.randint(0, len(solution) - 1)
                current_set = set(solution[idx])
                # 随机替换1-2个元素
                num_changes = random.randint(1, 2)
                modified = current_set.copy()
                
                for _ in range(num_changes):
                    if len(modified) > 0:
                        # 移除一个元素
                        to_remove = random.choice(list(modified))
                        modified.remove(to_remove)
                        # 添加一个新元素
                        available = set(all_items) - modified
                        if available:
                            to_add = random.choice(list(available))
                            modified.add(to_add)
                solution[idx] = tuple(sorted(modified))
        
        # 尝试确保解满足最小组数要求
        while len(solution) < min_groups:
            new_group = tuple(sorted(random.sample(all_items, k)))
            solution.append(new_group)
        
        # 确保解满足基本约束
        if satisfies_condition(solution, n, k, j, s):
            return solution
    return solution

def find_uncovered_combinations(solution, n, k, j, s, min_cover):
    """找出未被充分覆盖的j组合及其s子集"""
    solution_sets = [set(group) for group in solution]
    all_items = list(range(n))
    j_combinations = list(combinations(all_items, j))
    uncovered = []
    
    for j_comb in j_combinations:
        j_set = set(j_comb)
        s_combinations = list(combinations(j_comb, s))
        covered_s_combinations = set()
        
        # 找出已被覆盖的s组合
        for group in solution_sets:
            if len(group & j_set) >= s:
                for s_comb in s_combinations:
                    if set(s_comb).issubset(group):
                        covered_s_combinations.add(tuple(sorted(s_comb)))
        
        # 如果覆盖不足，记录未覆盖的s组合
        if len(covered_s_combinations) < min_cover:
            uncovered_s = [s_comb for s_comb in s_combinations 
                         if tuple(sorted(s_comb)) not in covered_s_combinations]
            uncovered.append((j_comb, uncovered_s))
    
    return uncovered

def compute_solution_fitness(args):
    """计算解的适应度"""
    solution, n, k, j, s, strict_coverage, min_cover = args
    return fitness(solution, n, k, j, s, strict_coverage=strict_coverage, min_cover=min_cover), solution

def local_search(solution, n, k, j, s, min_cover, max_iterations=100, n_jobs=1):
    """局部搜索优化解，支持并行和串行模式"""
    best_solution = solution[:]
    best_fitness = fitness(solution, n, k, j, s, strict_coverage=False, min_cover=min_cover)
    
    # 决定是否使用并行
    use_parallel = should_use_parallel(n, k, j) and n_jobs > 1 and n > 14
    
    # 首先尝试精简解
    min_groups = max(6, j)  # 最小组数要求
    if len(best_solution) > min_groups + 3:
        # 只有当解大小远大于最小要求时，才尝试精简
        pruned_solution = prune_solution(best_solution.copy(), n, k, j, s, min_groups)
        
        # 只有当精简后的解仍然满足条件时才更新
        if satisfies_condition(pruned_solution, n, k, j, s):
            pruned_fitness = fitness(pruned_solution, n, k, j, s, strict_coverage=False, min_cover=min_cover)
            if pruned_fitness > best_fitness:
                best_solution = pruned_solution
                best_fitness = pruned_fitness
    
    for _ in range(max_iterations):
        # 找出未被充分覆盖的组合
        uncovered = find_uncovered_combinations(best_solution, n, k, j, s, min_cover)
        if not uncovered:
            break
        
        # 生成候选解
        candidate_solutions = []
        
        # 1. 尝试为未覆盖的组合构造新集合
        sample_size = min(len(uncovered), 5)
        for j_comb, uncovered_s in random.sample(uncovered, sample_size):
            if uncovered_s:
                s_comb = set(random.choice(uncovered_s))
                # 从j组合中选择额外的元素来构成k元素集合
                remaining = set(j_comb) - s_comb
                if len(remaining) + len(s_comb) >= k:
                    additional = random.sample(list(remaining), k - len(s_comb))
                    new_set = tuple(sorted(list(s_comb) + additional))
                    
                    # 创建新解
                    new_solution = best_solution + [new_set]
                    candidate_solutions.append(new_solution)
        
        # 2. 尝试修改现有集合
        sample_size = min(len(best_solution), 5)
        for idx in random.sample(range(len(best_solution)), sample_size):
            group = best_solution[idx]
            group_set = set(group)
            # 随机替换1-2个元素
            num_changes = random.randint(1, 2)
            modified = group_set.copy()
            
            for _ in range(num_changes):
                if len(modified) > 0:
                    to_remove = random.choice(list(modified))
                    modified.remove(to_remove)
                    available = set(range(n)) - modified
                    if available:
                        to_add = random.choice(list(available))
                        modified.add(to_add)
            
            # 创建新解
            new_solution = best_solution[:]
            new_solution[idx] = tuple(sorted(modified))
            candidate_solutions.append(new_solution)
        
        # 3. 尝试移除冗余集合
        if len(best_solution) > min_groups:
            # 构建一个移除一个集合的候选解
            for i in range(len(best_solution)):
                new_solution = best_solution[:i] + best_solution[i+1:]
                if satisfies_condition(new_solution, n, k, j, s):
                    candidate_solutions.append(new_solution)
                    
                    # 如果去掉一个集合后仍然满足条件，继续尝试去掉第二个
                    if len(new_solution) > min_groups:
                        for j in range(len(new_solution)):
                            ultra_slim = new_solution[:j] + new_solution[j+1:]
                            if satisfies_condition(ultra_slim, n, k, j, s):
                                candidate_solutions.append(ultra_slim)
                                break  # 只添加一个双重精简解
        
        # 如果没有候选解，跳过本次迭代
        if not candidate_solutions:
            continue
        
        # 计算所有候选解的适应度
        if use_parallel:
            # 并行计算适应度
            args_list = [(sol, n, k, j, s, False, min_cover) for sol in candidate_solutions]
            
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                fitness_results = list(executor.map(compute_solution_fitness, args_list))
            
            # 找出最佳候选解
            if fitness_results:
                best_candidate_fitness, best_candidate = max(fitness_results, key=lambda x: x[0])
                if best_candidate_fitness > best_fitness:
                    best_fitness = best_candidate_fitness
                    best_solution = best_candidate
        else:
            # 串行计算适应度
            improved = False
            for candidate in candidate_solutions:
                candidate_fitness = fitness(candidate, n, k, j, s, strict_coverage=False, min_cover=min_cover)
                if candidate_fitness > best_fitness:
                    best_fitness = candidate_fitness
                    best_solution = candidate
                    improved = True
                    break  # 找到改进就提前退出
    
    return best_solution

def prune_solution(solution, n, k, j, s, min_groups):
    """尝试移除解中的冗余集合"""
    # 计算每个集合的"价值"
    solution_sets = [set(group) for group in solution]
    all_items = list(range(n))
    j_combinations = list(combinations(all_items, j))
    
    # 对于每个j组合，计算哪些k集合对它有贡献
    j_coverage = {j_idx: set() for j_idx in range(len(j_combinations))}
    for k_idx, k_set in enumerate(solution_sets):
        for j_idx, j_comb in enumerate(j_combinations):
            j_set = set(j_comb)
            if len(k_set & j_set) >= s:
                j_coverage[j_idx].add(k_idx)
    
    # 对于每个k集合，计算它独自覆盖的j组合数量
    unique_coverage = [0] * len(solution_sets)
    for j_idx, covering_k_indices in j_coverage.items():
        if len(covering_k_indices) == 1:
            k_idx = next(iter(covering_k_indices))
            unique_coverage[k_idx] += 1
    
    # 按照独特覆盖价值排序集合
    sorted_indices = sorted(range(len(solution_sets)), key=lambda i: unique_coverage[i])
    
    # 从最不重要到最重要，尝试移除集合
    pruned = solution.copy()
    for idx in sorted_indices:
        if len(pruned) <= min_groups:
            break  # 保持最小集合数
        
        # 尝试移除当前集合
        test_solution = pruned.copy()
        test_solution.remove(solution[idx])
        
        # 检查移除后是否仍然满足条件
        if satisfies_condition(test_solution, n, k, j, s):
            pruned = test_solution
    
    return pruned

def get_dynamic_mutation_rate(generation, max_generations, base_rate=0.1):
    """动态调整突变率：前期低，后期高"""
    # 前期保持较低的基础突变率，随着代数增加而逐渐增大
    return base_rate * (1 + generation / max_generations * 2)

def get_dynamic_crossover_rate(generation, max_generations, base_rate=0.9):
    """动态调整交叉率：前期大，后期小"""
    # 前期保持较高的交叉率，随着代数增加而逐渐降低
    return base_rate * (1 - generation / max_generations * 0.5)

def should_use_parallel(n, k, j):
    """判断是否应该使用并行计算"""
    if n is None or k is None or j is None:
        return False
    
    # 小规模问题不使用并行
    if n < 10:
        return False
    
    # 计算组合数大小
    k_combinations_size = math.comb(n, k)
    j_combinations_size = math.comb(n, j)
    
    # 当组合数量足够大时使用并行
    return k_combinations_size * j_combinations_size > 10000

def calculate_fitness(solution, all_coverage, j_combinations):
    """计算单个解的适应度"""
    covered_j_indices = set()
    for k_set in solution:
        k_set_frozen = frozenset(k_set)
        if k_set_frozen in all_coverage:
            covered_j_indices.update(all_coverage[k_set_frozen])
    
    # 返回覆盖率和解大小的加权组合
    coverage_ratio = len(covered_j_indices) / len(j_combinations) if j_combinations else 0
    solution_size_penalty = 0.05 * len(solution)  # 每个额外的集合有轻微惩罚
    
    # 返回最终得分 (0-100分)
    return coverage_ratio * 100 - solution_size_penalty

def calculate_fitness_for_chunk(solution_chunk, all_coverage, j_combinations):
    """计算一组解的适应度"""
    result = {}
    for solution in solution_chunk:
        solution_frozen = frozenset(map(frozenset, solution))
        result[solution_frozen] = calculate_fitness(solution, all_coverage, j_combinations)
    return result

def genetic_algorithm(n, k, j, s, population_size=100, generations=100, base_mutation_rate=0.1, base_crossover_rate=0.9, elitism=True, strict_coverage=True, min_cover=1, use_cache=True, n_jobs=4):
    """遗传算法求解集合覆盖问题，支持自适应并行"""
    # 根据问题规模决定是否使用并行
    use_parallel = should_use_parallel(n, k, j)
    actual_n_jobs = n_jobs if use_parallel else 1
    
    # 对于中等规模问题(n=12-14)，减少并行进程数以避免开销
    if use_parallel and 12 <= n <= 14:
        actual_n_jobs = min(6, actual_n_jobs)
        print(f"对于中等规模问题(n={n})，调整并行进程数为 {actual_n_jobs}")
    
    if use_parallel:
        print(f"问题规模足够大，使用 {actual_n_jobs} 个CPU核心进行并行计算...")
    else:
        print("问题规模较小，使用串行计算以避免并行开销...")
    
    total_start_time = time.time()
    
    # Define min_groups and max_groups - 修改最小组数
    # 对于n=12, k=6, j=6, s=4的问题，理论上至少需要6个集合
    min_groups = max(6, j)  # 最小组数应该至少为j或6，取较大者
    max_groups = n * 2      # 最大组数可以是元素数的两倍
    
    # 预计算覆盖关系，支持缓存
    precompute_start_time = time.time()
    print("正在预计算覆盖关系...")
    coverage_lookup, all_k_sets, all_j_combinations = precompute_coverage_relations(n, k, j, s, use_cache, actual_n_jobs)
    precompute_time = time.time() - precompute_start_time
    print(f"预计算完成，用时: {precompute_time:.2f}秒")
    
    # 初始化种群
    init_start_time = time.time()
    if use_parallel and n > 13:  # 只有n>13时才使用并行初始化
        population = create_initial_population_parallel(n, k, population_size, min_groups, max_groups, j, s, actual_n_jobs)
    else:
        # 对于中小规模问题，使用串行版本的初始化更高效
        print("使用串行初始化以避免小规模问题的并行开销...")
        population = create_initial_population_serial(n, k, population_size, min_groups, max_groups, j, s)
    init_time = time.time() - init_start_time
    print(f"初始化种群完成，用时: {init_time:.2f}秒")
    
    best_solution = None
    best_fitness = float('-inf')
    generations_without_improvement = 0
    
    print("\n遗传算法进化过程:")
    print("代数\t最佳适应度\t突变率\t交叉率\t解大小\t本代用时(秒)")
    
    evolution_times = []  # 记录每代的进化时间
    
    # 创建适应度缓存，减少重复计算
    fitness_cache = {}
    
    for generation in range(generations):
        generation_start_time = time.time()
        
        # 动态调整突变率和交叉率
        current_mutation_rate = get_dynamic_mutation_rate(generation, generations, base_mutation_rate)
        current_crossover_rate = get_dynamic_crossover_rate(generation, generations, base_crossover_rate)
        
        # 评估适应度 - 自适应并行/串行
        fitness_start_time = time.time()
        if use_parallel and n > 13:  # 仅在较大规模问题上使用并行适应度计算
            # 修复函数调用，确保传递正确的参数
            fitness_scores_map = parallel_fitness_calculation(
                population, 
                coverage_lookup, 
                all_j_combinations, 
                n=n, 
                k=k, 
                j=j, 
                n_jobs=actual_n_jobs
            )
            # 转换结果格式以与后续代码兼容
            fitness_scores = []
            for solution in population:
                solution_frozen = frozenset(map(frozenset, solution))
                if solution_frozen in fitness_scores_map:
                    fitness_scores.append((fitness_scores_map[solution_frozen], solution))
                else:
                    # 如果在结果中找不到，重新计算
                    fitness_value = fitness_with_lookup(solution, coverage_lookup, all_k_sets, all_j_combinations, s, strict_coverage, min_cover)
                    fitness_scores.append((fitness_value, solution))
        else:
            # 使用缓存优化串行版本
            fitness_scores = []
            for solution in population:
                # 生成缓存键
                solution_key = tuple(sorted(tuple(sorted(group)) for group in solution))
                if solution_key in fitness_cache:
                    fitness_value = fitness_cache[solution_key]
                else:
                    fitness_value = fitness_with_lookup(solution, coverage_lookup, all_k_sets, all_j_combinations, s, strict_coverage, min_cover)
                    fitness_cache[solution_key] = fitness_value
                fitness_scores.append((fitness_value, solution))
        
        fitness_scores.sort(reverse=True)
        fitness_time = time.time() - fitness_start_time
        
        # 更新最佳解，强制保证最小组数
        current_best_solution = fitness_scores[0][1][:]
        current_best_fitness = fitness_scores[0][0]
        
        # 检查当前最佳解的大小是否满足最小组数要求
        if len(current_best_solution) < min_groups:
            # 如果解太小，跳过更新最佳解
            print(f"当前最佳解大小({len(current_best_solution)})小于最小要求({min_groups})，跳过更新")
        else:
            # 只有当解大小满足要求时才更新最佳解
            if best_solution is None or current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = current_best_solution
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
        
        # 选择 - 使用锦标赛选择
        population_list = [sol for _, sol in fitness_scores]
        fitness_list = [score for score, _ in fitness_scores]
        selected = tournament_selection(population_list, fitness_list, population_size // 2)
        
        # 保留精英
        if elitism:
            elite_count = max(1, int(population_size * 0.1))
            elite = [fitness_scores[i][1][:] for i in range(elite_count)]
        
        # 交叉和变异 - 自适应并行/串行
        if use_parallel and n > 14:  # 只有n>14时使用并行交叉变异
            population = parallel_crossover_and_mutation(selected, n, k, j, s, current_crossover_rate, current_mutation_rate, actual_n_jobs)
        else:
            # 串行交叉和变异
            new_population = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    parent1, parent2 = selected[i], selected[i+1]
                    if random.random() < current_crossover_rate:
                        child1, child2 = crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1[:], parent2[:]
                    
                    child1 = mutate(child1, n, k, current_mutation_rate, j, s)
                    child2 = mutate(child2, n, k, current_mutation_rate, j, s)
                    
                    new_population.append(child1)
                    new_population.append(child2)
            
            # 处理奇数情况
            if len(selected) % 2 == 1:
                last = selected[-1][:]
                last = mutate(last, n, k, current_mutation_rate, j, s)
                new_population.append(last)
                
            population = new_population
        
        # 添加精英
        if elitism:
            population.extend(elite)
            population = population[:population_size]  # 确保种群大小不变
        
        # 每10代执行一次局部搜索，n=13时减少到每15代一次以降低开销
        local_search_interval = 15 if 12 <= n <= 14 else 10
        if generation % local_search_interval == 0 and best_solution:
            # 对最佳解进行局部搜索优化
            improved_solution = local_search(best_solution, n, k, j, s, min_cover, n_jobs=actual_n_jobs)
            
            # 计算改进后的适应度
            solution_key = tuple(sorted(tuple(sorted(group)) for group in improved_solution))
            if solution_key in fitness_cache:
                improved_fitness = fitness_cache[solution_key]
            else:
                improved_fitness = fitness_with_lookup(improved_solution, coverage_lookup, all_k_sets, all_j_combinations, s, strict_coverage, min_cover)
                fitness_cache[solution_key] = improved_fitness
            
            if improved_fitness > best_fitness and len(improved_solution) >= min_groups:
                best_fitness = improved_fitness
                best_solution = improved_solution
                # 将优化后的解添加到种群中
                population[0] = improved_solution
        
        generation_time = time.time() - generation_start_time
        evolution_times.append(generation_time)
        
        # 每10代清理一次缓存，避免内存占用过高
        if generation % 10 == 0 and len(fitness_cache) > 10000:
            fitness_cache = {k: v for k, v in sorted(fitness_cache.items(), key=lambda item: item[1], reverse=True)[:1000]}
        
        # 输出本代统计信息
        print(f"{generation}\t{fitness_scores[0][0]:.2f}\t{current_mutation_rate:.2f}\t"
              f"{current_crossover_rate:.2f}\t{len(fitness_scores[0][1])}\t{generation_time:.2f}")
        
        # 修改终止条件：如果连续40代没有改进，提前终止（增加容忍度）
        if generations_without_improvement >= 40:
            print("连续40代没有改进，提前终止算法")
            break
    
    total_time = time.time() - total_start_time
    
    # 输出总体统计信息
    print("\n算法执行统计:")
    print(f"预计算时间: {precompute_time:.2f}秒")
    print(f"初始化时间: {init_time:.2f}秒")
    print(f"平均每代时间: {sum(evolution_times)/len(evolution_times):.2f}秒")
    print(f"总执行时间: {total_time:.2f}秒")
    print(f"并行模式: {'开启' if use_parallel else '关闭'}")
    
    # 如果最终解不满足覆盖要求，尝试再次使用局部搜索改进
    if best_solution and not satisfies_condition(best_solution, n, k, j, s, strict_coverage, min_cover):
        print("最终解不满足覆盖要求，尝试额外局部搜索...")
        final_solution = local_search(best_solution, n, k, j, s, min_cover, max_iterations=200)
        
        # 检查改进后的解是否满足条件
        if satisfies_condition(final_solution, n, k, j, s, strict_coverage, min_cover):
            print("额外局部搜索成功找到满足条件的解！")
            best_solution = final_solution
        else:
            print("额外局部搜索未能找到满足条件的解。")
    
    return best_solution  # Return the best solution found during evolution

def tournament_selection(population, fitness_scores, num_selected):
    """锦标赛选择，选出表现最好的个体"""
    selected = []
    for _ in range(num_selected):
        # 随机选择3个个体进行锦标赛
        indices = random.sample(range(len(population)), min(3, len(population)))
        tournament = [(fitness_scores[i], population[i]) for i in indices]
        winner = max(tournament, key=lambda x: x[0])
        selected.append(winner[1])
    return selected

def compute_fitness_chunk(args):
    """计算一个种群块的适应度"""
    chunk, coverage_lookup, all_k_sets, all_j_combinations, s, strict_coverage, min_cover = args
    return [(fitness_with_lookup(solution, coverage_lookup, all_k_sets, all_j_combinations, s, strict_coverage, min_cover), solution) for solution in chunk]

def process_pair(args):
    """处理一对父代的交叉和变异"""
    parent1, parent2, do_crossover, n, k, j, s, mutation_rate = args
    if do_crossover:
        child1, child2 = crossover(parent1, parent2)
    else:
        child1, child2 = parent1[:], parent2[:]
    
    # 变异操作
    child1 = mutate(child1, n, k, mutation_rate, j, s)
    child2 = mutate(child2, n, k, mutation_rate, j, s)
    
    return child1, child2

def parallel_crossover_and_mutation(selected, n, k, j, s, crossover_rate, mutation_rate, n_jobs=-1):
    """并行执行交叉和变异操作"""
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    # 生成所有需要进行交叉操作的对
    pairs = []
    for i in range(0, len(selected), 2):
        if i + 1 < len(selected):
            pairs.append((selected[i], selected[i+1], random.random() < crossover_rate, n, k, j, s, mutation_rate))
    
    # 并行执行交叉和变异
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(process_pair, pairs))
    
    # 合并结果
    new_population = []
    for child1, child2 in results:
        new_population.append(child1)
        new_population.append(child2)
    
    # 如果原始选择数量是奇数，需要处理最后一个
    if len(selected) % 2 == 1:
        last = selected[-1][:]
        last = mutate(last, n, k, mutation_rate, j, s)
        new_population.append(last)
    
    return new_population

def create_initial_population_serial(n, k, population_size, min_groups, max_groups, j, s):
    """串行版的种群初始化函数"""
    population = []
    
    # 预计算所有组合
    all_items = list(range(n))
    all_k_combinations = list(combinations(all_items, k))
    all_j_combinations = list(combinations(all_items, j))
    
    # 计算每个k组合的覆盖能力
    coverage_scores = []
    for k_set in all_k_combinations:
        k_set = set(k_set)
        score = 0
        for j_comb in all_j_combinations:
            if len(set(j_comb) & k_set) >= s:
                score += 1
        coverage_scores.append(score)
    
    # 选择覆盖能力最强的前50%组合
    sorted_indices = sorted(range(len(coverage_scores)), key=lambda i: coverage_scores[i], reverse=True)
    top_indices = sorted_indices[:len(sorted_indices)//2]
    top_combinations = [all_k_combinations[i] for i in top_indices]
    
    # 创建初始种群
    for _ in range(population_size):
        solution = []
        uncovered = set(range(len(all_j_combinations)))
        
        # 贪心地添加覆盖最多未覆盖j组合的k集合
        while uncovered and len(solution) < max_groups:
            best_combination = None
            best_coverage = -1
            
            # 随机选择一些候选集合来减少计算量
            candidates = random.sample(top_combinations, min(20, len(top_combinations)))
            
            for k_set in candidates:
                k_set = set(k_set)
                coverage = 0
                for j_idx in list(uncovered):
                    j_comb = all_j_combinations[j_idx]
                    if len(set(j_comb) & k_set) >= s:
                        coverage += 1
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_combination = tuple(sorted(k_set))
            
            if best_combination and best_coverage > 0:
                solution.append(best_combination)
                # 更新未覆盖列表
                for j_idx in list(uncovered):
                    j_comb = all_j_combinations[j_idx]
                    if len(set(j_comb) & set(best_combination)) >= s:
                        uncovered.remove(j_idx)
            else:
                # 如果没有找到有效覆盖，添加一个随机组合
                random_group = tuple(sorted(random.sample(all_items, k)))
                solution.append(random_group)
        
        # 对于n=12, k=6, j=6, s=4，确保解至少有6个集合
        while len(solution) < min_groups:
            # 添加随机集合直到达到最小要求
            random_group = tuple(sorted(random.sample(all_items, k)))
            if random_group not in solution:  # 避免重复
                solution.append(random_group)
        
        # 移除冗余集合，但确保解大小不小于min_groups
        if len(solution) > min_groups + 2:  # 保留更多集合，增加冗余
            # 随机尝试删除一些集合
            for _ in range(min(3, len(solution) - min_groups)):
                if len(solution) <= min_groups:
                    break
                idx = random.randint(0, len(solution) - 1)
                removed = solution.pop(idx)
                # 检查删除后是否仍满足覆盖条件
                if not satisfies_condition(solution, n, k, j, s):
                    solution.append(removed)  # 如果删除后不满足条件，恢复
        
        # 确保最终解满足最小组数要求
        assert len(solution) >= min_groups, f"初始化解大小({len(solution)})小于最小要求({min_groups})"
        
        population.append(solution)
    
    return population

def get_optimal_processes(n, available_cores=14, memory_gb=24):
    """
    根据问题规模和系统配置智能分配进程数
    专为Mac M4 Pro (14核心, 24GB内存)优化
    
    参数:
        n: 问题规模
        available_cores: 系统可用核心数
        memory_gb: 系统内存(GB)
    
    返回:
        建议的进程数
    """
    # 始终保留系统使用的核心
    reserved_cores = 2  # 保留给系统使用
    
    # 为n=13的中等规模问题特别优化
    if 12 <= n <= 14:
        # 中等规模问题会遇到并行开销的瓶颈，增加系统保留核心数
        reserved_cores = 4
        # 减少为4个核心，进一步降低并行开销
        max_cores = min(4, available_cores - reserved_cores)
        print(f"针对n={n}的中等规模问题优化，保留{reserved_cores}个核心给系统，最多使用{max_cores}个核心")
        return max_cores
    
    # 根据问题规模调整使用的核心数
    if n <= 11:  # 小规模问题
        max_cores = 4  # 小规模问题使用较少核心
    elif n <= 16:  # 中等规模问题
        max_cores = 8  # 使用较多核心但不是全部
    elif n <= 20:  # 大规模问题
        max_cores = 10  # 使用更多核心
    else:  # 非常大的问题
        max_cores = available_cores - reserved_cores  # 使用几乎所有可用核心
    
    # 确保n>=16时至少使用8个核心
    if n >= 16:
        max_cores = max(max_cores, 8)
    
    # 估计每个进程的内存使用量 (GB)
    memory_per_process = 0.5  # 基础值
    # 大规模问题内存使用会增加
    if n > 16:
        memory_per_process = 1.0
    if n > 20:
        memory_per_process = 2.0
    
    # 确保内存使用不超过系统内存的80%
    max_processes_by_memory = int(memory_gb * 0.8 / memory_per_process)
    
    # 取两者的最小值确保不会过载系统
    optimal_processes = min(max_cores, max_processes_by_memory)
    
    # 确保至少使用1个核心
    return max(1, optimal_processes)

def main(only_count=False, show_sets=True, use_cache=True, n_jobs=None, test_case=None):
    """主函数，控制整个算法的执行流程
    
    参数:
    only_count: 是否只输出最小集合数量，不显示具体集合
    show_sets: 是否显示具体集合
    use_cache: 是否使用缓存
    n_jobs: 并行处理的进程数量，None表示自动决定
    test_case: 特定测试用例，None则使用默认测试用例
    """
    # 默认测试用例
    if test_case is None:
        test_case = {"n": 16, "k": 6, "j": 6, "s": 4, "strict_coverage": False, "min_cover": 1}
    
    n = test_case['n']
    # 如果没有指定核心数，根据问题规模自动分配
    if n_jobs is None:
        optimal_jobs = get_optimal_processes(n, available_cores=14, memory_gb=24)
        print(f"为问题规模 n={n} 自动分配 {optimal_jobs} 个CPU核心...")
    else:
        optimal_jobs = n_jobs
        print(f"使用指定的 {optimal_jobs} 个CPU核心进行计算...")
    
    print(f"\n{'='*50}")
    print(f"测试用例:")
    print(f"参数: n={test_case['n']}, k={test_case['k']}, j={test_case['j']}, s={test_case['s']}")
    print(f"覆盖模式: {'严格覆盖' if test_case['strict_coverage'] else '宽松覆盖'}")
    if not test_case['strict_coverage']:
        print(f"最小覆盖数: {test_case['min_cover']}")
    
    # 测量性能 - 预计算时间
    start_time = time.time()
    
    # 使用遗传算法生成初始解
    print("使用遗传算法生成初始解...")
    initial_solution = generate_initial_solution_ga(
        test_case['n'], test_case['k'], test_case['j'], test_case['s'],
        population_size=100,  # 种群大小
        generations=100,     # 迭代次数
        use_cache=use_cache,  # 使用缓存
        n_jobs=optimal_jobs  # 使用优化的核心数
    )
    
    elapsed_time = time.time() - start_time
    print(f"遗传算法找到的初始解包含 {len(initial_solution)} 个集合")
    print(f"总运行时间: {elapsed_time:.2f}秒")
    
    # 如果n大于8，直接使用遗传算法结果
    if test_case['n'] > 8:
        print(f"\n由于n={test_case['n']} > 8，跳过ILP求解，直接使用遗传算法结果")
        if show_sets:
            print("\n遗传算法找到的集合:")
            for i, subset in enumerate(initial_solution, 1):
                # 将数字转换为字母表示
                subset_letters = [chr(ord('A') + item) for item in subset]
                print(f"{i}. {','.join(sorted(subset_letters))}")
        return initial_solution
    
    # 对于小规模问题 (n<=8)，使用ILP求解
    # 构建ILP问题
    solver, x, k_subsets, j_subsets, s_subsets = construct_ilp_problem(
        test_case['n'], test_case['k'], test_case['j'], test_case['s'],
        test_case['strict_coverage'], test_case['min_cover']
    )
    
    # 设置SCIP求解器参数以优化性能
    solver.SetTimeLimit(1800000)  # 30分钟，单位是毫秒
    solver.SetNumThreads(optimal_jobs)  # 使用优化的核心数
    solver.EnableOutput()  # 启用求解器输出
    
    # 求解问题
    print("开始求解...")
    status = solver.Solve()
    
    # 输出结果
    if status == pywraplp.Solver.OPTIMAL:
        print("找到最优解")
        count = sum(1 for i in range(len(k_subsets)) if x[i].solution_value() == 1)
        print(f"最小数量: {count}")
        
        if not only_count and show_sets:
            # 输出完整结果
            selected_subsets = []
            for i in range(len(k_subsets)):
                if x[i].solution_value() == 1:
                    selected_subsets.append(k_subsets[i])
            
            print("选择的集合:")
            for i, subset in enumerate(selected_subsets, 1):
                print(f"{i}. {','.join(sorted(subset))}")
            return selected_subsets
    else:
        print(f"求解状态: {status}")
        if status == pywraplp.Solver.INFEASIBLE:
            print("问题无可行解")
        elif status == pywraplp.Solver.ABNORMAL:
            print("求解器异常终止")
    
    return initial_solution  # 如果ILP求解失败，返回遗传算法解

if __name__ == "__main__":
    # 性能优化建议
    print("\n============ 性能优化建议 ============")
    print("1. 并行计算速度: n=10-12通常串行更快，n=16+并行更有优势")
    print("2. 中等规模问题(n=13-14): 需要特殊处理并限制并行度")
    print("3. 内存使用: 预计算缓存在重复运行相同参数时有优势")
    print("4. 运行不同规模问题(n=10,12,14,16)以比较性能")
    print("==============================================\n")
    
    # 测试用例，可以通过修改这个字典来测试不同参数
    test_case = {"n": 12, "k": 6, "j": 6, "s": 4, "strict_coverage": False, "min_cover": 1}
    
    # 运行主函数
    solution = main(test_case=test_case, use_cache=True)
