from ortools.linear_solver import pywraplp
from itertools import combinations
import os
import numpy as np
import random
import time  # 添加time模块导入
import pickle  # 添加pickle模块导入
import hashlib  # 添加hashlib模块导入

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

def construct_ilp_problem(n, k, j, s, m, strict_coverage=True, min_cover=1):
    """
    构建ILP问题

    参数:
    n: 样本数量
    k: k元素子集大小
    j: j元素子集大小
    s: s元素子集大小
    m: 整数范围上限，从1-m中选择n个整数作为样本
    strict_coverage: 是否使用严格覆盖模式
    min_cover: 宽松覆盖模式下，每个j元素子集至少需要被覆盖的s元素子集数量
    """
    # 生成样本集合（从1-m的整数中选择n个整数）
    samples = sorted(random.sample(range(1, m+1), n))
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

    return solver, x, k_subsets, j_subsets, s_subsets, samples

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

def precompute_coverage_relations(n, k, j, s, m=None, use_cache=True):
    """预计算k集合和j组合的覆盖关系，使用优化的矩阵运算和数据结构"""
    # 检查缓存
    cache_filename = get_cache_filename(n, k, j, s)
    if use_cache:
        cache_data = load_coverage_cache(cache_filename)
        if cache_data:
            return cache_data

    print(f"开始预计算覆盖关系 (n={n}, k={k}, j={j}, s={s})...")

    if m is not None:
        # 如果提供了m，则从1-m中随机选择n个数作为样本
        import random
        samples = sorted(random.sample(range(1, m+1), n))
        print(f"使用随机样本: {samples}")
        
        # 生成所有组合基于随机样本
        all_k_sets = list(combinations(samples, k))
        all_j_combinations = list(combinations(samples, j))
        
        # 创建样本到索引的映射
        sample_to_idx = {v: i for i, v in enumerate(samples)}
        
        # 创建位集合表示
        k_sets_bits = np.zeros((len(all_k_sets), n), dtype=bool)
        for i, k_set in enumerate(all_k_sets):
            for elem in k_set:
                k_sets_bits[i, sample_to_idx[elem]] = True
        
        j_sets_bits = np.zeros((len(all_j_combinations), n), dtype=bool)
        for i, j_comb in enumerate(all_j_combinations):
            for elem in j_comb:
                j_sets_bits[i, sample_to_idx[elem]] = True
    else:
        # 使用默认的索引范围
        all_k_sets = list(combinations(range(n), k))
        all_j_combinations = list(combinations(range(n), j))
        
        # 使用位集合表示每个组合以提高集合运算效率
        k_sets_bits = np.zeros((len(all_k_sets), n), dtype=bool)
        for i, k_set in enumerate(all_k_sets):
            k_sets_bits[i, list(k_set)] = True
        
        j_sets_bits = np.zeros((len(all_j_combinations), n), dtype=bool)
        for i, j_comb in enumerate(all_j_combinations):
            j_sets_bits[i, list(j_comb)] = True

    # 使用矩阵运算计算交集大小
    # 为避免大规模矩阵计算导致内存问题，使用分块处理
    block_size = 1000  # 根据内存情况调整

    # 创建高效的数据结构存储结果
    coverage_lookup = {}

    for k_idx in range(len(all_k_sets)):
        coverage_lookup[k_idx] = {}
        k_bit = k_sets_bits[k_idx]

        # 分块处理j组合
        for j_block_start in range(0, len(all_j_combinations), block_size):
            j_block_end = min(j_block_start + block_size, len(all_j_combinations))
            j_block = j_sets_bits[j_block_start:j_block_end]

            # 使用矩阵运算计算交集
            # k_bit 是一个长度为n的布尔数组，j_block是多个j组合的布尔数组
            intersections = np.logical_and(k_bit, j_block).sum(axis=1)

            # 处理每个j组合
            for j_offset, intersection_size in enumerate(intersections):
                j_idx = j_block_start + j_offset
                j_comb = all_j_combinations[j_idx]

                # 只有当交集大小满足条件时才计算s子集
                if intersection_size >= s:
                    # 计算该k集合覆盖的s子集
                    # 使用迭代器避免一次性生成所有组合
                    s_combs = combinations(j_comb, s)
                    covered_s = set()

                    # 只考虑那些可能被k_set覆盖的s组合
                    for s_comb in s_combs:
                        # 使用位运算优化子集判断
                        if all(item in all_k_sets[k_idx] for item in s_comb):
                            covered_s.add(tuple(sorted(s_comb)))

                    coverage_lookup[k_idx][j_idx] = covered_s
                else:
                    coverage_lookup[k_idx][j_idx] = set()  # 没有覆盖任何s子集

    # 使用更高效的数据结构优化
    # 转换为嵌套字典，只保存有覆盖的关系，减少内存使用
    optimized_lookup = {}
    for k_idx in coverage_lookup:
        non_empty = {j_idx: s_sets for j_idx, s_sets in coverage_lookup[k_idx].items() if s_sets}
        if non_empty:
            optimized_lookup[k_idx] = non_empty

    # 保存缓存
    if use_cache:
        result = (optimized_lookup, all_k_sets, all_j_combinations)
        save_coverage_cache(result, cache_filename)

    return optimized_lookup, all_k_sets, all_j_combinations

def fitness_with_lookup(solution, coverage_lookup, all_k_sets, all_j_combinations, s, strict_coverage=True, min_cover=1):
    """使用优化的查找表计算适应度"""
    total_coverage_quality = 0
    covered_j_count = 0
    
    # 将solution中的k集合转换为索引
    solution_indices = []
    for k_set in solution:
        # 处理可能的类型差异：确保k_set和all_k_sets中的元素格式一致
        k_set_tuple = tuple(sorted(k_set)) if not isinstance(k_set, tuple) else k_set
        # 找到匹配的索引
        found = False
        for idx, reference_set in enumerate(all_k_sets):
            if set(k_set_tuple) == set(reference_set):
                solution_indices.append(idx)
                found = True
                break
        if not found:
            # 如果找不到匹配项，忽略这个集合
            # 这可能是因为使用随机样本导致的格式差异
            pass
    
    # 如果没有可用的索引，返回低分数
    if not solution_indices:
        return -1000
    
    # 预计算每个j组合被覆盖的s子集
    j_covered_s_sets = {}
    
    # 使用向量化操作加速计算
    for j_idx in range(len(all_j_combinations)):
        covered_s = set()
        # 只考虑有覆盖关系的k集合
        for k_idx in solution_indices:
            if k_idx in coverage_lookup and j_idx in coverage_lookup[k_idx]:
                covered_s.update(coverage_lookup[k_idx][j_idx])
        
        j_covered_s_sets[j_idx] = covered_s
    
    # 计算覆盖质量
    for j_idx, j_comb in enumerate(all_j_combinations):
        covered_s = j_covered_s_sets[j_idx]
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
    
    # 计算基础适应度
    coverage_score = total_coverage_quality / len(all_j_combinations)
    
    # 增加对解数量的惩罚
    size_penalty = len(solution) ** 1.5 * 0.1
    
    # 对于宽松覆盖模式，增加额外的覆盖质量奖励
    if not strict_coverage:
        coverage_bonus = (covered_j_count / len(all_j_combinations)) * 0.3
    else:
        coverage_bonus = 0
    
    return coverage_score * 100 - size_penalty + coverage_bonus

def generate_initial_solution_ga(n, k, j, s, m, population_size=50, generations=100, use_cache=True, use_intelligent_mutation=True, use_greedy_crossover=True):
    """使用遗传算法生成初始解，支持缓存"""
    # 生成样本集合（从1-m的整数中选择n个整数）
    import random
    samples = sorted(random.sample(range(1, m+1), n))
    print(f"遗传算法使用的样本集合: {samples}")
    
    # 使用GA_version_2中的遗传算法
    # 由于遗传算法使用的是索引0到n-1，我们需要将样本映射到这些索引
    sample_to_index = {num: i for i, num in enumerate(samples)}
    index_to_sample = {i: num for i, num in enumerate(samples)}
    
    # 使用索引运行遗传算法
    solution = genetic_algorithm(n, k, j, s, m,
                                population_size=population_size,
                                generations=generations,
                                base_mutation_rate=0.1,
                                base_crossover_rate=0.9,
                                elitism=True,
                                strict_coverage=False,
                                min_cover=1,
                                use_cache=use_cache,
                                use_intelligent_mutation=use_intelligent_mutation,
                                use_greedy_crossover=use_greedy_crossover)

    # 检查解的覆盖情况
    satisfies = satisfies_condition(solution, n, k, j, s, strict_coverage=False, min_cover=1)

    # 计算并输出最终解的适应度
    final_fitness = fitness(solution, n, k, j, s, strict_coverage=False, min_cover=1)
    print(f"遗传算法找到的解的适应度: {final_fitness:.2f}")
    print(f"覆盖状态: {'完全覆盖' if satisfies else '部分覆盖'}")

    # 将索引解转换为实际样本值的解
    solution_with_samples = []
    for subset in solution:
        # 防止索引越界：确保每个索引都在index_to_sample中
        subset_samples = []
        for idx in subset:
            # 确保索引在有效范围内
            if idx < 0 or idx >= n:
                # 如果索引无效，随机选择一个有效索引
                valid_idx = random.randint(0, n-1)
                subset_samples.append(index_to_sample[valid_idx])
            else:
                subset_samples.append(index_to_sample[idx])
        solution_with_samples.append(set(subset_samples))
    
    # 输出结果
    print("\n遗传算法找到的集合:")
    for i, subset in enumerate(solution_with_samples, 1):
        print(f"{i}. {','.join(map(str, sorted(subset)))}")

    return solution_with_samples, samples

def generate_combinations(items, k):
    return list(combinations(items, k))

def check_coverage(group, j_combination, s):
    # Check if the group contains at least s items from the j_combination
    common_elements = set(group) & set(j_combination)
    return len(common_elements) >= s

def satisfies_condition(solution, n, k, j, s, strict_coverage=True, min_cover=1):
    """检查解是否满足约束条件，支持宽松和严格覆盖模式"""
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
                    return False
        else:
            # 宽松覆盖模式：检查每个j组合中是否有足够多的s组合被覆盖
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
                return False
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
                total_coverage_quality += 1 + extra_coverage * 0.1  # 降低超额覆盖的奖励
            else:
                # 对接近目标的解给予部分奖励
                coverage_ratio = s_covered / min_cover
                total_coverage_quality += coverage_ratio * 0.5
    
    # 计算基础适应度
    coverage_score = total_coverage_quality / len(j_combinations)
    
    # 增加对解数量的惩罚，使用指数惩罚
    size_penalty = len(solution) ** 1.5 * 0.1  # 使用指数惩罚，使大解受到更严重的惩罚
    
    # 对于宽松覆盖模式，增加额外的覆盖质量奖励
    if not strict_coverage:
        coverage_bonus = (covered_j_count / len(j_combinations)) * 0.3  # 降低覆盖奖励的权重
    else:
        coverage_bonus = 0
    
    return coverage_score * 100 - size_penalty + coverage_bonus

def create_initial_population(n, k, population_size, min_groups, max_groups, j, s):
    """高性能的种群初始化函数"""
    population = []
    
    # 预计算所有组合并转换为numpy数组以提高性能
    all_k_combinations = np.array(list(combinations(range(n), k)))
    all_j_combinations = np.array(list(combinations(range(n), j)))
    
    # 预计算覆盖关系矩阵
    coverage_matrix = np.zeros((len(all_k_combinations), len(all_j_combinations)), dtype=bool)
    for k_idx, k_set in enumerate(all_k_combinations):
        k_set = set(k_set)
        for j_idx, j_comb in enumerate(all_j_combinations):
            coverage_matrix[k_idx, j_idx] = len(set(j_comb) & k_set) >= s
    
    # 计算每个k组合的覆盖能力
    coverage_scores = np.sum(coverage_matrix, axis=1)
    
    # 选择覆盖能力最强的前50%组合
    top_indices = np.argsort(coverage_scores)[-len(coverage_scores)//2:]
    top_combinations = all_k_combinations[top_indices]
    top_coverage_matrix = coverage_matrix[top_indices]
    
    for _ in range(population_size):
        solution = []
        uncovered = np.ones(len(all_j_combinations), dtype=bool)
        
        # 使用向量化操作构建解
        while np.any(uncovered) and len(solution) < max_groups:
            # 计算每个候选集合对未覆盖组合的贡献
            contributions = np.sum(top_coverage_matrix[:, uncovered], axis=1)
            best_idx = np.random.choice(np.where(contributions == contributions.max())[0])
            
            # 添加最佳集合
            solution.append(tuple(top_combinations[best_idx]))
            
            # 更新未覆盖状态
            uncovered &= ~top_coverage_matrix[best_idx]
        
        # 优化解的大小
        if len(solution) > min_groups + 2:
            # 随机移除一些集合，但保持基本覆盖
            indices = list(range(len(solution)))
            random.shuffle(indices)
            for idx in indices[min_groups+2:]:
                temp_solution = solution[:idx] + solution[idx+1:]
                if satisfies_condition(temp_solution, n, k, j, s):
                    solution = temp_solution
                    break
        
        population.append(solution)
    
    return population

def greedy_crossover(parent1, parent2, coverage_matrix, all_k_sets, all_j_combinations):
    """
    基于贪心策略的智能交叉，优先选择覆盖能力最强的基因
    
    参数:
    - parent1, parent2: 父代解
    - coverage_matrix: 覆盖矩阵
    - all_k_sets: 所有k子集
    - all_j_combinations: 所有j组合
    
    返回:
    - 子代解
    """
    # 将parent1和parent2中的k集合转换为索引
    parent1_indices = []
    parent2_indices = []
    
    # 处理parent1
    for k_set in parent1:
        # 处理可能的类型差异
        k_set_tuple = tuple(sorted(k_set)) if not isinstance(k_set, tuple) else k_set
        # 找到匹配的索引
        found = False
        for idx, reference_set in enumerate(all_k_sets):
            if set(k_set_tuple) == set(reference_set):
                parent1_indices.append(idx)
                found = True
                break
        # 如果没找到匹配的集合，添加一个有效的索引（避免后续错误）
        if not found and len(all_k_sets) > 0:
            parent1_indices.append(0)  # 添加第一个k集合的索引
    
    # 处理parent2
    for k_set in parent2:
        # 处理可能的类型差异
        k_set_tuple = tuple(sorted(k_set)) if not isinstance(k_set, tuple) else k_set
        # 找到匹配的索引
        found = False
        for idx, reference_set in enumerate(all_k_sets):
            if set(k_set_tuple) == set(reference_set):
                parent2_indices.append(idx)
                found = True
                break
        # 如果没找到匹配的集合，添加一个有效的索引
        if not found and len(all_k_sets) > 0:
            parent2_indices.append(0)  # 添加第一个k集合的索引
    
    # 合并父代基因
    combined_indices = list(set(parent1_indices + parent2_indices))
    
    # 如果索引为空，返回父代中较短的一个
    if not combined_indices:
        return parent1 if len(parent1) <= len(parent2) else parent2
    
    # 确保所有索引都在有效范围内
    valid_indices = [idx for idx in combined_indices if 0 <= idx < len(coverage_matrix)]
    if not valid_indices:
        return parent1 if len(parent1) <= len(parent2) else parent2
    
    # 使用矩阵快速评估每个k集合的覆盖价值
    coverage_values = np.sum(coverage_matrix[valid_indices, :], axis=1)
    
    # 按覆盖价值排序
    ranked_indices = [valid_indices[i] for i in np.argsort(-coverage_values)]
    
    # 贪心构建子代
    covered = np.zeros(coverage_matrix.shape[1], dtype=bool)
    child_indices = []
    
    for k_idx in ranked_indices:
        # 计算新增的覆盖
        new_covered = covered | coverage_matrix[k_idx, :]
        if np.any(new_covered > covered):  # 检查是否增加覆盖
            child_indices.append(k_idx)
            covered = new_covered
            if np.all(covered):  # 提前终止
                break
    
    # 将索引转换回k集合
    child = []
    for idx in child_indices:
        if 0 <= idx < len(all_k_sets):
            child.append(all_k_sets[idx])
    
    # 如果子代为空，返回父代中较短的一个
    if not child:
        return parent1 if len(parent1) <= len(parent2) else parent2
    
    return child

def crossover(parent1, parent2, coverage_matrix=None, all_k_sets=None, all_j_combinations=None, use_greedy=False):
    """
    交叉操作，支持传统交叉和贪心交叉
    
    参数:
    - parent1, parent2: 父代解
    - coverage_matrix, all_k_sets, all_j_combinations: 贪心交叉所需的数据
    - use_greedy: 是否使用贪心交叉
    
    返回:
    - 两个子代解
    """
    if use_greedy and coverage_matrix is not None and all_k_sets is not None and all_j_combinations is not None:
        # 使用贪心交叉生成第一个子代
        child1 = greedy_crossover(parent1, parent2, coverage_matrix, all_k_sets, all_j_combinations)
        
        # 为了保持多样性，第二个子代可以通过随机变异生成
        if random.random() < 0.5:
            # 使用传统交叉生成第二个子代
            temp_child1, child2 = traditional_crossover(parent1, parent2)
        else:
            # 或者也使用贪心交叉，但随机选择不同的起始集合
            child2 = greedy_crossover(parent2, parent1, coverage_matrix, all_k_sets, all_j_combinations)
        
        return child1, child2
    else:
        # 使用传统交叉
        return traditional_crossover(parent1, parent2)

def traditional_crossover(parent1, parent2):
    """传统的多点交叉操作"""
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
        operation = random.choice(['add', 'remove', 'replace', 'optimize'])
        
        if operation == 'add' and len(solution) < n:
            # 生成一个新的k元素集合
            new_group = tuple(sorted(random.sample(all_items, k)))
            solution.append(new_group)
        elif operation == 'remove' and len(solution) > 1:
            # 移除一个贡献最小的集合
            worst_idx = random.randint(0, len(solution) - 1)
            solution.pop(worst_idx)
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
                for _ in range(num_changes):
                    if len(current_set) > 0:
                        # 移除一个元素
                        to_remove = random.choice(list(current_set))
                        current_set.remove(to_remove)
                        # 添加一个新元素
                        available = set(all_items) - current_set
                        if available:
                            to_add = random.choice(list(available))
                            current_set.add(to_add)
                solution[idx] = tuple(sorted(current_set))
        
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

def local_search(solution, n, k, j, s, min_cover, max_iterations=100):
    """局部搜索优化解"""
    best_solution = solution[:]
    best_fitness = fitness(solution, n, k, j, s, strict_coverage=False, min_cover=min_cover)
    
    for _ in range(max_iterations):
        # 找出未被充分覆盖的组合
        uncovered = find_uncovered_combinations(best_solution, n, k, j, s, min_cover)
        if not uncovered:
            break
        
        # 随机选择一个未覆盖的组合
        j_comb, uncovered_s = random.choice(uncovered)
        
        # 尝试构造一个新的集合来覆盖未覆盖的s组合
        if uncovered_s:
            s_comb = set(random.choice(uncovered_s))
            # 从j组合中选择额外的元素来构成k元素集合
            remaining = set(j_comb) - s_comb
            if len(remaining) + len(s_comb) >= k:
                additional = random.sample(list(remaining), k - len(s_comb))
                new_set = tuple(sorted(list(s_comb) + additional))
                
                # 尝试添加新集合
                new_solution = best_solution + [new_set]
                new_fitness = fitness(new_solution, n, k, j, s, strict_coverage=False, min_cover=min_cover)
                
                if new_fitness > best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness
        
        # 尝试修改现有集合
        for idx, group in enumerate(best_solution):
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
            
            # 检查修改后的集合是否更好
            new_solution = best_solution[:]
            new_solution[idx] = tuple(sorted(modified))
            new_fitness = fitness(new_solution, n, k, j, s, strict_coverage=False, min_cover=min_cover)
            
            if new_fitness > best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
    
    return best_solution

def get_dynamic_mutation_rate(generation, max_generations, base_rate=0.1):
    """动态调整突变率：前期低，后期高"""
    # 前期保持较低的基础突变率，随着代数增加而逐渐增大
    return base_rate * (1 + generation / max_generations * 2)

def get_dynamic_crossover_rate(generation, max_generations, base_rate=0.9):
    """动态调整交叉率：前期大，后期小"""
    # 前期保持较高的交叉率，随着代数增加而逐渐降低
    return base_rate * (1 - generation / max_generations * 0.5)

def build_coverage_matrix(n, k, j, s, all_k_sets, all_j_combinations):
    """
    构建覆盖矩阵，表示每个k子集对j组合的覆盖关系
    返回:
    - 覆盖矩阵: shape为(len(all_k_sets), len(all_j_combinations))的布尔矩阵
    """
    print("构建覆盖矩阵...")
    coverage_matrix = np.zeros((len(all_k_sets), len(all_j_combinations)), dtype=bool)
    
    # 优化：使用集合表示每个k子集和j组合
    k_sets = [set(k_set) for k_set in all_k_sets]
    j_sets = [set(j_comb) for j_comb in all_j_combinations]
    
    for k_idx, k_set in enumerate(k_sets):
        for j_idx, j_set in enumerate(j_sets):
            # 检查k_set是否覆盖j_set的至少s个元素
            intersection = len(k_set.intersection(j_set))
            if intersection >= s:
                coverage_matrix[k_idx, j_idx] = True
    
    print(f"覆盖矩阵构建完成，形状: {coverage_matrix.shape}")
    return coverage_matrix

def intelligent_mutation(solution, n, k, j, s, coverage_matrix, all_k_sets, all_j_combinations, mutation_rate=0.3):
    """
    基于覆盖矩阵的智能变异，优先添加能覆盖未覆盖j组合的k集合
    
    参数:
    - solution: 当前解
    - coverage_matrix: 覆盖矩阵
    - all_k_sets: 所有k子集
    - all_j_combinations: 所有j组合
    - mutation_rate: 变异概率
    
    返回:
    - 变异后的解
    """
    if random.random() > mutation_rate:
        return solution
    
    # 将solution中的k集合转换为索引
    solution_indices = []
    for k_set in solution:
        # 处理可能的类型差异
        k_set_tuple = tuple(sorted(k_set)) if not isinstance(k_set, tuple) else k_set
        # 找到匹配的索引
        found = False
        for idx, reference_set in enumerate(all_k_sets):
            if set(k_set_tuple) == set(reference_set):
                solution_indices.append(idx)
                found = True
                break
    
    # 如果找不到匹配的索引，使用随机变异
    if not solution_indices:
        return mutate(solution, n, k, mutation_rate, j, s)
    
    # 检查索引是否在有效范围内
    valid_indices = [idx for idx in solution_indices if 0 <= idx < len(coverage_matrix)]
    if not valid_indices:
        return mutate(solution, n, k, mutation_rate, j, s)
    
    # 计算当前解的覆盖情况
    solution_indices = np.array(valid_indices)
    current_coverage = np.any(coverage_matrix[solution_indices, :], axis=0)
    
    # 找出未被覆盖的j组合
    uncovered_indices = np.where(~current_coverage)[0]
    
    # 如果所有j组合都被覆盖，尝试优化解
    if len(uncovered_indices) == 0:
        # 计算每个k子集的贡献度
        # 贡献度为该子集独自覆盖的j组合数量
        unique_coverage = []
        for idx in solution_indices:
            # 移除该子集后的覆盖情况
            remaining_indices = [i for i in solution_indices if i != idx]
            if not remaining_indices:
                unique_coverage.append(len(all_j_combinations))  # 如果只有一个子集，其贡献是全部
                continue
                
            remaining_coverage = np.any(coverage_matrix[remaining_indices, :], axis=0)
            unique_to_this = np.sum(~remaining_coverage & coverage_matrix[idx])
            unique_coverage.append(unique_to_this)
        
        # 找到贡献最小的子集
        if unique_coverage:
            worst_idx = solution_indices[np.argmin(unique_coverage)]
            
            # 考虑移除该子集
            if random.random() < 0.5 and len(solution) > 2:
                # 找到该子集在solution中的位置
                for i, k_set in enumerate(solution):
                    k_set_tuple = tuple(sorted(k_set)) if not isinstance(k_set, tuple) else k_set
                    found = False
                    for idx, reference_set in enumerate(all_k_sets):
                        if set(k_set_tuple) == set(reference_set) and idx == worst_idx:
                            solution.pop(i)
                            found = True
                            break
                    if found:
                        break
            # 否则替换为更好的子集
            else:
                # 找出能覆盖worst_idx独特覆盖的j组合的最佳子集
                remaining_indices = [i for i in solution_indices if i != worst_idx]
                if not remaining_indices:
                    return solution
                    
                remaining_coverage = np.any(coverage_matrix[remaining_indices, :], axis=0)
                uncovered_after_removal = ~remaining_coverage
                
                # 计算每个未使用的k子集对这些j组合的覆盖
                unused_indices = [i for i in range(len(all_k_sets)) if i not in solution_indices and 0 <= i < len(coverage_matrix)]
                if unused_indices:
                    coverage_scores = np.sum(coverage_matrix[unused_indices][:, uncovered_after_removal], axis=1)
                    if np.max(coverage_scores) > 0:
                        best_replacement = unused_indices[np.argmax(coverage_scores)]
                        
                        # 替换worst_idx为best_replacement
                        for i, k_set in enumerate(solution):
                            k_set_tuple = tuple(sorted(k_set)) if not isinstance(k_set, tuple) else k_set
                            found = False
                            for idx, reference_set in enumerate(all_k_sets):
                                if set(k_set_tuple) == set(reference_set) and idx == worst_idx:
                                    # 确保best_replacement索引有效
                                    if 0 <= best_replacement < len(all_k_sets):
                                        solution[i] = all_k_sets[best_replacement]
                                    found = True
                                    break
                            if found:
                                break
    
    # 如果有未覆盖的j组合，选择能覆盖最多未覆盖j组合的k集合
    elif uncovered_indices.size > 0:
        # 计算每个k子集对未覆盖j组合的覆盖
        unused_indices = [i for i in range(len(all_k_sets)) if i not in solution_indices and 0 <= i < len(coverage_matrix)]
        if unused_indices:
            coverage_scores = np.sum(coverage_matrix[unused_indices][:, uncovered_indices], axis=1)
            if np.max(coverage_scores) > 0:
                best_idx = unused_indices[np.argmax(coverage_scores)]
                # 添加到解中，确保索引有效
                if 0 <= best_idx < len(all_k_sets):
                    solution.append(all_k_sets[best_idx])
    
    return solution

def genetic_algorithm(n, k, j, s, m, population_size=100, generations=100, base_mutation_rate=0.1, base_crossover_rate=0.9, elitism=True, strict_coverage=True, min_cover=1, use_cache=True, use_intelligent_mutation=True, use_greedy_crossover=True):
    """使用预计算的遗传算法求解集合覆盖问题，支持缓存"""
    total_start_time = time.time()
    
    # Define min_groups and max_groups
    min_groups = max(3, j)  # Minimum groups should be at least j or 3
    max_groups = n * 2      # Maximum groups can be twice the number of elements
    
    # 预计算覆盖关系，支持缓存
    precompute_start_time = time.time()
    print("正在预计算覆盖关系...")
    coverage_lookup, all_k_sets, all_j_combinations = precompute_coverage_relations(n, k, j, s, m, use_cache)
    precompute_time = time.time() - precompute_start_time
    print(f"预计算完成，用时: {precompute_time:.2f}秒")
    
    # 构建覆盖矩阵，用于智能变异和贪心交叉
    coverage_matrix = None
    if use_intelligent_mutation or use_greedy_crossover:
        coverage_matrix = build_coverage_matrix(n, k, j, s, all_k_sets, all_j_combinations)
    
    # 初始化种群
    init_start_time = time.time()
    population = create_initial_population(n, k, population_size, min_groups, max_groups, j, s)
    init_time = time.time() - init_start_time
    print(f"初始化种群完成，用时: {init_time:.2f}秒")
    
    best_solution = None
    best_fitness = float('-inf')
    generations_without_improvement = 0
    
    print("\n遗传算法进化过程:")
    print("代数\t最佳适应度\t突变率\t交叉率\t解大小\t本代用时(秒)")
    
    evolution_times = []  # 记录每代的进化时间
    
    for generation in range(generations):
        generation_start_time = time.time()
        
        # 动态调整突变率和交叉率
        current_mutation_rate = get_dynamic_mutation_rate(generation, generations, base_mutation_rate)
        current_crossover_rate = get_dynamic_crossover_rate(generation, generations, base_crossover_rate)
        
        # 评估适应度
        fitness_start_time = time.time()
        fitness_scores = [(fitness_with_lookup(solution, coverage_lookup, all_k_sets, all_j_combinations, s, strict_coverage, min_cover), solution) 
                         for solution in population]
        fitness_scores.sort(reverse=True)
        fitness_time = time.time() - fitness_start_time
        
        # 更新最佳解
        if fitness_scores[0][0] > best_fitness:
            best_fitness = fitness_scores[0][0]
            best_solution = fitness_scores[0][1][:]
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
        
        # 如果连续多代没有改进，增加变异率
        if generations_without_improvement > 10:
            current_mutation_rate = min(0.5, current_mutation_rate * 1.5)
        
        # 精英选择 - 保留最佳解
        new_population = []
        if elitism:
            elite_size = max(1, int(population_size * 0.1))  # 10%的精英
            new_population.extend([fs[1] for fs in fitness_scores[:elite_size]])
        
        # 选择和繁殖
        while len(new_population) < population_size:
            if random.random() < current_crossover_rate:
                # 选择父母
                parents = tournament_selection([fs[1] for fs in fitness_scores], [fs[0] for fs in fitness_scores], 2)
                
                # 交叉
                child1, child2 = crossover(parents[0], parents[1], coverage_matrix, all_k_sets, all_j_combinations, use_greedy=use_greedy_crossover)
                
                # 变异
                if use_intelligent_mutation and coverage_matrix is not None:
                    # 使用智能变异
                    child1 = intelligent_mutation(child1, n, k, j, s, coverage_matrix, all_k_sets, all_j_combinations, current_mutation_rate)
                    child2 = intelligent_mutation(child2, n, k, j, s, coverage_matrix, all_k_sets, all_j_combinations, current_mutation_rate)
                else:
                    # 使用常规变异
                    child1 = mutate(child1, n, k, current_mutation_rate, j, s)
                    child2 = mutate(child2, n, k, current_mutation_rate, j, s)
                
                new_population.append(child1)
                if len(new_population) < population_size:
                    new_population.append(child2)
            else:
                # 直接选择
                parent = tournament_selection([fs[1] for fs in fitness_scores], [fs[0] for fs in fitness_scores], 1)[0]
                
                # 变异
                if use_intelligent_mutation and coverage_matrix is not None:
                    child = intelligent_mutation(parent[:], n, k, j, s, coverage_matrix, all_k_sets, all_j_combinations, current_mutation_rate)
                else:
                    child = mutate(parent[:], n, k, current_mutation_rate, j, s)
                
                new_population.append(child)
        
        # 更新种群
        population = new_population
        
        # 每10代或最后一代，对最佳解进行局部搜索优化
        if (generation % 10 == 0 or generation == generations - 1) and best_solution:
            optimized = local_search(best_solution[:], n, k, j, s, min_cover, max_iterations=20)
            opt_fitness = fitness_with_lookup(optimized, coverage_lookup, all_k_sets, all_j_combinations, s, strict_coverage, min_cover)
            if opt_fitness > best_fitness:
                best_solution = optimized
                best_fitness = opt_fitness
                generations_without_improvement = 0
        
        generation_time = time.time() - generation_start_time
        evolution_times.append(generation_time)
        
        # 输出本代统计信息
        print(f"{generation}\t{fitness_scores[0][0]:.2f}\t{current_mutation_rate:.2f}\t"
              f"{current_crossover_rate:.2f}\t{len(fitness_scores[0][1])}\t{generation_time:.2f}")
    
    total_time = time.time() - total_start_time
    
    # 输出总体统计信息
    print("\n算法执行统计:")
    print(f"预计算时间: {precompute_time:.2f}秒")
    print(f"初始化时间: {init_time:.2f}秒")
    print(f"平均每代时间: {sum(evolution_times)/len(evolution_times):.2f}秒")
    print(f"总执行时间: {total_time:.2f}秒")
    
    return best_solution  # Return the best solution found during evolution

def tournament_selection(population, fitness_scores, num_selected):
    selected = []
    for _ in range(num_selected):
        tournament = random.sample(list(zip(fitness_scores, population)), 3)
        winner = max(tournament, key=lambda x: x[0])
        selected.append(winner[1])
    return selected

def create_letter_to_number_mapping(n, m):
    """
    创建字母到随机数的映射关系
    
    参数:
    n: 需要多少个字母映射
    m: 随机数的范围上限，从1-m中选择n个不同的随机数
    
    返回:
    letter_to_num: 字母到数字的映射字典
    num_to_letter: 数字到字母的映射字典
    """
    # 从1-m中随机选择n个不同的数字
    random_numbers = sorted(random.sample(range(1, m+1), n))
    
    # 创建映射
    letter_to_num = {}
    num_to_letter = {}
    for i in range(n):
        letter = chr(ord('A') + i)
        num = random_numbers[i]
        letter_to_num[letter] = num
        num_to_letter[num] = letter
    
    print(f"生成的字母到数字映射:")
    for letter, num in sorted(letter_to_num.items()):
        print(f"{letter} -> {num}")
    
    return letter_to_num, num_to_letter

def main(only_count=False, show_sets=False, use_cache=True, use_intelligent_mutation=True, use_greedy_crossover=True):
    # 测试用例列表
    test_cases = [
        # 测试用例8: n=16, k=6, j=6, s=4
        {"n": 13, "k": 6, "j": 6, "s": 4, "m": 45, "strict_coverage": False, "min_cover": 1},
    ]
    
    # 运行每个测试用例
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"测试用例 {i}:")
        print(f"参数: n={case['n']}, k={case['k']}, j={case['j']}, s={case['s']}, m={case['m']}")
        print(f"覆盖模式: {'严格覆盖' if case['strict_coverage'] else '宽松覆盖'}")
        if not case['strict_coverage']:
            print(f"最小覆盖数: {case['min_cover']}")
        
        # 测量性能 - 预计算时间
        start_time = time.time()
        
        # 使用遗传算法生成初始解
        print("使用遗传算法生成初始解...")
        initial_solution, samples = generate_initial_solution_ga(
            case['n'], case['k'], case['j'], case['s'], case['m'],
            population_size=50,  # 减少种群大小
            generations=100,     # 减少迭代次数
            use_cache=use_cache,  # 使用缓存
            use_intelligent_mutation=use_intelligent_mutation,  # 使用智能变异
            use_greedy_crossover=use_greedy_crossover  # 使用贪心交叉
        )
        
        elapsed_time = time.time() - start_time
        print(f"遗传算法找到的初始解包含 {len(initial_solution)} 个集合")
        print(f"总运行时间: {elapsed_time:.2f}秒")
        
        # 如果n大于8，直接使用遗传算法结果
        if case['n'] > 8:
            print(f"\n由于n={case['n']} > 8，跳过ILP求解，直接使用遗传算法结果")
            # 结果已经在generate_initial_solution_ga函数中输出
            continue
        
        # 构建ILP问题
        solver, x, k_subsets, j_subsets, s_subsets, ilp_samples = construct_ilp_problem(
            case['n'], case['k'], case['j'], case['s'], case['m'],
            case['strict_coverage'], case['min_cover']
        )
        
        # 设置SCIP求解器参数以优化性能
        solver.SetTimeLimit(1800000)  # 30分钟，单位是毫秒
        solver.SetNumThreads(0)  # 使用所有可用的CPU核心
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
                    print(f"{i}. {','.join(map(str, sorted(subset)))}")
        else:
            print(f"求解状态: {status}")
            if status == pywraplp.Solver.INFEASIBLE:
                print("问题无可行解")
            elif status == pywraplp.Solver.ABNORMAL:
                print("求解器异常终止")

if __name__ == "__main__":
    main(only_count=False, show_sets=True, use_cache=True, use_intelligent_mutation=True, use_greedy_crossover=True)
