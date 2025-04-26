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

def precompute_coverage_relations(n, k, j, s, use_cache=True):
    """预计算k集合和j组合的覆盖关系，使用优化的矩阵运算和数据结构"""
    # 检查缓存
    cache_filename = get_cache_filename(n, k, j, s)
    if use_cache:
        cache_data = load_coverage_cache(cache_filename)
        if cache_data:
            return cache_data
    
    print(f"开始预计算覆盖关系 (n={n}, k={k}, j={j}, s={s})...")
    
    # 使用numpy高效生成所有组合
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
    solution_indices = [all_k_sets.index(k_set) for k_set in solution]
    
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

def generate_initial_solution_ga(n, k, j, s, population_size=50, generations=100, use_cache=True):
    """使用遗传算法生成初始解，支持缓存"""
    # 使用GA_version_2中的遗传算法
    solution = genetic_algorithm(n, k, j, s, 
                               population_size=population_size,
                               generations=generations,
                               base_mutation_rate=0.1,
                               base_crossover_rate=0.9,
                               elitism=True,
                               strict_coverage=False,
                               min_cover=1,
                               use_cache=use_cache)
    
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

def genetic_algorithm(n, k, j, s, population_size=100, generations=100, base_mutation_rate=0.1, base_crossover_rate=0.9, elitism=True, strict_coverage=True, min_cover=1, use_cache=True):
    """使用预计算的遗传算法求解集合覆盖问题，支持缓存"""
    total_start_time = time.time()
    
    # Define min_groups and max_groups
    min_groups = max(3, j)  # Minimum groups should be at least j or 3
    max_groups = n * 2      # Maximum groups can be twice the number of elements
    
    # 预计算覆盖关系，支持缓存
    precompute_start_time = time.time()
    print("正在预计算覆盖关系...")
    coverage_lookup, all_k_sets, all_j_combinations = precompute_coverage_relations(n, k, j, s, use_cache)
    precompute_time = time.time() - precompute_start_time
    print(f"预计算完成，用时: {precompute_time:.2f}秒")
    
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
        
        # ... existing code ...
        
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

def main(only_count=False, show_sets=False, use_cache=True):
    # 测试用例列表
    test_cases = [
        # 测试用例7: n=10, k=6, j=6, s=4
        {"n": 10, "k": 6, "j": 6, "s": 4, "strict_coverage": False, "min_cover": 1},
        
        # 测试用例8: n=12, k=6, j=6, s=4
        {"n": 12, "k": 6, "j": 6, "s": 4, "strict_coverage": False, "min_cover": 1},
        
    ]
    
    # 运行每个测试用例
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"测试用例 {i}:")
        print(f"参数: n={case['n']}, k={case['k']}, j={case['j']}, s={case['s']}")
        print(f"覆盖模式: {'严格覆盖' if case['strict_coverage'] else '宽松覆盖'}")
        if not case['strict_coverage']:
            print(f"最小覆盖数: {case['min_cover']}")
        
        # 测量性能 - 预计算时间
        start_time = time.time()
        
        # 使用遗传算法生成初始解
        print("使用遗传算法生成初始解...")
        initial_solution = generate_initial_solution_ga(
            case['n'], case['k'], case['j'], case['s'],
            population_size=100,  # 减少种群大小
            generations=100,     # 减少迭代次数
            use_cache=use_cache  # 使用缓存
        )
        
        elapsed_time = time.time() - start_time
        print(f"遗传算法找到的初始解包含 {len(initial_solution)} 个集合")
        print(f"总运行时间: {elapsed_time:.2f}秒")
        
        # 如果n大于8，直接使用遗传算法结果
        if case['n'] > 8:
            print(f"\n由于n={case['n']} > 8，跳过ILP求解，直接使用遗传算法结果")
            if show_sets:
                print("\n遗传算法找到的集合:")
                for i, subset in enumerate(initial_solution, 1):
                    # 将数字转换为字母表示
                    subset_letters = [chr(ord('A') + item) for item in subset]
                    print(f"{i}. {','.join(sorted(subset_letters))}")
            continue
        
        # 构建ILP问题
        solver, x, k_subsets, j_subsets, s_subsets = construct_ilp_problem(
            case['n'], case['k'], case['j'], case['s'],
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
                    print(f"{i}. {','.join(sorted(subset))}")
        else:
            print(f"求解状态: {status}")
            if status == pywraplp.Solver.INFEASIBLE:
                print("问题无可行解")
            elif status == pywraplp.Solver.ABNORMAL:
                print("求解器异常终止")

if __name__ == "__main__":
    main(only_count=False, show_sets=True, use_cache=True)
