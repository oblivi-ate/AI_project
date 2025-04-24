from ortools.linear_solver import pywraplp
from itertools import combinations
import os
import numpy as np
import random

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

def generate_initial_solution_ga(n, k, j, s, population_size=50, generations=100):
    """使用遗传算法生成初始解"""
    def fitness(solution):
        # 计算解的适应度
        covered = set()
        for subset in solution:
            for j_subset in combinations(subset, j):
                for s_subset in combinations(j_subset, s):
                    covered.add(s_subset)
        return len(covered)
    
    def create_individual():
        # 创建单个个体
        all_subsets = list(combinations(range(n), k))
        num_subsets = random.randint(1, len(all_subsets))
        return random.sample(all_subsets, num_subsets)
    
    def crossover(parent1, parent2):
        # 交叉操作
        child = list(set(parent1 + parent2))
        return child
    
    def mutate(individual):
        # 变异操作
        if random.random() < 0.1:
            all_subsets = list(combinations(range(n), k))
            if random.random() < 0.5 and len(individual) > 1:
                individual.pop(random.randint(0, len(individual)-1))
            else:
                individual.append(random.choice(all_subsets))
        return individual
    
    # 初始化种群
    population = [create_individual() for _ in range(population_size)]
    
    # 进化过程
    for _ in range(generations):
        # 评估适应度
        fitnesses = [fitness(ind) for ind in population]
        
        # 选择
        selected = []
        for _ in range(population_size):
            tournament = random.sample(list(zip(population, fitnesses)), 3)
            selected.append(max(tournament, key=lambda x: x[1])[0])
        
        # 交叉和变异
        new_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = selected[i], selected[i+1]
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            new_population.extend([child1, child2])
        
        population = new_population
    
    # 返回最佳解
    best_solution = max(population, key=fitness)
    return best_solution

def main(only_count=False, show_sets=False):
    # 参数设置
    n = 10  # 样本数量
    k = 6  # k元素子集大小
    j = 6  # j元素子集大小
    s = 4  # s元素子集大小
    strict_coverage = False  # 是否使用严格覆盖模式
    min_cover = 1  # 宽松覆盖模式下，每个j元素子集至少需要被覆盖的s元素子集数量
    
    # 参数检查
    if not (1 <= s <= j <= k <= n):
        print("参数错误：必须满足 1 <= s <= j <= k <= n")
        return
    
    if not strict_coverage and min_cover < 1:
        print("参数错误：宽松覆盖模式下，min_cover必须大于等于1")
        return
    
    print(f"参数设置: n={n}, k={k}, j={j}, s={s}, strict_coverage={strict_coverage}, min_cover={min_cover}")
    
    # 使用遗传算法生成初始解
    print("使用遗传算法生成初始解...")
    initial_solution = generate_initial_solution_ga(n, k, j, s)
    print(f"遗传算法找到的初始解包含 {len(initial_solution)} 个集合")
    
    # 构建ILP问题
    solver, x, k_subsets, j_subsets, s_subsets = construct_ilp_problem(n, k, j, s, strict_coverage, min_cover)
    
    # 设置初始解
    for i, subset in enumerate(k_subsets):
        if subset in initial_solution:
            x[i].SetStartValue(1)
        else:
            x[i].SetStartValue(0)
    
    # 设置SCIP求解器参数以优化性能
    solver.SetTimeLimit(1800000)  # 30分钟，单位是毫秒
    solver.SetNumThreads(0)  # 使用所有可用的CPU核心
    solver.EnableOutput()  # 启用求解器输出
    solver.SetSolverSpecificParametersAsString("""
        limits/time = 1800
        limits/memory = 4096
        limits/nodes = 1000000
        parallel/maxnthreads = 0
        presolving/maxrounds = 0
        separating/maxrounds = 0
        separating/maxroundsroot = 0
        separating/maxcuts = 0
        separating/maxcutsroot = 0
        heuristics/emphasis = aggressive
        branching/pscost/priority = 1000
        branching/pscost/scorefac = 0.167
        branching/pscost/scorefac = 0.167
        branching/pscost/scorefac = 0.167
    """)
    
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
                
            # 输出覆盖统计
            print("\n覆盖统计:")
            for j_idx, j_subset in enumerate(j_subsets):
                print(f"\nj子集 {','.join(sorted(j_subset))}:")
                for s_subset in s_subsets[j_idx]:
                    covering_count = sum(1 for k_subset in selected_subsets if s_subset.issubset(k_subset))
                    print(f"  s子集 {','.join(sorted(s_subset))} 被 {covering_count} 个k子集覆盖")
    else:
        print(f"求解状态: {status}")
        if status == pywraplp.Solver.INFEASIBLE:
            print("问题无可行解，请尝试调整参数：")
            print("1. 减小min_cover的值")
            print("2. 减小s的值")
            print("3. 增大k的值")
        elif status == pywraplp.Solver.ABNORMAL:
            print("求解器异常终止，可能是由于：")
            print("1. 问题规模太大")
            print("2. 数值精度问题")
            print("3. 求解时间不足")

if __name__ == "__main__":
    main(only_count=False, show_sets=False)
