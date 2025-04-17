from pulp import *
from itertools import combinations
import os
import numpy as np
from pulp import GLPK

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

def construct_ilp_problem(n, k, j, s):
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
    
    # 创建ILP问题
    prob = LpProblem("Set_Covering_Problem", LpMinimize)
    
    # 创建变量
    x = LpVariable.dicts("x", range(len(k_subsets)), cat='Binary')
    
    # 目标函数：最小化选择的集合数量
    prob += lpSum(x[i] for i in range(len(k_subsets)))
    
    # 添加约束：每个j元素子集必须被至少一个k元素子集覆盖
    for j_idx, j_subset in enumerate(j_subsets):
        covering_sets = [i for i, k_subset in enumerate(k_subsets) if j_subset.issubset(k_subset)]
        if covering_sets:  # 只有当有覆盖集时才添加约束
            prob += lpSum(x[i] for i in covering_sets) >= 1
    
    return prob, x, k_subsets

def main():
    # 参数设置
    n = 9  # 样本数量
    k = 6  # k元素子集大小
    j = 4  # j元素子集大小
    s = 4  # s元素子集大小
    
    # 参数检查
    if not (1 <= s <= j <= k <= n):
        print("参数错误：必须满足 1 <= s <= j <= k <= n")
        return
    
    print(f"参数设置: n={n}, k={k}, j={j}, s={s}")
    
    # 构建ILP问题
    prob, x, k_subsets = construct_ilp_problem(n, k, j, s)
    
    # 设置求解器
    solver = GLPK()
    solver.timeLimit = 300  # 设置5分钟的时间限制
    
    # 求解问题
    print("开始求解...")
    prob.solve(solver)
    
    # 输出结果
    print(f"求解状态: {LpStatus[prob.status]}")
    if prob.status == LpStatusOptimal:
        selected_subsets = []
        for i in range(len(k_subsets)):
            if x[i].value() == 1:
                selected_subsets.append(k_subsets[i])
        
        print(f"最小数量: {len(selected_subsets)}")
        print("选择的集合:")
        for i, subset in enumerate(selected_subsets, 1):
            print(f"{i}. {','.join(sorted(subset))}")
    else:
        print("未找到最优解")

if __name__ == "__main__":
    main()
