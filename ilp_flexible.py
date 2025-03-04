from pulp import *
from itertools import combinations

# 定义参数
m = 45  # 样本总数 (45 <= m <= 54)
n = 8  # 随机选择的样本数 (7 <= n <= 25)
k = 6  # 大集合的大小 (4 <= k <= 7) 
j = 4  # 小集合的大小 (s <= j <= k)
s = 4     # 最小集合的大小 (3 <= s <= 7)

# 样本
samples = [chr(ord('A') + i) for i in range(n)]  # 根据n生成样本

# 所有k元素子集（S）
S_list = list(combinations(samples, k))
S_dict = {i: set(S_list[i]) for i in range(len(S_list))}

# 所有j元素子集（T）
T_list = list(combinations(samples, j))
T_dict = {i: set(T_list[i]) for i in range(len(T_list))}

# 所有s元素子集（U），针对每个T
U_dict = {}
for i in range(len(T_list)):
    U_dict[i] = list(combinations(T_list[i], s))

# 定义问题
prob = LpProblem("SetCover", LpMinimize)

# 变量
x = LpVariable.dicts("S", range(len(S_list)), cat='Binary')  # 0或1

# 目标
prob += lpSum([x[i] for i in range(len(S_list))])

# 约束
for j_index in range(len(T_list)):
    # 确保每个T至少有一个S包含其至少一个s元素子集
    prob += lpSum([x[i] for i in range(len(S_list)) if any(set(u).issubset(S_dict[i]) for u in U_dict[j_index])]) >= 1

# 求解
prob.solve()

# 打印结果
print("最小数量:", value(prob.objective))
for i in range(len(S_list)):
    if value(x[i]) == 1:
        print(S_list[i])
