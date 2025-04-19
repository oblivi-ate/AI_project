import random
from itertools import combinations



#endregion

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
            s_combinations = list(combinations(j_comb, s))
            covered_s_combinations = set()  # 使用集合来避免重复计数
            
            # 对于每个s组合，检查是否被任何一个解集合覆盖
            for s_comb in s_combinations:
                for group in solution_sets:
                    if set(s_comb).issubset(group):
                        covered_s_combinations.add(tuple(sorted(s_comb)))
            
            # 检查是否有足够多的s组合被覆盖
            if len(covered_s_combinations) < min_cover:
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
    population = []
    all_combinations = list(combinations(range(n), k))
    
    for _ in range(population_size):
        solution = []
        uncovered_j_combinations = list(combinations(range(n), j))
        
        # 使用贪心算法生成初始解，但限制解的大小
        while uncovered_j_combinations and len(solution) < max_groups:
            # 选择能够覆盖最多未覆盖j组合的集合
            best_group = max(all_combinations, key=lambda group: sum(
                check_coverage(group, j_comb, s) for j_comb in uncovered_j_combinations
            ))
            solution.append(best_group)
            
            # 更新未覆盖的j组合
            uncovered_j_combinations = [
                j_comb for j_comb in uncovered_j_combinations
                if not check_coverage(best_group, j_comb, s)
            ]
        
        # 如果生成的解太大，随机移除一些集合
        while len(solution) > min_groups + 2:  # 允许比最小解大2个集合
            solution.pop(random.randint(0, len(solution) - 1))
        
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

def genetic_algorithm(n, k, j, s, population_size=100, generations=200, mutation_rate=0.1, elitism=True, strict_coverage=True, min_cover=1):
    """使用遗传算法求解集合覆盖问题，支持宽松和严格覆盖模式"""
    min_groups = max(3, j)
    max_groups = n * 2
    
    # 初始化种群
    population = create_initial_population(n, k, population_size, min_groups, max_groups, j, s)
    best_solution = None
    best_fitness = float('-inf')
    generations_without_improvement = 0
    
    for generation in range(generations):
        # 动态调整变异率
        current_mutation_rate = mutation_rate * (1 + generations_without_improvement / 100)  # 增加分母，使变异率增长更慢
        
        # 评估适应度
        fitness_scores = [(fitness(solution, n, k, j, s, strict_coverage, min_cover), solution) 
                         for solution in population]
        fitness_scores.sort(reverse=True)
        
        # 更新最佳解
        current_best_fitness = fitness_scores[0][0]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = fitness_scores[0][1]
            generations_without_improvement = 0
            
            # 对最佳解进行局部搜索优化
            improved_solution = local_search(best_solution, n, k, j, s, min_cover)
            improved_fitness = fitness(improved_solution, n, k, j, s, strict_coverage, min_cover)
            
            if improved_fitness > best_fitness:
                best_fitness = improved_fitness
                best_solution = improved_solution
        else:
            generations_without_improvement += 1
        
        # 如果找到完美解，提前终止
        if best_fitness >= 99.5 and satisfies_condition(best_solution, n, k, j, s, strict_coverage, min_cover):
            print(f"找到完美解，提前终止遗传算法")
            break
        
        # 如果长时间没有改进，重新初始化部分种群
        if generations_without_improvement > 30:  # 增加这个阈值，减少重新初始化的频率
            new_individuals = create_initial_population(n, k, population_size // 4, 
                                                     min_groups, max_groups, j, s)
            population = population[:3*population_size//4] + new_individuals
            generations_without_improvement = 0
        else:
            # 精英保留
            next_population = [best_solution] if elitism else []
            
            # 选择
            selected = tournament_selection(population, fitness_scores, population_size // 2)
            
            # 交叉和变异
            while len(next_population) < population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, n, k, current_mutation_rate, j, s)
                child2 = mutate(child2, n, k, current_mutation_rate, j, s)
                next_population.extend([child1, child2])
            
            population = next_population[:population_size]
    
    return best_solution

def tournament_selection(population, fitness_scores, num_selected):
    selected = []
    for _ in range(num_selected):
        tournament = random.sample(list(zip(fitness_scores, population)), 3)
        winner = max(tournament, key=lambda x: x[0])
        selected.append(winner[1])
    return selected

def format_output(solution, m, n, k, j, s, x):
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    y = len(solution)
    
    header = f"{m}-{n}-{k}-{j}-{s}-{x}-{y}"
    groups = []
    for group in solution:
        group_str = ','.join(letters[i] for i in group)
        groups.append(group_str)
    
    # Add coverage statistics
    coverage_stats = f"Total groups: {y}, Coverage: {fitness(solution, n, k, j, s)}"
    return header + '\n' + '\n'.join(f"{i+1}.{group}" for i, group in enumerate(groups)) + '\n' + coverage_stats

def verify_solution(solution, n, k, j, s, min_cover):
    """验证给定的解是否满足条件"""
    # 将字母转换为数字
    letter_to_num = {chr(ord('A') + i): i for i in range(n)}
    num_to_letter = {i: chr(ord('A') + i) for i in range(n)}
    solution_sets = []
    
    print("\n解的详细信息:")
    for group in solution:
        # 分割字符串并去除空格
        letters = [c.strip() for c in group.split(',')]
        nums = [letter_to_num[c] for c in letters]
        solution_sets.append(set(nums))
        print(f"集合: {group} -> {nums}")
    
    all_items = list(range(n))
    j_combinations = list(combinations(all_items, j))
    print(f"\n总共有 {len(j_combinations)} 个{j}元素组合")
    
    # 检查是否所有j组合都被覆盖
    total_j_covered = 0
    for j_idx, j_comb in enumerate(j_combinations):
        j_set = set(j_comb)
        s_combinations = list(combinations(j_comb, s))
        covered_s_combinations = set()
        
        print(f"\nj组合 {','.join(num_to_letter[i] for i in j_comb)} ({j_comb}):")
        print(f"可能的{s}元素子集:")
        for s_comb in s_combinations:
            s_letters = [num_to_letter[i] for i in s_comb]
            print(f"- {','.join(s_letters)}")
        
        # 对于每个解集合，找出它能覆盖的s组合
        for group_idx, group in enumerate(solution_sets):
            # 只考虑与当前j组合有足够交集的集合
            if len(group & j_set) >= s:
                # 找出这个集合能覆盖的所有s组合
                for s_comb in s_combinations:
                    if set(s_comb).issubset(group):
                        covered_s_combinations.add(tuple(sorted(s_comb)))
                        print(f"  * {','.join(num_to_letter[i] for i in s_comb)} 被集合 {solution[group_idx]} 覆盖")
        
        # 输出覆盖统计
        s_covered = len(covered_s_combinations)
        print(f"\n该j组合的覆盖统计:")
        print(f"- 总共有 {len(s_combinations)} 个可能的{s}元素子集")
        print(f"- 需要覆盖至少 {min_cover} 个{s}元素子集")
        print(f"- 实际覆盖了 {s_covered} 个{s}元素子集")
        
        if s_covered < min_cover:
            print(f"警告：该j组合的覆盖数量不足！")
            return False
        else:
            total_j_covered += 1
    
    print(f"\n所有{j}元素组合都满足覆盖要求！")
    print(f"总共有 {total_j_covered} 个{j}元素组合被正确覆盖")
    return True

def main():
    # 测试用例列表
    test_cases = [
       
        
        # 测试用例5: n=8, k=6, j=6, s=5
        {"n": 8, "k": 6, "j": 6, "s": 5, "strict_coverage": False, "min_cover": 4},
        
        
        # 测试用例8: n=12, k=6, j=6, s=4
        {"n": 12, "k": 6, "j": 6, "s": 4, "strict_coverage": False, "min_cover": 1}
    ]
    
    # 运行每个测试用例
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"测试用例 {i}:")
        print(f"参数: n={case['n']}, k={case['k']}, j={case['j']}, s={case['s']}")
        print(f"覆盖模式: {'严格覆盖' if case['strict_coverage'] else '宽松覆盖'}")
        if not case['strict_coverage']:
            print(f"最小覆盖数: {case['min_cover']}")
        
        # 运行遗传算法
        best_solution = genetic_algorithm(
            case['n'], case['k'], case['j'], case['s'],
            population_size=100,    # 减小种群大小
            generations=200,        # 减少迭代次数
            mutation_rate=0.1,      # 降低变异率
            elitism=True,
            strict_coverage=case['strict_coverage'],
            min_cover=case['min_cover']
        )
        
        # 输出结果
        print(f"\n找到的解包含 {len(best_solution)} 个集合:")
        for idx, group in enumerate(best_solution, 1):
            print(f"{idx}. {','.join(chr(ord('A') + x) for x in sorted(group))}")
        
        # 验证解的可行性
        if satisfies_condition(best_solution, case['n'], case['k'], case['j'], case['s'], 
                             case['strict_coverage'], case['min_cover']):
            print("解是可行的")
        else:
            print("警告：解不可行！")
        
        # 计算适应度
        fitness_score = fitness(best_solution, case['n'], case['k'], case['j'], case['s'],
                              case['strict_coverage'], case['min_cover'])
        print(f"适应度: {fitness_score}")

if __name__ == "__main__":
    main()