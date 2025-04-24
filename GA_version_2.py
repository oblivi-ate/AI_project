import random
from itertools import combinations

#region test cases

def test_case_1():
    n = 7
    k = 6
    j = 5
    s = 5
    m = 45
    x = 1
    return n, k, j, s, m, x

def test_case_2():
    n = 8
    k = 6
    j = 4
    s = 4
    m = 45
    x = 1
    return n, k, j, s, m, x

def test_case_3():
    n = 9
    k = 6
    j = 4
    s = 4
    m = 45
    x = 1
    return n, k, j, s, m, x

def test_case_4():
    n = 8
    k = 6
    j = 6
    s = 5
    m = 45
    x = 1
    return n, k, j, s, m, x

def test_case_5():
    n = 8
    k = 6
    j = 6
    s = 5
    m = 45
    x = 4
    return n, k, j, s, m, x

def test_case_6():
    n = 9
    k = 6
    j = 6
    s = 4
    m = 45
    x = 1
    return n, k, j, s, m, x

def test_case_7():
    n = 10
    k = 6
    j = 6
    s = 4
    m = 45
    x = 1
    return n, k, j, s, m, x

def test_case_8():
    n = 12
    k = 6
    j = 6
    s = 4
    m = 45
    x = 1
    return n, k, j, s, m, x

#endregion

def generate_combinations(items, k):
    return list(combinations(items, k))

def check_coverage(group, j_combination, s):
    # Check if the group contains at least s items from the j_combination
    common_elements = set(group) & set(j_combination)
    return len(common_elements) >= s

def satisfies_condition(group, n, k, j, s):
    # Generate all possible j-sized combinations from n items
    all_items = list(range(n))
    j_combinations = list(combinations(all_items, j))
    
    # Check if at least one j-combination has all its s-sized subgroups covered
    for j_comb in j_combinations:
        if check_coverage(group, j_comb, s):
            return True
    return False

def fitness(solution, n, k, j, s):
    solution_sets = [set(group) for group in solution]
    all_items = list(range(n))
    j_combinations = list(combinations(all_items, j))
    
    # Count covered j-combinations
    covered_count = 0
    for j_comb in j_combinations:
        for group in solution_sets:
            if check_coverage(group, j_comb, s):
                covered_count += 1
                break
    
    # Penalize uncovered combinations and large solutions
    uncovered_penalty = len(j_combinations) - covered_count
    size_penalty = len(solution) * 0.5  # Adjust weight for solution size penalty
    return covered_count - uncovered_penalty - size_penalty

def create_initial_population(n, k, population_size, min_groups, max_groups, j, s):
    population = []
    all_combinations = list(combinations(range(n), k))
    
    for _ in range(population_size):
        solution = []
        uncovered_j_combinations = list(combinations(range(n), j))
        
        while uncovered_j_combinations and len(solution) < max_groups:
            # Select the group that covers the most uncovered j-combinations
            best_group = max(all_combinations, key=lambda group: sum(
                check_coverage(group, j_comb, s) for j_comb in uncovered_j_combinations
            ))
            solution.append(best_group)
            
            # Remove covered j-combinations
            uncovered_j_combinations = [
                j_comb for j_comb in uncovered_j_combinations
                if not check_coverage(best_group, j_comb, s)
            ]
        
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
    if random.random() < mutation_rate:
        all_combinations = list(combinations(range(n), k))
        operation = random.choice(['add', 'remove', 'replace'])
        
        if operation == 'add' and len(solution) < n:
            new_group = random.choice(all_combinations)
            solution.append(new_group)
        elif operation == 'remove' and len(solution) > 1:
            solution.pop(random.randint(0, len(solution) - 1))
        elif operation == 'replace':
            idx = random.randint(0, len(solution) - 1)
            solution[idx] = random.choice(all_combinations)
        
        # Ensure the solution satisfies constraints
        if satisfies_condition(solution, n, k, j, s):
            return solution
    return solution

def genetic_algorithm(n, k, j, s, population_size=50, generations=100, mutation_rate=0.1, elitism=True):
    min_groups = max(3, j)
    max_groups = n * 2
    
    population = create_initial_population(n, k, population_size, min_groups, max_groups, j, s)
    best_solution = None
    best_fitness = float('-inf')

    for generation in range(generations):
        # Adjust mutation rate dynamically
        mutation_rate = max(0.01, mutation_rate * (1 - generation / generations))
        
        # Evaluate fitness
        fitness_scores = [(fitness(solution, n, k, j, s), solution) for solution in population]
        fitness_scores.sort(reverse=True)
        
        # Update best solution
        if fitness_scores[0][0] > best_fitness:
            best_fitness = fitness_scores[0][0]
            best_solution = fitness_scores[0][1]
        
        # Elitism: carry the best solution to the next generation
        next_population = [best_solution] if elitism else []

        # Selection: Tournament selection
        selected = tournament_selection(population, fitness_scores, population_size // 2)
        
        # Crossover and mutation
        while len(next_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, n, k, mutation_rate, j, s)
            child2 = mutate(child2, n, k, mutation_rate, j, s)
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

# Example usage
def main():
    
    [n, k, j, s, m, x] = test_case_4()
    
    best_solution = genetic_algorithm(n, k, j, s)
    output = format_output(best_solution, m, n, k, j, s, x)
    print(output)

if __name__ == "__main__":
    main()