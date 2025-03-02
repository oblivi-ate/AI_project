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
    print("correct answer:", 6)
    return n, k, j, s, m, x

def test_case_2():
    n = 8
    k = 6
    j = 4
    s = 4
    m = 45
    x = 1
    print("correct answer:", 7)
    return n, k, j, s, m, x

def test_case_3():
    n = 9
    k = 6
    j = 4
    s = 4
    m = 45
    x = 1
    print("correct answer:", 12)
    return n, k, j, s, m, x

def test_case_4():
    n = 8
    k = 6
    j = 4
    s = 5
    m = 45
    x = 1
    print("correct answer:", 15)
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
    # Convert solution to set of tuples for faster operations
    solution_sets = [set(group) for group in solution]
    
    # Generate all j-combinations
    all_items = list(range(n))
    j_combinations = list(combinations(all_items, j))
    
    # Count how many j-combinations are covered
    covered_count = 0
    for j_comb in j_combinations:
        for group in solution_sets:
            if check_coverage(group, j_comb, s):
                covered_count += 1
                break
    
    # Penalize larger solutions to find minimum number of groups
    size_penalty = len(solution) / n
    return covered_count - size_penalty

def create_initial_population(n, k, population_size, min_groups, max_groups):
    population = []
    all_combinations = list(combinations(range(n), k))
    
    # Adjust max_groups to not exceed the number of possible combinations
    max_groups = min(max_groups, len(all_combinations))
    
    for _ in range(population_size):
        num_groups = random.randint(min_groups, max_groups)
        solution = random.sample(all_combinations, num_groups)
        population.append(solution)
    return population

def crossover(parent1, parent2):
    # Single point crossover
    if len(parent1) > 1 and len(parent2) > 1:
        point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2

def mutate(solution, n, k, mutation_rate):
    if random.random() < mutation_rate:
        # Either add, remove, or replace a group
        operation = random.choice(['add', 'remove', 'replace'])
        all_combinations = list(combinations(range(n), k))
        
        if operation == 'add' and len(solution) < n:
            new_group = random.choice(all_combinations)
            return list(solution) + [new_group]
        elif operation == 'remove' and len(solution) > 1:
            return random.sample(solution, len(solution) - 1)
        elif operation == 'replace':
            idx = random.randint(0, len(solution) - 1)
            new_solution = list(solution)
            new_solution[idx] = random.choice(all_combinations)
            return new_solution
    return solution

def genetic_algorithm(n, k, j, s, population_size=50, generations=100, mutation_rate=0.1, elitism=True):
    min_groups = max(3, j)  # Minimum groups based on problem examples
    max_groups = n*2  # Maximum possible groups
    
    population = create_initial_population(n, k, population_size, min_groups, max_groups)
    best_solution = None
    best_fitness = float('-inf')

    for generation in range(generations):
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
            child1 = mutate(child1, n, k, mutation_rate)
            child2 = mutate(child2, n, k, mutation_rate)
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
    # Convert numeric solutions to letter representation
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    y = len(solution)
    
    header = f"{m}-{n}-{k}-{j}-{s}-{x}-{y}"
    groups = []
    for group in solution:
        group_str = ','.join(letters[i] for i in group)
        groups.append(group_str)
    
    return header + '\n' + '\n'.join(f"{i+1}.{group}" for i, group in enumerate(groups))

# Example usage
def main():
    
    [n, k, j, s, m, x] = test_case_4()
    
    best_solution = genetic_algorithm(n, k, j, s)
    output = format_output(best_solution, m, n, k, j, s, x)
    print(output)

if __name__ == "__main__":
    main()