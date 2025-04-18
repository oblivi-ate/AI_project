from pulp import *
from itertools import combinations

def is_dominated(set1, set2):
    return set1.issubset(set2) and len(set1) >= len(set2)

def check_dominated(subsets):
    non_dominated = []
    for s1 in subsets:
        dominated = False
        for s2 in subsets:
            if s1 != s2 and s2.issubset(s1) and len(s2) <= len(s1):
                dominated = True
                break
        if not dominated:
            non_dominated.append(s1)
    return non_dominated

def construct_ilp_problem(n, k, j, s):
    samples = set([chr(ord('A') + i) for i in range(n)])
    print(f"Samples: {samples}")
    
    # Generate k-subsets and filter dominated
    k_subsets = [set(comb) for comb in combinations(samples, k)]
    k_subsets = check_dominated(k_subsets)
    print(f"Filtered {k}-subsets: {len(k_subsets)}")
    
    # Generate j-subsets
    j_subsets = [set(comb) for comb in combinations(samples, j)]
    print(f"{j}-subsets: {len(j_subsets)}")
    
    # Create ILP problem
    prob = LpProblem("SetCover", LpMinimize)
    x = LpVariable.dicts("x", range(len(k_subsets)), cat='Binary')
    prob += lpSum(x[i] for i in range(len(k_subsets)))
    
    # Add constraints: each j-subset must intersect with at least one k-subset in >=s elements
    for j_subset in j_subsets:
        covering = [i for i, k_sub in enumerate(k_subsets) if len(k_sub & j_subset) >= s]
        if covering:
            prob += lpSum(x[i] for i in covering) >= 1
        else:
            print(f"No cover for {j_subset}")
            return None, None, None
    
    return prob, x, k_subsets

def main():
    n, k, j, s = 8, 6, 6, 5  # Example 4 parameters
    prob, x, k_subsets = construct_ilp_problem(n, k, j, s)
    
    if prob:
        solver = GLPK(msg=1, timeLimit=300)
        status = prob.solve(solver)
        
        if LpStatus[status] == 'Optimal':
            selected = [k_subsets[i] for i in range(len(k_subsets)) if value(x[i]) == 1]
            print(f"Minimal sets: {len(selected)}")
            for idx, sset in enumerate(selected, 1):
                print(f"{idx}. {','.join(sorted(sset))}")
        else:
            print("No optimal solution found")

if __name__ == "__main__":
    main()