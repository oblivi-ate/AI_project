from itertools import combinations


def verify_coverage(solution, n, k, j, s, strict_coverage=False, min_cover=1):
    """验证解是否正确覆盖所有组合"""
    # 基本验证
    if not solution:
        return False
    
    # 验证每个集合的大小是否为k
    for group in solution:
        if len(group) != k or not all(0 <= x < n for x in group):
            return False
    
    # 获取所有j组合并检查覆盖情况
    j_combinations = list(combinations(range(n), j))
    solution_sets = [set(group) for group in solution]
    
    for j_comb in j_combinations:
        if strict_coverage:
            # 严格覆盖模式：至少一个k集合包含所有s组合
            covered = False
            for group in solution_sets:
                s_combinations = list(combinations(j_comb, s))
                if all(set(s_comb).issubset(group) for s_comb in s_combinations):
                    covered = True
                    break
            if not covered:
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
                return False

    return True

def test_verify():
    """测试验证函数"""
    solution = [
        (0,5,6,8,9,11),    # A,F,G,I,J,L
        (1,4,7,8,9,12),    # B,E,H,I,J,M
        (1,5,9,10,11,12),  # B,F,J,K,L,M
        (2,6,7,8,9,10),    # C,G,H,I,J,K
        (0,5,6,7,11,12),   # A,F,G,H,L,M
        (0,2,3,4,5,11)     # A,C,D,E,F,L
    ]
    
    # Test parameters
    n, k, j, s = 13, 6, 6, 4  # 13 elements (A-M), 6 per group
    result = verify_coverage(solution, n, k, j, s, strict_coverage=False, min_cover=1)
    
    return result

if __name__ == "__main__":
    print(test_verify())