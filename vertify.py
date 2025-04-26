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
        (1,2,6,10,13,14),    # B,C,G,K,N,O
        (1,2,6,7,9,14),      # B,C,G,H,J,O
        (0,6,8,10,11,12),    # A,G,I,K,L,M
        (0,3,7,9,10,11),     # A,D,H,J,K,L
        (1,2,11,12,13,14),   # B,C,L,M,N,O
        (1,2,3,7,8,11),      # B,C,D,H,I,L
        (1,3,5,6,7,10),      # B,D,F,G,H,K
        (0,4,9,10,12,14),    # A,E,J,K,M,O
        (2,3,10,11,12,14),   # C,D,K,L,M,O
        (0,2,3,4,9,13),      # A,C,D,E,J,N
        (4,5,7,8,9,13),      # E,F,H,I,J,N
        (0,1,3,8,10,13),     # A,B,D,I,K,N
        (0,3,5,9,12,14),     # A,D,F,J,M,O
        (0,1,2,5,9,10),      # A,B,C,F,J,K
        (0,3,5,6,11,13),     # A,D,F,G,L,N
        (1,2,3,6,8,14),      # B,C,D,G,I,O
        (0,4,8,9,12,13),     # A,E,I,J,M,N
        (3,5,7,10,12,13),    # D,F,H,K,M,N
        (1,2,3,4,7,13),      # B,C,D,E,H,N
        (0,4,6,9,10,11),     # A,E,G,J,K,L
        (2,5,6,8,13,14),     # C,F,G,I,N,O
        (1,2,4,5,9,11),      # B,C,E,F,J,L
        (1,3,4,5,8,14),      # B,D,E,F,I,O
        (1,2,7,8,10,12),     # B,C,H,I,K,M
        (0,6,8,9,11,13),     # A,G,I,J,L,N
        (0,4,5,7,11,14),     # A,E,F,H,L,O
        (1,2,5,6,9,12),      # B,C,F,G,J,M
        (0,3,4,6,7,12),      # A,D,E,G,H,M
        (7,8,10,11,13,14)    # H,I,K,L,N,O
    ]
    
    # Test parameters
    n, k, j, s = 15, 6, 6, 4  # 15 elements (A-O), 6 per group
    result = verify_coverage(solution, n, k, j, s, strict_coverage=False, min_cover=1)
    
    return result

if __name__ == "__main__":
    print(test_verify())