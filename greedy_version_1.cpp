#include <vector>
#include <algorithm>
#include <set>
#include <functional>
#include <iostream>
using namespace std;

vector<vector<int>> generate_combinations(int n, int size) {
    vector<vector<int>> result;
    vector<int> combination(size);
    function<void(int, int)> backtrack = [&](int start, int pos) {
        if (pos == size) {
            result.push_back(combination);
            return;
        }
        for (int i = start; i <= n; ++i) {
            combination[pos] = i;
            backtrack(i + 1, pos + 1);
        }
    };
    backtrack(1, 0);
    return result;
}

vector<vector<int>> solve(int n, int k, int j, int s) {
    if (j != s) return {}; 

    auto j_groups = generate_combinations(n, j);
    auto k_groups = generate_combinations(n, k);

    set<vector<int>> uncovered(j_groups.begin(), j_groups.end());
    vector<vector<int>> solution;

    while (!uncovered.empty()) {
        int max_coverage = -1;
        vector<int> best_k_group;
        int best_index = -1;

        for (int i = 0; i < k_groups.size(); ++i) {
            const auto& kg = k_groups[i];
            set<vector<int>> covered;
            vector<int> indices(kg.size());
            for (int idx = 0; idx < kg.size(); ++idx) indices[idx] = idx;
            function<void(int, int, vector<int>&)> comb = [&](int start, int depth, vector<int>& curr) {
                if (depth == j) {
                    vector<int> jg;
                    for (int idx : curr) jg.push_back(kg[idx]);
                    sort(jg.begin(), jg.end());
                    if (uncovered.count(jg)) covered.insert(jg);
                    return;
                }
                for (int i = start; i < indices.size(); ++i) {
                    curr.push_back(indices[i]);
                    comb(i + 1, depth + 1, curr);
                    curr.pop_back();
                }
            };
            vector<int> curr;
            comb(0, 0, curr);
            int cnt = covered.size();
            if (cnt > max_coverage) {
                max_coverage = cnt;
                best_k_group = kg;
                best_index = i;
            }
        }

        if (max_coverage <= 0) break;

        solution.push_back(best_k_group);
        vector<int> indices(best_k_group.size());
        for (int idx = 0; idx < best_k_group.size(); ++idx) indices[idx] = idx;
        function<void(int, int, vector<int>&)> comb = [&](int start, int depth, vector<int>& curr) {
            if (depth == j) {
                vector<int> jg;
                for (int idx : curr) jg.push_back(best_k_group[idx]);
                sort(jg.begin(), jg.end());
                uncovered.erase(jg);
                return;
            }
            for (int i = start; i < indices.size(); ++i) {
                curr.push_back(indices[i]);
                comb(i + 1, depth + 1, curr);
                curr.pop_back();
            }
        };
        vector<int> curr;
        comb(0, 0, curr);

        if (best_index != -1) {
            k_groups.erase(k_groups.begin() + best_index);
        }
    }

    return solution;
}

int main() {
    auto solution = solve(8, 6, 4, 4);
    printf("Solution: %d\n", solution.size());
    for (const auto& group : solution) {
        for (int i = 0; i < group.size(); ++i) {
            if (i > 0) printf(" ");
            printf("%c", group[i] + 'A' - 1);
        }
        printf("\n");
    }
    system("pause");
    return 0;
}