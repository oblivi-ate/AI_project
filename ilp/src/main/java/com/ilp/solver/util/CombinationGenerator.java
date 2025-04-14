package com.ilp.solver.util;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 组合生成工具类，用于生成各种大小的子集
 */
public class CombinationGenerator {
    
    /**
     * 计算组合数 C(n, k)
     *
     * @param n 总数
     * @param k 选择数
     * @return 组合数
     */
    public static int combinationCount(int n, int k) {
        if (k < 0 || k > n) {
            return 0;
        }
        if (k == 0 || k == n) {
            return 1;
        }
        k = Math.min(k, n - k); // 利用对称性减少计算
        long result = 1;
        for (int i = 1; i <= k; i++) {
            result = result * (n - k + i) / i;
        }
        return (int) result;
    }
    
    /**
     * 生成指定大小的所有组合
     *
     * @param <T> 元素类型
     * @param input 输入集合
     * @param size 组合大小
     * @return 所有可能的组合列表
     */
    public static <T> List<Set<T>> generateCombinations(List<T> input, int size) {
        if (input == null || size <= 0 || size > input.size()) {
            throw new IllegalArgumentException("Invalid input parameters");
        }
        
        List<Set<T>> result = new ArrayList<>();
        generateCombinationsHelper(input, size, 0, new HashSet<>(), result);
        return result;
    }

    /**
     * 递归生成组合的辅助方法
     *
     * @param <T> 元素类型
     * @param input 输入集合
     * @param size 目标大小
     * @param start 起始索引
     * @param current 当前组合
     * @param result 结果列表
     */
    private static <T> void generateCombinationsHelper(List<T> input, int size, 
            int start, Set<T> current, List<Set<T>> result) {
        if (current.size() == size) {
            result.add(new HashSet<>(current));
            return;
        }
        
        for (int i = start; i < input.size(); i++) {
            current.add(input.get(i));
            generateCombinationsHelper(input, size, i + 1, current, result);
            current.remove(input.get(i));
        }
    }
} 