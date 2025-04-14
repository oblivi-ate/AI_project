package com.ilp.solver;

import com.ilp.solver.config.ProblemConfig;

/**
 * 主类，用于运行集合覆盖求解器
 */
public class Main {
    public static void main(String[] args) {
        try {
            // 测试用例1：m=45, n=7, k=6, j=5, s=5
            testCase("测试用例1", new ProblemConfig(45, 7, 6, 5, 5));
            
            // 测试用例2：m=45, n=8, k=6, j=4, s=4
            testCase("测试用例2", new ProblemConfig(45, 8, 6, 4, 4));
            
            // 测试用例3：m=45, n=9, k=6, j=4, s=4
            testCase("测试用例3", new ProblemConfig(45, 9, 6, 4, 4));
            
            // 测试用例4：m=45, n=8, k=6, j=6, s=5
            testCase("测试用例4", new ProblemConfig(45, 8, 6, 6, 5));
            
            // 测试用例5：m=45, n=8, k=6, j=6, s=5 (要求至少4个s=5的子集)
            testCase("测试用例5", new ProblemConfig(45, 8, 6, 6, 5));
            
            // 测试用例6：m=45, n=9, k=6, j=5, s=4
            testCase("测试用例6", new ProblemConfig(45, 9, 6, 5, 4));
            
            // 测试用例7：m=45, n=10, k=6, j=6, s=4
            testCase("测试用例7", new ProblemConfig(45, 10, 6, 6, 4));
            
            // 测试用例8：m=45, n=12, k=6, j=6, s=4
            testCase("测试用例8", new ProblemConfig(45, 12, 6, 6, 4));
            
        } catch (Exception e) {
            System.err.println("程序运行出错: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testCase(String caseName, ProblemConfig config) {
        System.out.println("\n=== " + caseName + " ===");
        System.out.println("配置参数:");
        System.out.println("- 样本总数(m): " + config.getTotalSamples());
        System.out.println("- 随机选择的样本数(n): " + config.getSelectedSamples());
        System.out.println("- 大集合的大小(k): " + config.getLargeSetSize());
        System.out.println("- 中集合的大小(j): " + config.getMediumSetSize());
        System.out.println("- 最小集合的大小(s): " + config.getSmallSetSize());
        System.out.println();

        // 创建求解器
        SetCoverSolver solver = new SetCoverSolver(config);

        // 求解并输出结果
        SetCoverSolution solution = solver.solve();
        
        if (solution != null) {
            System.out.println("求解成功！");
            System.out.println(solution);
            
            if (solution.getSelectedSets().isEmpty()) {
                System.out.println("警告：未找到任何满足条件的集合");
            }
        } else {
            System.out.println("求解失败！");
        }
    }
} 