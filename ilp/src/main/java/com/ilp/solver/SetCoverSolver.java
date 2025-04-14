package com.ilp.solver;

import com.ilp.solver.config.ProblemConfig;
import com.ilp.solver.util.CombinationGenerator;
import com.google.ortools.Loader;
import com.google.ortools.linearsolver.MPSolver;
import com.google.ortools.linearsolver.MPVariable;
import com.google.ortools.linearsolver.MPConstraint;
import com.google.ortools.linearsolver.MPObjective;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 集合覆盖问题的ILP求解器
 */
public class SetCoverSolver {
    private final ProblemConfig config;
    private final ArrayList<Character> samples;
    private final ArrayList<HashSet<Character>> largeSets;
    private final ArrayList<HashSet<Character>> mediumSets;
    private final ConcurrentHashMap<Integer, ArrayList<HashSet<Character>>> smallSetsMap;
    private static final double INFINITY = Double.POSITIVE_INFINITY;
    private final Map<Set<Character>, Boolean> containmentCache = new ConcurrentHashMap<>();

    /**
     * 创建求解器实例
     *
     * @param config 问题配置
     */
    public SetCoverSolver(ProblemConfig config) {
        this.config = config;
        this.samples = generateSamples();
        
        // 预计算组合数量
        int largeSetCount = CombinationGenerator.combinationCount(config.getSelectedSamples(), config.getLargeSetSize());
        int mediumSetCount = CombinationGenerator.combinationCount(config.getSelectedSamples(), config.getMediumSetSize());
        
        // 预分配集合大小
        this.largeSets = new ArrayList<>(largeSetCount);
        this.mediumSets = new ArrayList<>(mediumSetCount);
        this.smallSetsMap = new ConcurrentHashMap<>(mediumSetCount);
        
        // 生成组合
        generateCombinations();
        
        // 打印初始化信息
        System.out.println("初始化完成:");
        System.out.println("- 生成的大集合数量: " + largeSets.size());
        System.out.println("- 生成的中集合数量: " + mediumSets.size());
        System.out.println("- 生成的小集合映射大小: " + smallSetsMap.size());
        System.out.println();
    }

    private void generateCombinations() {
        // 生成大集合
        List<Set<Character>> largeCombinations = CombinationGenerator.generateCombinations(
                samples, config.getLargeSetSize());
        largeCombinations.forEach(set -> largeSets.add(new HashSet<>(set)));

        // 生成中集合
        List<Set<Character>> mediumCombinations = CombinationGenerator.generateCombinations(
                samples, config.getMediumSetSize());
        mediumCombinations.forEach(set -> mediumSets.add(new HashSet<>(set)));

        // 生成小集合映射
        for (int i = 0; i < mediumSets.size(); i++) {
            ArrayList<Character> mediumSetList = new ArrayList<>(mediumSets.get(i));
            List<Set<Character>> smallCombinations = CombinationGenerator.generateCombinations(
                    mediumSetList, config.getSmallSetSize());
            ArrayList<HashSet<Character>> smallSets = new ArrayList<>(smallCombinations.size());
            smallCombinations.forEach(set -> smallSets.add(new HashSet<>(set)));
            smallSetsMap.put(i, smallSets);
        }
    }

    /**
     * 生成样本集合
     *
     * @return 样本列表
     */
    private ArrayList<Character> generateSamples() {
        ArrayList<Character> result = new ArrayList<>(config.getSelectedSamples());
        IntStream.range(0, config.getSelectedSamples())
                .forEach(i -> result.add((char)('A' + i)));
        return result;
    }

    /**
     * 检查大集合是否包含任何小集合
     */
    private boolean containsAnySmallSet(Set<Character> largeSet, List<Set<Character>> smallSets) {
        for (Set<Character> smallSet : smallSets) {
            if (largeSet.containsAll(smallSet)) {
                return true;
            }
        }
        return false;
    }

    /**
     * 预计算约束条件
     */
    private List<List<Integer>> precomputeConstraints() {
        List<List<Integer>> constraintCoefficients = new ArrayList<>(mediumSets.size());
        int totalNonZeroCoefficients = 0;
        
        for (int j = 0; j < mediumSets.size(); j++) {
            List<Integer> coefficients = new ArrayList<>();
            List<Set<Character>> smallSets = new ArrayList<>(smallSetsMap.get(j));
            
            for (int i = 0; i < largeSets.size(); i++) {
                if (containsAnySmallSet(largeSets.get(i), smallSets)) {
                    coefficients.add(i);
                }
            }
            
            constraintCoefficients.add(coefficients);
            totalNonZeroCoefficients += coefficients.size();
            
            if (j < 5) {
                System.out.println("约束 " + j + ": 非零系数数量 = " + coefficients.size());
            }
        }
        
        System.out.println("总非零系数数量: " + totalNonZeroCoefficients);
        return constraintCoefficients;
    }

    /**
     * 求解集合覆盖问题
     *
     * @return 求解结果
     */
    public SetCoverSolution solve() {
        try {
            // 加载 OR-Tools 本地库
            Loader.loadNativeLibraries();
            
            System.out.println("开始构建整数线性规划问题...");
            
            // 创建求解器
            MPSolver solver = MPSolver.createSolver("SCIP");
            if (solver == null) {
                System.err.println("无法创建求解器");
                return null;
            }

            // 创建变量（每个大集合对应一个二进制变量）
            MPVariable[] x = new MPVariable[largeSets.size()];
            for (int i = 0; i < largeSets.size(); i++) {
                x[i] = solver.makeBoolVar("x[" + i + "]");
            }
            System.out.println("创建了 " + largeSets.size() + " 个二进制变量");

            // 预计算约束条件
            List<List<Integer>> constraintCoefficients = precomputeConstraints();
            
            // 添加约束条件
            for (int j = 0; j < constraintCoefficients.size(); j++) {
                List<Integer> coefficients = constraintCoefficients.get(j);
                if (!coefficients.isEmpty()) {
                    MPConstraint constraint = solver.makeConstraint(1.0, INFINITY);
                    for (int i : coefficients) {
                        constraint.setCoefficient(x[i], 1);
                    }
                }
            }

            // 设置目标函数（最小化选中的集合数量）
            MPObjective objective = solver.objective();
            for (int i = 0; i < largeSets.size(); i++) {
                objective.setCoefficient(x[i], 1);
            }
            objective.setMinimization();

            System.out.println("开始求解...");
            MPSolver.ResultStatus status = solver.solve();

            if (status == MPSolver.ResultStatus.OPTIMAL) {
                System.out.println("找到最优解！");
                System.out.println("目标函数值: " + objective.value());
                
                // 构建结果
                ArrayList<Set<Character>> selectedSets = new ArrayList<>();
                for (int i = 0; i < x.length; i++) {
                    if (x[i].solutionValue() > 0.5) {
                        selectedSets.add(new HashSet<>(largeSets.get(i)));
                    }
                }
                
                return new SetCoverSolution(objective.value(), selectedSets);
            } else {
                System.out.println("求解器未找到最优解，状态: " + status);
                return null;
            }
        } catch (Exception e) {
            System.err.println("求解过程中出现错误: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }
} 