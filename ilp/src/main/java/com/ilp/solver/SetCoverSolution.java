package com.ilp.solver;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * 集合覆盖问题的求解结果
 */
public class SetCoverSolution {
    private final double objectiveValue;
    private final List<Set<Character>> selectedSets;

    /**
     * 创建求解结果
     *
     * @param objectiveValue 目标函数值
     * @param selectedSets 选中的集合列表
     */
    public SetCoverSolution(double objectiveValue, List<Set<Character>> selectedSets) {
        this.objectiveValue = objectiveValue;
        this.selectedSets = selectedSets;
    }

    /**
     * 获取目标函数值（最小集合数量）
     *
     * @return 目标函数值
     */
    public double getObjectiveValue() {
        return objectiveValue;
    }

    /**
     * 获取选中的集合列表
     *
     * @return 选中的集合列表
     */
    public List<Set<Character>> getSelectedSets() {
        return selectedSets;
    }

    /**
     * 将结果转换为字符串表示
     *
     * @return 结果字符串
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("最小数量: %.0f%n", objectiveValue));
        if (selectedSets.isEmpty()) {
            sb.append("未找到可行解\n");
        } else {
            sb.append("选中的集合:\n");
            selectedSets.forEach(set -> 
                sb.append(set.stream()
                    .map(String::valueOf)
                    .collect(Collectors.joining(", ", "{", "}\n")))
            );
        }
        return sb.toString();
    }
} 