package com.ilp.solver.config;

/**
 * 问题配置类，用于存储ILP问题的参数
 */
public class ProblemConfig {
    private final int totalSamples;      // 样本总数 (45 <= m <= 54)
    private final int selectedSamples;   // 随机选择的样本数 (7 <= n <= 25)
    private final int largeSetSize;      // 大集合的大小 (4 <= k <= 7)
    private final int mediumSetSize;     // 中集合的大小 (s <= j <= k)
    private final int smallSetSize;      // 最小集合的大小 (3 <= s <= 7)

    /**
     * 创建问题配置实例
     *
     * @param totalSamples 样本总数
     * @param selectedSamples 随机选择的样本数
     * @param largeSetSize 大集合的大小
     * @param mediumSetSize 中集合的大小
     * @param smallSetSize 最小集合的大小
     */
    public ProblemConfig(int totalSamples, int selectedSamples, int largeSetSize, 
                        int mediumSetSize, int smallSetSize) {
        validateParameters(totalSamples, selectedSamples, largeSetSize, mediumSetSize, smallSetSize);
        this.totalSamples = totalSamples;
        this.selectedSamples = selectedSamples;
        this.largeSetSize = largeSetSize;
        this.mediumSetSize = mediumSetSize;
        this.smallSetSize = smallSetSize;
    }

    private void validateParameters(int totalSamples, int selectedSamples, int largeSetSize,
                                  int mediumSetSize, int smallSetSize) {
        if (totalSamples < 45 || totalSamples > 54) {
            throw new IllegalArgumentException("Total samples must be between 45 and 54");
        }
        if (selectedSamples < 7 || selectedSamples > 25) {
            throw new IllegalArgumentException("Selected samples must be between 7 and 25");
        }
        if (largeSetSize < 4 || largeSetSize > 7) {
            throw new IllegalArgumentException("Large set size must be between 4 and 7");
        }
        if (mediumSetSize < smallSetSize || mediumSetSize > largeSetSize) {
            throw new IllegalArgumentException("Medium set size must be between small set size and large set size");
        }
        if (smallSetSize < 3 || smallSetSize > 7) {
            throw new IllegalArgumentException("Small set size must be between 3 and 7");
        }
    }

    // Getters
    public int getTotalSamples() {
        return totalSamples;
    }

    public int getSelectedSamples() {
        return selectedSamples;
    }

    public int getLargeSetSize() {
        return largeSetSize;
    }

    public int getMediumSetSize() {
        return mediumSetSize;
    }

    public int getSmallSetSize() {
        return smallSetSize;
    }
} 