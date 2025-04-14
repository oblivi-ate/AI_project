package com.ilp.solver;

import com.ilp.solver.config.ProblemConfig;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SetCoverSolverTest {
    @Test
    void testSolve() {
        // 创建问题配置
        ProblemConfig config = new ProblemConfig(45, 8, 6, 4, 4);

        // 创建求解器
        SetCoverSolver solver = new SetCoverSolver(config);

        // 求解
        SetCoverSolution solution = solver.solve();

        // 验证结果
        assertNotNull(solution);
        assertTrue(solution.getObjectiveValue() > 0);
        assertFalse(solution.getSelectedSets().isEmpty());
        
        // 验证选中的集合大小
        solution.getSelectedSets().forEach(set -> 
            assertEquals(6, set.size(), "每个选中的集合大小应该为6")
        );
    }
} 