import numpy as np
from typing import Callable
from tqdm import tqdm
import time

class HeatForwardSolver1D():
    def __init__(self,
                 alpha: float,
                 delta_x: float,
                 domain_length: float,
                 delta_t: float,
                 max_iter: int) -> None:

        self.alpha = alpha
        self.delta_x = delta_x
        self.num_points = round(domain_length / delta_x)
        self.delta_t = delta_t
        self.max_iter = max_iter
        self.boundaries = None

        # 稳定性条件检查（CFL条件）
        assert delta_t <= (delta_x ** 2) / (2 * alpha), "This solver config will produce unstable solution."

        # 初始化温度场矩阵 u(x,t)
        self.u = np.zeros((self.num_points, self.max_iter), dtype=np.float32)

    def set_initial(self, u0: np.ndarray):
        '''
        设置初始条件 u(x, 0)。
        u0: 一维数组，长度为 num_points
        '''
        assert u0.shape[0] == self.num_points, "Initial condition array size mismatch."
        self.u[:, 0] = u0

    def set_boundaries(self, boundaries: Callable):
        '''
        设置边界条件函数。
        boundaries(u, k, delta_t) 接受当前温度场、当前步数、和delta_t。
        '''
        self.boundaries = boundaries

    def solve(self):
        '''
        执行显式有限差分求解过程
        '''
        start_time = time.time()
        const = (self.alpha * self.delta_t) / (self.delta_x**2)

        for k in tqdm(range(self.max_iter - 1), desc="Running solver"):
            
            # 应用边界条件
            self.boundaries(self.u, k, self.delta_t)

            # 更新内部节点温度场（矢量化实现）
            self.u[1:-1, k+1] = const * (
                self.u[2:, k] + self.u[:-2, k] - 2 * self.u[1:-1, k]
            ) + self.u[1:-1, k]

        runtime = round(time.time() - start_time, 5)
        print(f"Time: {runtime} seconds")

    def get_solution(self):
        '''
        返回计算结果
        '''
        return self.u
