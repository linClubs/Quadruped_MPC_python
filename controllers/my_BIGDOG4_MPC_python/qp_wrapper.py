import numpy as np
import scipy.sparse as sp
import osqp


def solve_qp(H, g, A=None, lb=None, ub=None, lbA=None, ubA=None):
    """
    使用 OSQP 求解二次规划, 替代 MATLAB qpOASES

    求解: min 0.5 * x'Hx + g'x
    s.t.  lb <= x <= ub          (变量边界, 可为 None)
          lbA <= A*x <= ubA       (线性约束, 可为 None)

    返回: x (最优解向量)
    """
    n = len(g)

    # 确保 H 为对称稀疏矩阵
    H_sym = (np.array(H) + np.array(H).T) / 2.0
    P = sp.csc_matrix(H_sym)
    q = np.array(g, dtype=np.float64)

    # 构造 OSQP 约束: l <= A_all * x <= u
    A_blocks = []
    l_blocks = []
    u_blocks = []

    # 变量边界 lb <= x <= ub
    if lb is not None and ub is not None:
        lb_arr = np.array(lb, dtype=np.float64).flatten()
        ub_arr = np.array(ub, dtype=np.float64).flatten()
        A_blocks.append(sp.eye(n, format='csc'))
        l_blocks.append(lb_arr)
        u_blocks.append(ub_arr)

    # 线性约束 lbA <= A*x <= ubA
    if A is not None and lbA is not None and ubA is not None:
        A_arr = np.array(A, dtype=np.float64)
        if A_arr.size > 0:
            A_blocks.append(sp.csc_matrix(A_arr))
            l_blocks.append(np.array(lbA, dtype=np.float64).flatten())
            u_blocks.append(np.array(ubA, dtype=np.float64).flatten())

    if len(A_blocks) > 0:
        A_all = sp.vstack(A_blocks, format='csc')
        l_all = np.concatenate(l_blocks)
        u_all = np.concatenate(u_blocks)
    else:
        A_all = sp.csc_matrix((0, n))
        l_all = np.array([])
        u_all = np.array([])

    # 替换 None / NaN 为 ±inf
    l_all = np.where(np.isnan(l_all), -np.inf, l_all)
    u_all = np.where(np.isnan(u_all), np.inf, u_all)

    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A_all, l=l_all, u=u_all,
               eps_abs=1e-8, eps_rel=1e-8, verbose=False,
               max_iter=10000, polish=True, polish_refine_iter=3)
    res = prob.solve()

    if res.info.status_val not in (1,):  # 1 = solved
        # 如果未收敛, 尝试放宽容差重解
        prob.setup(P=P, q=q, A=A_all, l=l_all, u=u_all,
                   eps_abs=1e-4, eps_rel=1e-4, verbose=False,
                   max_iter=20000, polish=True)
        res = prob.solve()

    return np.array(res.x)
