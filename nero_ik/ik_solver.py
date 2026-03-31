"""
Tsinghua 论文落地版：可行域（关节限位）内扫描臂角 → 加权限位惩罚下选最优臂角。
升级：新增1D QP优化，在解析解基础上局部优化关节角
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar  # 用于1D QP求解


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def pose_error(T_cur: np.ndarray, T_des: np.ndarray) -> np.ndarray:
    dp = T_cur[:3, 3] - T_des[:3, 3]
    R = T_cur[:3, :3]
    Rd = T_des[:3, :3]
    re = 0.5 * (
        np.cross(R[:, 0], Rd[:, 0]) + np.cross(R[:, 1], Rd[:, 1]) + np.cross(R[:, 2], Rd[:, 2])
    )
    return np.hstack([dp, re])


def _rotz(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _rotx(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _dh_A(theta: float, d: float, a_prev: float, alpha_prev: float) -> np.ndarray:
    # NOTE: YAIK/NERO uses Modified DH:
    # T_i^{i-1} = RotX(alpha_{i-1}) * TransX(a_{i-1}) * RotZ(theta_i) * TransZ(d_i)
    T = np.eye(4)
    Rx = _rotx(alpha_prev)
    Rz = _rotz(theta)
    T[:3, :3] = Rx @ Rz
    T[:3, 3] = np.array(
        [a_prev, -math.sin(alpha_prev) * d, math.cos(alpha_prev) * d],
        dtype=float,
    )
    return T


def _invert_rigid_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Rt = R.T
    out = np.eye(4, dtype=float)
    out[:3, :3] = Rt
    out[:3, 3] = -(Rt @ t)
    return out


def _remove_post_transform(T07: np.ndarray, p: "NeroParams") -> np.ndarray:
    """
    Equivalent to T07 @ inv(T_post) where T_post = TransZ(post_transform_d8),
    but avoids building/inverting a matrix each call.
    """
    T_chain = np.array(T07, dtype=float, copy=True)
    T_chain[:3, 3] = T07[:3, 3] - float(p.post_transform_d8) * T07[:3, 2]
    return T_chain


def _axis_angle_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    u = axis / max(np.linalg.norm(axis), 1e-12)
    ux, uy, uz = u
    K = np.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]], dtype=float)
    I = np.eye(3)
    return I + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


@dataclass
class NeroParams:
    # Keep this exactly as paper-style DH used for NERO experiments here.
    a_prev: np.ndarray
    alpha_prev: np.ndarray
    d_i: np.ndarray
    theta_offset: np.ndarray
    joint_limits: np.ndarray
    post_transform_d8: float
    theta0_coarse_divisor: int
    theta0_fine_count: int

    @staticmethod
    def default() -> "NeroParams":
        return NeroParams(
            a_prev=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
            alpha_prev=np.deg2rad(np.array([0.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0], dtype=float)),
            d_i=np.array([0.138, 0.0, 0.31, 0.0, 0.27, 0.0, 0.0235], dtype=float),
            theta_offset=np.deg2rad(np.array([0.0, -180.0, -180.0, -180.0, 90.0, 90.0, 0.0], dtype=float)),
            joint_limits=np.array(
                [
                    [-2.705261, 2.705261],
                    [-1.745330, 1.745330],
                    [-2.757621, 2.757621],
                    [-1.012291, 2.146755],
                    [-2.757621, 2.757621],
                    [-0.733039, 0.959932],
                    [-1.570797, 1.570797],
                ],
                dtype=float,
            ),
            post_transform_d8=-0.0235,
            theta0_coarse_divisor=3,
            theta0_fine_count=41,
        )


@dataclass
class ContinuityParams:
    # Local theta0 tracking window around previous selected theta0.
    local_theta0_window: float = 0.35
    local_theta0_count: int = 41
    # Candidate score: w_vel*|dq| + w_acc*|ddq| + w_pose*pose_err + w_theta0*|dtheta0|
    w_vel: float = 1.0
    w_acc: float = 0.25
    w_pose: float = 0.1
    w_theta0: float = 0.15
    # Hysteresis: keep locked branch unless significantly better candidate appears.
    hysteresis_margin: float = 0.03
    # Fallback to global scan when local window fails.
    enable_global_fallback: bool = True
    # 1D QP权重
    w_qp_joint_inc: float = 1.0  # 关节增量权重
    w_qp_pose_err: float = 0.5   # 位姿误差权重


@dataclass
class ContinuityRuntimeState:
    q_prev: np.ndarray
    q_prev2: Optional[np.ndarray] = None
    theta0_prev: Optional[float] = None
    q_lock: Optional[np.ndarray] = None


def fk_all(q: np.ndarray, p: NeroParams) -> List[np.ndarray]:
    Ts = [np.eye(4)]
    T = np.eye(4)
    for i in range(7):
        theta = q[i] + p.theta_offset[i]
        T = T @ _dh_A(theta, p.d_i[i], p.a_prev[i], p.alpha_prev[i])
        Ts.append(T.copy())
    return Ts


def fk(q: np.ndarray, p: Optional[NeroParams] = None) -> np.ndarray:
    if p is None:
        p = NeroParams.default()
    T = fk_all(q, p)[-1]
    T_post = np.eye(4)
    T_post[2, 3] = p.post_transform_d8
    return T @ T_post


def _within_limits(q: np.ndarray, limits: np.ndarray) -> bool:
    return bool(np.all(q >= limits[:, 0] - 1e-8) and np.all(q <= limits[:, 1] + 1e-8))


def _extract_123_from_R03_paper(R03: np.ndarray) -> List[np.ndarray]:
    # From paper eq.(21)-style extraction:
    # theta2 = +/- acos(R33), theta1 = atan2(-R23/s2, -R13/s2), theta3 = atan2(-R32/s2, R31/s2)
    sols = []
    c2 = float(np.clip(R03[2, 2], -1.0, 1.0))
    for sgn in (1.0, -1.0):
        s2 = sgn * math.sqrt(max(0.0, 1.0 - c2 * c2))
        if abs(s2) < 1e-8:
            continue
        th2 = math.atan2(s2, c2)
        th1 = math.atan2(float(-R03[1, 2] / s2), float(-R03[0, 2] / s2))
        th3 = math.atan2(float(-R03[2, 1] / s2), float(R03[2, 0] / s2))
        sols.append(np.array([th1, th2, th3], dtype=float))
    return sols


def _extract_567_from_T47_paper(T47: np.ndarray) -> List[np.ndarray]:
    # MDH analytic extraction for raw wrist angles (theta5, theta6, theta7):
    # R47(2,3) = cos(theta6)
    # R47(1,3) = sin(theta6)*cos(theta5), R47(3,3)=sin(theta5)*sin(theta6)
    # R47(2,1) = -sin(theta6)*cos(theta7), R47(2,2)=sin(theta6)*sin(theta7)
    sols = []
    c6 = float(np.clip(T47[1, 2], -1.0, 1.0))
    for sgn in (1.0, -1.0):
        s6 = sgn * math.sqrt(max(0.0, 1.0 - c6 * c6))
        if abs(s6) < 1e-8:
            continue
        th6 = math.atan2(s6, c6)
        th5 = math.atan2(float(T47[2, 2] / s6), float(T47[0, 2] / s6))
        th7 = math.atan2(float(T47[1, 1] / s6), float(-T47[1, 0] / s6))
        sols.append(np.array([th5, th6, th7], dtype=float))
    return sols


def _solve_q123_from_swe(E: np.ndarray, W: np.ndarray, q4: float, p: NeroParams) -> List[np.ndarray]:
    """
    Solve q1,q2,q3 from elbow point E and wrist center W for MDH NERO chain.
    """
    d0 = float(p.d_i[0])
    d2 = float(p.d_i[2])
    d4 = float(p.d_i[4])
    if abs(d2) < 1e-12 or abs(d4) < 1e-12:
        return []

    Ex, Ey, Ez = float(E[0]), float(E[1]), float(E[2])
    rho = math.hypot(Ex, Ey)
    c2 = (Ez - d0) / d2
    if c2 < -1.0 - 1e-8 or c2 > 1.0 + 1e-8:
        return []
    c2 = float(np.clip(c2, -1.0, 1.0))
    s2_abs = math.sqrt(max(0.0, 1.0 - c2 * c2))
    if rho > abs(d2) + 1e-7:
        return []

    v = W - E
    n_v = float(np.linalg.norm(v))
    if n_v < 1e-10:
        return []
    col2 = -v / d4
    # Normalize against numeric drift.
    col2 = col2 / max(np.linalg.norm(col2), 1e-12)
    u1, u2, u3 = float(col2[0]), float(col2[1]), float(col2[2])

    s4 = math.sin(q4)
    c4 = math.cos(q4)
    if abs(s4) < 1e-8:
        return []

    sols: List[np.ndarray] = []
    for s2 in (s2_abs, -s2_abs):
        if abs(s2) < 1e-10:
            continue
        c1 = -Ex / (d2 * s2)
        s1 = -Ey / (d2 * s2)
        n1 = math.hypot(c1, s1)
        if n1 < 1e-12:
            continue
        c1 /= n1
        s1 /= n1
        q1 = math.atan2(s1, c1)
        q2 = math.atan2(s2, c2)

        b1 = (s2 * c1 * c4 - u1) / s4
        b2 = (u2 - s1 * s2 * c4) / s4
        s3 = s1 * b1 + c1 * b2
        c2c3 = -c1 * b1 + s1 * b2
        if abs(c2) > 1e-8:
            c3 = c2c3 / c2
        else:
            c3 = (u3 + c2 * c4) / (s2 * s4)
        n3 = math.hypot(s3, c3)
        if n3 < 1e-12:
            continue
        s3 /= n3
        c3 /= n3
        q3 = math.atan2(s3, c3)
        sols.append(np.array([q1, q2, q3], dtype=float))
    return sols


def _solve_theta4_from_triangle(S: np.ndarray, W: np.ndarray, p: NeroParams) -> Optional[float]:
    # Paper geometric triangle SEW
    l_sw = float(np.linalg.norm(W - S))
    l_se = abs(float(p.d_i[2]))
    l_ew = abs(float(p.d_i[4]))
    if l_sw < 1e-10:
        return None
    c4 = (l_sw * l_sw - l_se * l_se - l_ew * l_ew) / (2.0 * l_se * l_ew)
    if c4 < -1.0 or c4 > 1.0:
        return None
    c4 = float(np.clip(c4, -1.0, 1.0))
    return math.acos(c4)


def _compute_swe_from_target(T07: np.ndarray, p: NeroParams) -> Tuple[np.ndarray, np.ndarray, Optional[float], np.ndarray]:
    """
    Build S-W-E triangle primitives from target pose:
      - S: shoulder center
      - W: wrist center
      - q4_abs from triangle law of cosines
      - u_sw: unit vector from S to W
    """
    R = T07[:3, :3]
    p_target = T07[:3, 3]
    z7 = R[:, 2]
    d6 = float(p.d_i[6])
    d1 = float(p.d_i[0])

    O7 = p_target - p.post_transform_d8 * z7
    # For this MDH chain, frame-5 origin (wrist center W) is along z7 from O7 by d6.
    W = O7 - d6 * z7
    S = np.array([0.0, 0.0, d1], dtype=float)

    q4_abs = _solve_theta4_from_triangle(S, W, p)
    v_sw = W - S
    n_sw = float(np.linalg.norm(v_sw))
    if n_sw < 1e-12:
        u_sw = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        u_sw = v_sw / n_sw
    return S, W, q4_abs, u_sw


def _elbow_from_arm_angle(S: np.ndarray, W: np.ndarray, theta0: float, p: NeroParams) -> Optional[np.ndarray]:
    """
    Construct elbow point E(theta0) on the SEW circle.
    """
    l_se = abs(float(p.d_i[2]))
    l_ew = abs(float(p.d_i[4]))
    sw = W - S
    l_sw = float(np.linalg.norm(sw))
    if l_sw < 1e-12:
        return None
    u_sw = sw / l_sw

    # Circle center on SW line and radius from triangle geometry.
    x = (l_se * l_se - l_ew * l_ew + l_sw * l_sw) / (2.0 * l_sw)
    r2 = l_se * l_se - x * x
    if r2 < -1e-10:
        return None
    r = math.sqrt(max(0.0, r2))
    C = S + x * u_sw

    # Reference plane basis: use OS direction per paper-style definition.
    os_vec = S.copy()
    t = np.cross(os_vec, u_sw)
    if np.linalg.norm(t) < 1e-10:
        t = np.cross(np.array([1.0, 0.0, 0.0], dtype=float), u_sw)
    if np.linalg.norm(t) < 1e-10:
        t = np.cross(np.array([0.0, 1.0, 0.0], dtype=float), u_sw)
    e1 = t / max(np.linalg.norm(t), 1e-12)
    e2 = np.cross(u_sw, e1)
    e2 = e2 / max(np.linalg.norm(e2), 1e-12)
    E = C + r * (math.cos(theta0) * e1 + math.sin(theta0) * e2)
    return E


def _weight_limits(q: float, q_min: float, q_max: float) -> float:
    """清华论文式(20)：越靠近限位，权重越大。"""
    span = q_max - q_min
    if span < 1e-12:
        return 1.0
    x = 2.0 * (q - (q_min + q_max) * 0.5) / span
    a = 2.38
    b = 2.28
    if x >= 0:
        if x >= 1.0:
            return 1e6
        den = math.exp(a * (1.0 - x)) - 1.0
        if abs(den) < 1e-12:
            return 1e6
        return b * x / den
    if x <= -1.0:
        return 1e6
    den = math.exp(a * (1.0 + x)) - 1.0
    if abs(den) < 1e-12:
        return 1e6
    return -b * x / den


def _get_theta0_feasible_region(
    T07: np.ndarray, p: NeroParams, step: float = 0.01
) -> List[float]:
    """
    臂角可行域：在 [-pi, pi) 上离散扫描，若该臂角下存在在限位内的解析解则计入。
    使用 `_ik_one_arm_angle`（与原文一致的几何分支），等价于文档中“用解析关节角判断限位”。
    """
    feasible: List[float] = []
    for theta0 in np.arange(-math.pi, math.pi, step):
        t = float(theta0)
        if _ik_one_arm_angle(T07, t, p):
            feasible.append(t)
    return feasible


def _optimal_theta0(
    feasible_theta0: List[float], T07: np.ndarray, p: NeroParams, q_prev: np.ndarray
) -> float:
    """在可行臂角集合上，按加权关节增量平方和最小选最优臂角（论文 4.1 风格）。"""
    best_cost = float("inf")
    best_t = feasible_theta0[0]
    for t in feasible_theta0:
        sols = _ik_one_arm_angle(T07, float(t), p)
        for q_full in sols:
            q = q_full[:7]
            cost = 0.0
            for i in range(7):
                lo, hi = float(p.joint_limits[i, 0]), float(p.joint_limits[i, 1])
                w = _weight_limits(q[i], lo, hi)
                dq = abs(q[i] - q_prev[i])
                cost += w * dq * dq
            if cost < best_cost:
                best_cost = cost
                best_t = float(t)
    return best_t


def _scan_theta0_solutions(T07: np.ndarray, p: NeroParams, step: float) -> Tuple[List[float], List[Tuple[float, List[np.ndarray]]]]:
    """
    One-pass theta0 sweep:
      - returns feasible theta0 list
      - and cached per-theta solutions to avoid recomputing _ik_one_arm_angle
    """
    feasible_theta0: List[float] = []
    cached: List[Tuple[float, List[np.ndarray]]] = []
    for theta0 in np.arange(-math.pi, math.pi, step):
        t = float(theta0)
        sols = _ik_one_arm_angle(T07, t, p)
        if sols:
            feasible_theta0.append(t)
            cached.append((t, sols))
    return feasible_theta0, cached


def _best_weighted_from_cached(
    cached: List[Tuple[float, List[np.ndarray]]], p: NeroParams, q_prev: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    Select global best candidate over cached theta0 branches with original weighted rule.
    Returns (q_best_7d, q_best_full, theta0_selected).
    """
    best_cost = float("inf")
    q_best: Optional[np.ndarray] = None
    q_best_full: Optional[np.ndarray] = None
    theta0_best: Optional[float] = None
    for t, sols in cached:
        for q_full in sols:
            q = q_full[:7]
            cost = 0.0
            for i in range(7):
                lo, hi = float(p.joint_limits[i, 0]), float(p.joint_limits[i, 1])
                w = _weight_limits(q[i], lo, hi)
                dq = abs(q[i] - q_prev[i])
                cost += w * dq * dq
            if cost < best_cost:
                best_cost = cost
                q_best = q.copy()
                q_best_full = q_full
                theta0_best = t
    return q_best, q_best_full, theta0_best


def _best_q_at_theta0_weighted(
    theta0: float, T07: np.ndarray, p: NeroParams, q_prev: np.ndarray
) -> Optional[np.ndarray]:
    """同一臂角下多分支时，取加权代价最小的 7 维关节向量。"""
    sols = _ik_one_arm_angle(T07, float(theta0), p)
    best_cost = float("inf")
    best_q: Optional[np.ndarray] = None
    for q_full in sols:
        q = q_full[:7]
        cost = 0.0
        for i in range(7):
            lo, hi = float(p.joint_limits[i, 0]), float(p.joint_limits[i, 1])
            w = _weight_limits(q[i], lo, hi)
            dq = abs(q[i] - q_prev[i])
            cost += w * dq * dq
        if cost < best_cost:
            best_cost = cost
            best_q = q.copy()
    return best_q


# ------------------- 新增1D QP优化函数 -------------------
def _qp_1d_objective(dq: float, idx: int, q_base: np.ndarray, T_target: np.ndarray, p: NeroParams, continuity: ContinuityParams) -> float:
    """
    1D QP目标函数：最小化关节增量 + 位姿误差
    :param dq: 单个关节的增量（优化变量）
    :param idx: 待优化的关节索引
    :param q_base: 解析解得到的基础关节角
    :param T_target: 目标位姿
    :param p: 机器人DH参数
    :param continuity: 权重参数
    :return: 目标函数值
    """
    # 构造新的关节角
    q_new = q_base.copy()
    q_new[idx] = wrap_to_pi(q_new[idx] + dq)
    
    # 1. 关节限位惩罚（硬约束通过优化边界实现，这里加软惩罚）
    q_min, q_max = p.joint_limits[idx]
    if q_new[idx] < q_min or q_new[idx] > q_max:
        limit_penalty = 1e6 * abs(q_new[idx] - np.clip(q_new[idx], q_min, q_max))
    else:
        limit_penalty = 0.0
    
    # 2. 关节增量代价
    inc_cost = continuity.w_qp_joint_inc * (dq **2)
    
    # 3. 位姿误差代价
    T_cur = fk(q_new, p)
    pose_err = np.linalg.norm(pose_error(T_cur, T_target))
    pose_cost = continuity.w_qp_pose_err * (pose_err** 2)
    
    return inc_cost + pose_cost + limit_penalty


def _optimize_q_with_1d_qp(q_init: np.ndarray, T_target: np.ndarray, p: NeroParams, continuity: ContinuityParams) -> np.ndarray:
    """
    对解析解进行1D QP优化（逐关节优化）
    :param q_init: 解析解初始关节角
    :param T_target: 目标位姿
    :param p: 机器人DH参数
    :param continuity: 权重参数
    :return: 优化后的关节角
    """
    q_opt = q_init.copy()
    for idx in range(7):
        # 关节限位作为优化边界
        q_min, q_max = p.joint_limits[idx]
        # 计算当前关节角与限位的距离，限制优化范围
        delta_max = min(0.1, q_max - q_opt[idx])  # 最大正增量（0.1弧度约5.7度）
        delta_min = max(-0.1, q_min - q_opt[idx])  # 最大负增量
        
        # 1D最小化优化
        res = minimize_scalar(
            _qp_1d_objective,
            bounds=(delta_min, delta_max),
            args=(idx, q_opt, T_target, p, continuity),
            method='bounded'
        )
        
        # 更新关节角（仅当优化成功时）
        if res.success:
            q_opt[idx] = wrap_to_pi(q_opt[idx] + res.x)
    
    return q_opt


def ik_arm_angle(
    T07: np.ndarray,
    p: Optional[NeroParams] = None,
    q_prev: Optional[np.ndarray] = None,
    n_psi: int = 181,
) -> Tuple[Optional[np.ndarray], List[float]]:
    """
    主流程：S,W → 臂角可行域 → 最优臂角 → 加权选支路 → 1D QP优化。
    `n_psi` 用于将步长限制为 `min(0.01, 2*pi/max(n_psi,1))`，与原文网格密度参数对齐。
    """
    if p is None:
        p = NeroParams.default()
    T = np.array(T07, dtype=float)
    _, _, q4_abs, _ = _compute_swe_from_target(T, p)
    if q4_abs is None:
        return None, []
    if q_prev is None:
        q_prev = np.zeros(7, dtype=float)
    step = min(0.01, 2.0 * math.pi / max(1, n_psi))
    feasible_theta0 = _get_theta0_feasible_region(T, p, step=step)
    if len(feasible_theta0) == 0:
        return None, feasible_theta0
    best_t0 = _optimal_theta0(feasible_theta0, T, p, q_prev)
    q_best = _best_q_at_theta0_weighted(best_t0, T, p, q_prev)
    
    # 新增：1D QP优化
    if q_best is not None:
        continuity = ContinuityParams()  # 使用默认QP权重
        q_best = _optimize_q_with_1d_qp(q_best, T, p, continuity)
    
    return q_best, feasible_theta0


def _ik_one_arm_angle(T07: np.ndarray, theta0: float, p: NeroParams) -> List[np.ndarray]:
    # Build S-W-E geometric primitives first (paper S-R-S flow).
    S, W_from_pose, q4_abs, _ = _compute_swe_from_target(T07, p)
    E_point = _elbow_from_arm_angle(S, W_from_pose, theta0, p)

    # Rebuild chain-frame target without post transform for downstream T47.
    T_chain = _remove_post_transform(T07, p)
    if q4_abs is None:
        return []
    W = W_from_pose

    q_solutions = []

    for q4 in (q4_abs, -q4_abs):
        if E_point is None:
            continue
        q123_sols = _solve_q123_from_swe(E_point, W, q4, p)
        for q123 in q123_sols:
            q1, q2, q3 = q123.tolist()

            # Convert original q -> raw theta used by DH transform.
            th1 = q1 + p.theta_offset[0]
            th2 = q2 + p.theta_offset[1]
            th3 = q3 + p.theta_offset[2]
            th4 = q4 + p.theta_offset[3]

            T04 = np.eye(4)
            th_raw = [th1, th2, th3, th4]
            for i in range(4):
                T04 = T04 @ _dh_A(th_raw[i], p.d_i[i], p.a_prev[i], p.alpha_prev[i])
            T47 = _invert_rigid_transform(T04) @ T_chain

            for th567 in _extract_567_from_T47_paper(T47):
                th5, th6, th7 = th567.tolist()
                theta_raw = np.array([th1, th2, th3, th4, th5, th6, th7], dtype=float)
                q = wrap_to_pi(theta_raw - p.theta_offset)
                if _within_limits(q, p.joint_limits):
                    # Attach theta0 on the fly for logging/selection.
                    # Append theta0 and geometric points for debug.
                    extra = np.array(
                        [
                            theta0,
                            S[0], S[1], S[2],
                            W[0], W[1], W[2],
                            *(E_point.tolist() if E_point is not None else [np.nan, np.nan, np.nan]),
                        ],
                        dtype=float,
                    )
                    q = np.concatenate([q, extra])
                    q_solutions.append(q)
    return q_solutions


def _q_from_theta0(theta0: float, T07: np.ndarray, p: NeroParams) -> Optional[np.ndarray]:
    """
    文档「关节角 → 臂角」逆映射的单支路表示：给定 theta0 取一条在限位内的解析解（7 维）。
    论文中的显式三角式针对特定 DH；本工程统一用 `_ik_one_arm_angle` 的 MDH 几何结果以保证与 FK 一致。
    """
    sols = _ik_one_arm_angle(T07, float(theta0), p)
    if not sols:
        return None
    return sols[0][:7].copy()


def _theta0_candidates_from_target(T: np.ndarray, n_psi: int) -> np.ndarray:
    # Paper-style arm-angle parameter full sweep in (-pi, pi]
    return np.linspace(-math.pi, math.pi, max(31, n_psi), endpoint=True)


def _collect_unique_solutions_for_theta0_grid(T: np.ndarray, p: NeroParams, theta0_grid: np.ndarray) -> List[np.ndarray]:
    all_solutions: List[np.ndarray] = []
    for theta0 in theta0_grid:
        t = float(wrap_to_pi(np.array([theta0]))[0])
        all_solutions.extend(_ik_one_arm_angle(T, t, p))
    unique_solutions: List[np.ndarray] = []
    for q in all_solutions:
        if not any(np.linalg.norm(wrap_to_pi(q[:7] - u[:7])) < 1e-4 for u in unique_solutions):
            unique_solutions.append(q)
    return unique_solutions


def ik_arm_angle_with_report(
    T_target: np.ndarray,
    p: Optional[NeroParams] = None,
    q_prev: Optional[np.ndarray] = None,
    n_psi: int = 181,
) -> Tuple[Optional[np.ndarray], List[np.ndarray], Dict[str, object]]:
    if p is None:
        p = NeroParams.default()
    if q_prev is None:
        q_prev = np.zeros(7, dtype=float)

    T = np.array(T_target, dtype=float)
    S_dbg, W_dbg, q4_dbg, _ = _compute_swe_from_target(T, p)
    step = min(0.01, 2.0 * math.pi / max(1, n_psi))
    feasible_theta0, cached = _scan_theta0_solutions(T, p, step=step)

    if len(feasible_theta0) == 0:
        return None, [], {
            "method": "failed",
            "candidate_count": 0,
            "pose_err_best": None,
            "swe_debug": {
                "S": S_dbg.tolist(),
                "W": W_dbg.tolist(),
                "E": [float("nan"), float("nan"), float("nan")],
                "q4_abs": None if q4_dbg is None else float(q4_dbg),
            },
        }

    q_best, q_best_full, best_t0 = _best_weighted_from_cached(cached, p, q_prev)
    
    # 新增：1D QP优化
    if q_best is not None:
        continuity = ContinuityParams()
        q_best = _optimize_q_with_1d_qp(q_best, T, p, continuity)

    if q_best is None:
        return None, [], {
            "method": "failed",
            "candidate_count": 0,
            "pose_err_best": None,
            "swe_debug": {
                "S": S_dbg.tolist(),
                "W": W_dbg.tolist(),
                "E": [float("nan"), float("nan"), float("nan")],
                "q4_abs": None if q4_dbg is None else float(q4_dbg),
            },
        }

    # 仅保留最优解一条分支，便于与原文接口一致；候选数用可行臂角个数表征
    if q_best_full is None or best_t0 is None:
        q_best_full = np.concatenate(
            [q_best, np.array([0.0, S_dbg[0], S_dbg[1], S_dbg[2], W_dbg[0], W_dbg[1], W_dbg[2], float("nan"), float("nan"), float("nan")], dtype=float)]
        )
    theta0_best = float(q_best_full[7] if best_t0 is None else best_t0)
    swe_debug = {
        "S": q_best_full[8:11].tolist(),
        "W": q_best_full[11:14].tolist(),
        "E": q_best_full[14:17].tolist(),
        "q4_abs": None if q4_dbg is None else float(q4_dbg),
    }
    e_best = float(np.linalg.norm(pose_error(fk(q_best, p), T)))
    all_solutions = [q_best_full]
    return q_best, all_solutions, {
        "method": "feasible_region+1DQP",
        "candidate_count": len(feasible_theta0),
        "pose_err_best": e_best,
        "theta0_selected": theta0_best,
        "feasible_theta0_count": len(feasible_theta0),
        "swe_debug": swe_debug,
    }


def ik_arm_angle_simple(
    T_target: np.ndarray,
    p: Optional[NeroParams] = None,
    q_prev: Optional[np.ndarray] = None,
    n_psi: int = 181,
) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
    q_best, all_solutions_full, _ = ik_arm_angle_with_report(T_target, p=p, q_prev=q_prev, n_psi=n_psi)
    return q_best, [q[:7] for q in all_solutions_full]


def solve_trajectory(
    T_targets: List[np.ndarray],
    p: Optional[NeroParams] = None,
    q_init: Optional[np.ndarray] = None,
    n_psi: int = 181,
) -> Tuple[List[Optional[np.ndarray]], List[Dict[str, object]]]:
    """
    Sequential analytic IK for a pose list.
    Uses previous successful joint result as continuity prior.
    """
    if p is None:
        p = NeroParams.default()
    if q_init is None:
        q_prev = np.zeros(7, dtype=float)
    else:
        q_prev = np.array(q_init, dtype=float).reshape(7)

    q_list: List[Optional[np.ndarray]] = []
    reports: List[Dict[str, object]] = []
    for T in T_targets:
        q, _, report = ik_arm_angle_with_report(np.array(T, dtype=float), p=p, q_prev=q_prev, n_psi=n_psi)
        q_list.append(q)
        reports.append(report)
        if q is not None:
            q_prev = q
    return q_list, reports


def solve_trajectory_continuous(
    T_targets: List[np.ndarray],
    p: Optional[NeroParams] = None,
    q_init: Optional[np.ndarray] = None,
    n_psi: int = 181,
    continuity: Optional[ContinuityParams] = None,
) -> Tuple[List[Optional[np.ndarray]], List[Dict[str, object]]]:
    """
    Enhanced trajectory IK:
      1) local theta0 tracking around previous theta0
      2) continuity-aware branch scoring (velocity + acceleration + pose + theta0 change)
      3) branch hysteresis to suppress sudden switches
      4) 1D QP post-optimization for each step
    Old solve_trajectory is kept unchanged.
    """
    if p is None:
        p = NeroParams.default()
    if continuity is None:
        continuity = ContinuityParams()
    if q_init is None:
        q_prev = np.zeros(7, dtype=float)
    else:
        q_prev = np.array(q_init, dtype=float).reshape(7)
    state = ContinuityRuntimeState(q_prev=q_prev)

    q_list: List[Optional[np.ndarray]] = []
    reports: List[Dict[str, object]] = []

    for T_raw in T_targets:
        q_best, report, state = solve_pose_continuous_with_state(
            np.array(T_raw, dtype=float),
            state=state,
            p=p,
            n_psi=n_psi,
            continuity=continuity,
        )
        q_list.append(q_best)
        reports.append(report)

    return q_list, reports


def solve_pose_continuous_with_state(
    T_target: np.ndarray,
    state: ContinuityRuntimeState,
    p: Optional[NeroParams] = None,
    n_psi: int = 181,
    continuity: Optional[ContinuityParams] = None,
) -> Tuple[Optional[np.ndarray], Dict[str, object], ContinuityRuntimeState]:
    """
    Single-step continuous IK with external runtime state.
    This is equivalent to one iteration of solve_trajectory_continuous.
    新增：1D QP优化
    """
    if p is None:
        p = NeroParams.default()
    if continuity is None:
        continuity = ContinuityParams()

    T = np.array(T_target, dtype=float)
    q_prev = state.q_prev
    q_prev2 = state.q_prev2
    theta0_prev = state.theta0_prev
    q_lock = state.q_lock

    all_solutions: List[np.ndarray] = []
    method = "continuous_local_theta0"
    if theta0_prev is not None:
        theta0_grid = np.linspace(
            theta0_prev - continuity.local_theta0_window,
            theta0_prev + continuity.local_theta0_window,
            max(11, continuity.local_theta0_count),
            endpoint=True,
        )
        all_solutions = _collect_unique_solutions_for_theta0_grid(T, p, theta0_grid)

    if not all_solutions and continuity.enable_global_fallback:
        q_global, all_full, _ = ik_arm_angle_with_report(T, p=p, q_prev=q_prev, n_psi=n_psi)
        all_solutions = all_full
        method = "continuous_global_fallback"
        if q_global is None:
            return None, {
                "method": method,
                "candidate_count": 0,
                "selected_by": "failed",
                "pose_err_best": None,
            }, state
    elif not all_solutions:
        return None, {
            "method": method,
            "candidate_count": 0,
            "selected_by": "failed",
            "pose_err_best": None,
        }, state

    scored = []
    for cand in all_solutions:
        q = cand[:7]
        theta0 = float(cand[7]) if cand.shape[0] > 7 else 0.0
        dq = wrap_to_pi(q - q_prev)
        vel_cost = float(np.linalg.norm(dq))
        if q_prev2 is not None:
            ddq = wrap_to_pi(q - 2.0 * q_prev + q_prev2)
            acc_cost = float(np.linalg.norm(ddq))
        else:
            acc_cost = 0.0
        pose_cost = float(np.linalg.norm(pose_error(fk(q, p), T)))
        if theta0_prev is None:
            theta0_cost = 0.0
        else:
            theta0_cost = abs(float(wrap_to_pi(np.array([theta0 - theta0_prev]))[0]))
        score = (
            continuity.w_vel * vel_cost
            + continuity.w_acc * acc_cost
            + continuity.w_pose * pose_cost
            + continuity.w_theta0 * theta0_cost
        )
        scored.append((score, vel_cost, acc_cost, pose_cost, theta0_cost, cand))

    scored.sort(key=lambda x: x[0])
    best = scored[0]

    selected = best
    selected_by = "best_score"
    if q_lock is not None:
        lock_item = min(scored, key=lambda x: float(np.linalg.norm(wrap_to_pi(x[5][:7] - q_lock))))
        if lock_item[0] <= best[0] + continuity.hysteresis_margin:
            selected = lock_item
            selected_by = "hysteresis_locked"

    q_best_full = selected[5]
    q_best = q_best_full[:7]
    theta0_best = float(q_best_full[7]) if q_best_full.shape[0] > 7 else None
    
    # 新增：1D QP优化
    q_best = _optimize_q_with_1d_qp(q_best, T, p, continuity)
    
    pose_best = float(np.linalg.norm(pose_error(fk(q_best, p), T)))

    next_state = ContinuityRuntimeState(
        q_prev=q_best,
        q_prev2=q_prev.copy(),
        theta0_prev=theta0_best if theta0_best is not None else theta0_prev,
        q_lock=q_best,
    )
    report = {
        "method": f"{method}+1DQP",
        "candidate_count": len(all_solutions),
        "selected_by": selected_by,
        "score_best": float(selected[0]),
        "vel_cost_best": float(selected[1]),
        "acc_cost_best": float(selected[2]),
        "pose_cost_best": float(selected[3]),
        "theta0_cost_best": float(selected[4]),
        "pose_err_best": pose_best,
        "theta0_selected": theta0_best,
    }
    return q_best, report, next_state
