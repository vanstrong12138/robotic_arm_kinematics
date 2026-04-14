#!/usr/bin/env python3
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from ik_solver import (
    ContinuityParams,
    ContinuityRuntimeState,
    fk,
    solve_pose_continuous_with_state,
)


class IkJointStatePublisher(Node):
    """Bridge node: interactive marker pose -> IK -> joint_states."""

    def __init__(self) -> None:
        super().__init__("ik_joint_state_publisher")

        # 原有参数
        self.declare_parameter("target_pose_topic", "/ik_target_pose")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("fk_pose_topic", "/ik_fk_pose")
        self.declare_parameter("fk_state_topic", "/ik_fk_state")
        self.declare_parameter("ik_base_frame", "base_link")
        self.declare_parameter("default_target_frame", "base_link")
        self.declare_parameter("n_psi", 181)
        self.declare_parameter("republish_rate_hz", 20.0)
        self.declare_parameter("trajectory_window_size", 8)
        self.declare_parameter("local_theta0_window", 0.35)
        self.declare_parameter("local_theta0_count", 41)
        self.declare_parameter("w_vel", 1.0)
        self.declare_parameter("w_acc", 0.25)
        self.declare_parameter("w_pose", 0.1)
        self.declare_parameter("w_theta0", 0.15)
        self.declare_parameter("hysteresis_margin", 0.03)
        self.declare_parameter("enable_global_fallback", True)
        self.declare_parameter("target_to_ik_ee_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("target_to_ik_ee_rpy", [0.0, 0.0, 0.0])

        # 新增1D QP参数
        self.declare_parameter("w_qp_joint_inc", 1.0)
        self.declare_parameter("w_qp_pose_err", 0.5)

        # 读取原有参数
        self._target_pose_topic = str(self.get_parameter("target_pose_topic").value)
        self._joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self._fk_pose_topic = str(self.get_parameter("fk_pose_topic").value)
        self._fk_state_topic = str(self.get_parameter("fk_state_topic").value)
        self._ik_base_frame = str(self.get_parameter("ik_base_frame").value)
        self._default_target_frame = str(
            self.get_parameter("default_target_frame").value
        )
        self._n_psi = int(self.get_parameter("n_psi").value)
        self._republish_rate_hz = float(self.get_parameter("republish_rate_hz").value)
        self._trajectory_window_size = max(
            1, int(self.get_parameter("trajectory_window_size").value)
        )
        xyz = [float(v) for v in self.get_parameter("target_to_ik_ee_xyz").value]
        rpy = [float(v) for v in self.get_parameter("target_to_ik_ee_rpy").value]
        self._T_target_to_ik = self._xyz_rpy_to_transform(xyz, rpy)

        # 初始化连续性参数（包含QP权重）
        self._continuity = ContinuityParams()
        self._continuity.local_theta0_window = float(
            self.get_parameter("local_theta0_window").value
        )
        self._continuity.local_theta0_count = int(
            self.get_parameter("local_theta0_count").value
        )
        self._continuity.w_vel = float(self.get_parameter("w_vel").value)
        self._continuity.w_acc = float(self.get_parameter("w_acc").value)
        self._continuity.w_pose = float(self.get_parameter("w_pose").value)
        self._continuity.w_theta0 = float(self.get_parameter("w_theta0").value)
        self._continuity.hysteresis_margin = float(
            self.get_parameter("hysteresis_margin").value
        )
        self._continuity.enable_global_fallback = bool(
            self.get_parameter("enable_global_fallback").value
        )
        # 读取QP权重参数
        self._continuity.w_qp_joint_inc = float(
            self.get_parameter("w_qp_joint_inc").value
        )
        self._continuity.w_qp_pose_err = float(
            self.get_parameter("w_qp_pose_err").value
        )

        self._joint_names: List[str] = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        self._q_prev = np.zeros(7, dtype=float)
        self._last_q: Optional[np.ndarray] = None
        self._last_fk_pose: Optional[PoseStamped] = None
        self._last_fk_state: Optional[String] = None
        self._continuity_state = ContinuityRuntimeState(q_prev=self._q_prev.copy())

        self._joint_pub = self.create_publisher(JointState, self._joint_state_topic, 10)
        self._fk_pose_pub = self.create_publisher(PoseStamped, self._fk_pose_topic, 10)
        self._fk_state_pub = self.create_publisher(String, self._fk_state_topic, 10)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._target_sub = self.create_subscription(
            PoseStamped, self._target_pose_topic, self._on_target_pose, 10
        )

        timer_dt = 1.0 / max(self._republish_rate_hz, 1.0)
        self._timer = self.create_timer(timer_dt, self._republish_last_joint_state)

        # 完善日志输出
        self.get_logger().info(
            f"IK bridge : {self._target_pose_topic} -> IK(1D QP) -> {self._joint_state_topic}"
        )
        self.get_logger().info(f"FK pose publish topic: {self._fk_pose_topic}")
        self.get_logger().info(
            f"FK state publish topic: {self._fk_state_topic} "
            "(human readable text with position + Euler XYZ extrinsic rad/deg)"
        )
        self.get_logger().info(
            f"IK base frame: {self._ik_base_frame}, default target frame: {self._default_target_frame}"
        )
        self.get_logger().info(
            f"Using ik_solver.solve_pose_continuous_with_state (incremental+1DQP), trajectory_window_size={self._trajectory_window_size}"
        )
        self.get_logger().info(
            f"1D QP weights - joint increment: {self._continuity.w_qp_joint_inc}, pose error: {self._continuity.w_qp_pose_err}"
        )

    def _on_target_pose(self, msg: PoseStamped) -> None:
        source_frame = (
            msg.header.frame_id if msg.header.frame_id else self._default_target_frame
        )
        T_source_target = self._pose_to_transform(msg.pose)
        if T_source_target is None:
            self.get_logger().warning(
                "Ignore target pose: invalid quaternion norm too small"
            )
            return

        try:
            stamp = (
                Time.from_msg(msg.header.stamp)
                if (msg.header.stamp.sec or msg.header.stamp.nanosec)
                else Time()
            )
            tf_msg = self._tf_buffer.lookup_transform(
                self._ik_base_frame,
                source_frame,
                stamp,
                timeout=Duration(seconds=0.1),
            )
        except TransformException as exc:
            self.get_logger().warning(
                f"TF lookup failed ({source_frame} -> {self._ik_base_frame}): {exc}"
            )
            return

        T_ikbase_source = self._transform_stamped_to_matrix(tf_msg)
        T_ik = T_ikbase_source @ T_source_target @ self._T_target_to_ik
        q_best, report, self._continuity_state = solve_pose_continuous_with_state(
            T_ik,
            state=self._continuity_state,
            p=None,
            n_psi=self._n_psi,
            continuity=self._continuity,
        )
        if q_best is None:
            self.get_logger().warning(
                f"IK failed (method={report.get('method')}, candidate_count={report.get('candidate_count')})"
            )
            return

        q_best_np = np.array(q_best, dtype=float).reshape(7)
        self._q_prev = q_best_np
        self._last_q = q_best_np
        self._publish_joint_state(q_best, self._ik_base_frame)
        self._publish_fk_pose(q_best_np, self._ik_base_frame)

    def _publish_joint_state(self, q: np.ndarray, frame_id: str) -> None:
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = frame_id if frame_id else self._ik_base_frame
        out.name = self._joint_names
        out.position = [float(v) for v in q.tolist()]
        out.velocity = []
        out.effort = []
        self._joint_pub.publish(out)

    def _republish_last_joint_state(self) -> None:
        if self._last_q is None:
            return
        self._publish_joint_state(self._last_q, self._ik_base_frame)
        if self._last_fk_pose is not None:
            self._last_fk_pose.header.stamp = self.get_clock().now().to_msg()
            self._fk_pose_pub.publish(self._last_fk_pose)
        if self._last_fk_state is not None:
            self._fk_state_pub.publish(self._last_fk_state)

    def _publish_fk_pose(self, q: np.ndarray, frame_id: str) -> None:
        T = fk(np.array(q, dtype=float).reshape(7))
        p = T[:3, 3]
        R = T[:3, :3]
        quat = self._rot_to_quat(T[:3, :3])
        euler_xyz_ext = self._rot_to_euler_xyz_extrinsic(R)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id if frame_id else self._ik_base_frame
        msg.pose.position.x = float(p[0])
        msg.pose.position.y = float(p[1])
        msg.pose.position.z = float(p[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self._fk_pose_pub.publish(msg)
        self._last_fk_pose = msg

        euler_xyz_ext_deg = np.rad2deg(euler_xyz_ext)
        state_msg = String()
        state_msg.data = (
            f"frame={msg.header.frame_id}; "
            f"pos[m]=({float(p[0]):.6f}, {float(p[1]):.6f}, {float(p[2]):.6f}); "
            f"euler_xyz_ext[rad]=({float(euler_xyz_ext[0]):.6f}, {float(euler_xyz_ext[1]):.6f}, {float(euler_xyz_ext[2]):.6f}); "
            f"euler_xyz_ext[deg]=({float(euler_xyz_ext_deg[0]):.3f}, {float(euler_xyz_ext_deg[1]):.3f}, {float(euler_xyz_ext_deg[2]):.3f})"
        )
        self._fk_state_pub.publish(state_msg)
        self._last_fk_state = state_msg

    @staticmethod
    def _quat_to_rot(
        qx: float, qy: float, qz: float, qw: float
    ) -> Optional[np.ndarray]:
        n = qx * qx + qy * qy + qz * qz + qw * qw
        if n < 1e-12:
            return None
        s = 2.0 / n
        xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
        xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
        wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
        R = np.eye(3, dtype=float)
        R[0, 0] = 1.0 - (yy + zz)
        R[0, 1] = xy - wz
        R[0, 2] = xz + wy
        R[1, 0] = xy + wz
        R[1, 1] = 1.0 - (xx + zz)
        R[1, 2] = yz - wx
        R[2, 0] = xz - wy
        R[2, 1] = yz + wx
        R[2, 2] = 1.0 - (xx + yy)
        return R

    @staticmethod
    def _rot_to_quat(R: np.ndarray) -> np.ndarray:
        tr = float(R[0, 0] + R[1, 1] + R[2, 2])
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        q = np.array([qx, qy, qz, qw], dtype=float)
        n = np.linalg.norm(q)
        if n < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return q / n

    @staticmethod
    def _rot_to_euler_xyz_extrinsic(R: np.ndarray) -> np.ndarray:
        # Extrinsic XYZ (fixed axes X->Y->Z), equivalent matrix form:
        # R = Rz(gamma) @ Ry(beta) @ Rx(alpha)
        # Return [alpha_x, beta_y, gamma_z].
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-8
        if not singular:
            alpha_x = np.arctan2(R[2, 1], R[2, 2])
            beta_y = np.arctan2(-R[2, 0], sy)
            gamma_z = np.arctan2(R[1, 0], R[0, 0])
        else:
            alpha_x = np.arctan2(-R[1, 2], R[1, 1])
            beta_y = np.arctan2(-R[2, 0], sy)
            gamma_z = 0.0
        return np.array([alpha_x, beta_y, gamma_z], dtype=float)

    def _pose_to_transform(self, pose) -> Optional[np.ndarray]:
        R = self._quat_to_rot(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        if R is None:
            return None
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[0, 3] = float(pose.position.x)
        T[1, 3] = float(pose.position.y)
        T[2, 3] = float(pose.position.z)
        return T

    def _transform_stamped_to_matrix(self, tf_msg) -> np.ndarray:
        R = self._quat_to_rot(
            tf_msg.transform.rotation.x,
            tf_msg.transform.rotation.y,
            tf_msg.transform.rotation.z,
            tf_msg.transform.rotation.w,
        )
        if R is None:
            raise TransformException("Invalid TF quaternion")
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[0, 3] = float(tf_msg.transform.translation.x)
        T[1, 3] = float(tf_msg.transform.translation.y)
        T[2, 3] = float(tf_msg.transform.translation.z)
        return T

    @staticmethod
    def _xyz_rpy_to_transform(xyz: List[float], rpy: List[float]) -> np.ndarray:
        x, y, z = xyz
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        T = np.eye(4, dtype=float)
        T[:3, :3] = Rz @ Ry @ Rx
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z
        return T


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IkJointStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
