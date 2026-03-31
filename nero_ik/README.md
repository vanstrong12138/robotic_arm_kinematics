# NERO Arm-Angle IK (nero_ik)

这个目录提供一个可运行的 7DoF NERO 机械臂臂角参数化 IK 实现：

- 使用你提供的 NERO DH 参数与关节限位
- 以臂角 `psi` 参数化肘点轨迹
- 扫描 `psi` 求可行解并做限位过滤

## 文件说明

- `ik_solver.py`：主求解器
- `ik_joint_state_publisher.py`：ROS2桥接
- `interactive_target_marker`：可交互式Marker

## 快速运行

```bash
cd nero_ik
python ik_joint_state_publisher.py
python interactive_target_marker.py
```

## 说明

- 该版本采用“臂角参数化 + 数值闭环求解”的工程实现方式，便于直接落地验证。

