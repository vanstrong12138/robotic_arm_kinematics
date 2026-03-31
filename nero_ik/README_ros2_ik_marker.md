# ROS2 交互式 Marker + IK + JointState 发布

本目录新增了两个脚本，串起完整流程：

- `interactive_target_marker.py`：在 RViz 里拖动 6DoF marker，发布目标位姿到 `/ik_target_pose`
- `ik_joint_state_publisher.py`：订阅 `/ik_target_pose`，调用 `super_ik.Solver` 解析 IK，发布 `/joint_states`
  - 当前默认使用 `solve_trajectory_continuous` 连续方法（抑制分支跳变）

## 1) 启动交互式 Marker

```bash
python3 /home/agilex/ik_tool/new_nero_ik/interactive_target_marker.py
```

可选参数：

```bash
python3 /home/agilex/ik_tool/new_nero_ik/interactive_target_marker.py --ros-args -p frame_id:=base_link -p publish_topic:=/ik_target_pose
```

## 2) 启动 IK -> JointState 桥接节点

```bash
python3 /home/agilex/ik_tool/new_nero_ik/ik_joint_state_publisher.py
```

可选参数：

```bash
python3 /home/agilex/ik_tool/new_nero_ik/ik_joint_state_publisher.py --ros-args -p target_pose_topic:=/ik_target_pose -p joint_state_topic:=/joint_states -p ik_base_frame:=base_link -p default_target_frame:=base_link -p n_psi:=181
```

连续求解相关参数（可选）：

```bash
python3 /home/agilex/ik_tool/new_nero_ik/ik_joint_state_publisher.py --ros-args -p trajectory_window_size:=8 -p local_theta0_window:=0.35 -p local_theta0_count:=41 -p w_vel:=1.0 -p w_acc:=0.25 -p w_pose:=0.1 -p w_theta0:=0.15 -p hysteresis_margin:=0.03
```

脚本现在严格走 ROS TF 变换链路：  
`T(ik_base->target_for_ik) = T(ik_base->target_msg_frame) * T(target_msg_frame->target_pose) * T(target_pose->ik_ee)`

其中最后一项 `T(target_pose->ik_ee)` 用参数配置，默认是单位变换：

```bash
python3 /home/agilex/ik_tool/new_nero_ik/ik_joint_state_publisher.py --ros-args -p target_to_ik_ee_xyz:="[0.0, 0.0, 0.0]" -p target_to_ik_ee_rpy:="[0.0, 0.0, 0.0]"
```

## 3) RViz 配置

- `Fixed Frame` 设为 `base_link`（或与参数 `ik_base_frame` 保持一致）
- 添加显示项 `InteractiveMarkers`（server 名：`ik_target_marker`）
- 添加显示项 `RobotModel`（读取 `/joint_states`）

## 4) 快速验证

```bash
ros2 topic echo /ik_target_pose
ros2 topic echo /joint_states
```

拖动 marker 后，`/joint_states` 的 7 维关节角应更新。
