# 目标跟随导航、里程计与丢失目标处理

## 1. Nav2 的成熟跟随接口如何表示目标、停车和失目标？

### Takeaway

Nav2 将“目标位姿”作为输入，持续结合机器人新位姿计算控制；它不直接把检测框当作导航目标。可借鉴这个结构，但不能把其失目标等待时间、转向控制器或搜索动作直接作为 MicroDuck 已验证方案。[官方动态目标跟随教程](https://docs.nav2.org/rolling/tutorials/general_tutorials/navigation2_dynamic_point_following/navigation2_dynamic_point_following/)

### Cited Findings

- **核实日期：2026-10-10。版本边界：** Following Server 被列入 **Kilted → Lyrical** 迁移指南，PR #5565 新增 `opennav_following`。本笔记核查 Nav2 `main` 的 `3286119753ce1fab9d2ff411ea6d439b823fe18d`（2026-10-09），其中包版本为 **1.5.0**；不能据此假定旧版 Humble/Jazzy 已有同一接口。[固定版本迁移指南](https://github.com/ros-navigation/docs.nav2.org/blob/76849a4062be342499889f4442838d6ecca77046/docs/configuration_and_development/migration_guides/kilted/Kilted.md#following-server--kilted-to-lyrical-following-server-)，[package.xml](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/package.xml)
- **真实 action：** `follow_object` 的 `FollowObject` goal 仅包含 `pose_topic`、`tracked_frame`、`max_duration`。目标可来自带坐标系和时间戳的 `PoseStamped` 话题或 TF frame；检测、目标定位需由外部模块提供。`desired_distance`、`detection_timeout` 是服务器参数，不能照教程宽泛措辞写成 action goal 字段。[FollowObject.action](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_msgs/action/FollowObject.action)，[订阅与 action 初始化源码](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/src/following_server.cpp#L47-L72)，[动态目标教程](https://docs.nav2.org/rolling/tutorials/general_tutorials/navigation2_dynamic_point_following/navigation2_dynamic_point_following/)
- **目标与自身运动分开：** topic 模式把检测位姿按其时间戳变换至固定坐标系（默认 `odom`）；每个控制周期取得新的机器人位姿，再计算保留指定距离的目标点。这里的距离是机器人 base 原点到目标位置的几何距离，源码没有“前脚到球面间隙”的几何定义。[getRefinedPose / approachObject / getPoseAtDistance](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/src/following_server.cpp#L345-L399)
- **首次漏检不等于立即停车：** topic 模式缓存末次检测位姿，只要检测时间戳未超过 `detection_timeout` 就可继续使用；目标超时后抛出检测失败，进入有上限的重试，最终发布零速度并结束。恢复性异常分支没有保证每次失败先发零速度。因此不能把它描述为“任何漏检都会立即停车”。[检测新鲜度与失败处理源码](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/src/following_server.cpp#L289-L342)，[缓存与超时检查](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/src/following_server.cpp#L496-L558)
- **有界搜索且默认关闭：** 当前默认 `detection_timeout=2.0s`、`max_retries=3`、`search_by_rotating=false`、`rotate_to_object_timeout=10.0s`。启用搜索时围绕初始朝向分别尝试正、负 `search_angle`，边旋转边检查重新检测；到指定距离先清零等待目标移动。这些是软件默认值，不是双足机器人安全证据。[参数源码](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/src/parameter_handler.cpp#L35-L70)，[搜索源码](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/src/following_server.cpp#L402-L476)
- **能力范围：** 该版本 Following Server 初始化时强制关闭自身碰撞检测，并明确提示不支持。跟随接口存在，不代表已经拥有完整避障和运动安全保障。[初始化源码](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/src/following_server.cpp#L61-L72)

### Inferences

- **适用于本项目的结构推断：** 借鉴目标位姿、观测时效、持续机器人位姿、有界恢复四个接口即可；不需要为了一个球迁移整套 ROS2/Nav2。目标为静止球时，短时保留的是“局部坐标中的球位置估计”，同时更新自身运动，而不是保留上次前进速度。[结构依据](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/src/following_server.cpp#L345-L399)
- **本项目保守策略推断：** 漏检后可立即请求零速度，同时短时继续估计、等待观测恢复；超过时效、位姿失效或身份不确定则停止。主动搜索应作为独立状态，仅在站稳、停止前进且转向响应已验证后启用。等待多久由观测间隔、延迟、里程计误差与刹停距离决定，不能直接复制 Nav2 的 2 秒。[边界设计依据](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/src/parameter_handler.cpp#L35-L70)

### Gaps

- 官方跟随教程没有替本项目解决单目球的米制距离、目标身份重识别或双足步态稳定性，也没有给出适用于 MicroDuck 的超时与搜索角速度。因此本笔记不提供新的控制参数，更不把官方功能当作板端验证。[教程明确将目标检测定位留在范围外](https://docs.nav2.org/rolling/tutorials/general_tutorials/navigation2_dynamic_point_following/navigation2_dynamic_point_following/)

## 2. 一次定位后开环前进，与里程计预测加视觉校正有什么区别？

### Takeaway

不必在“每帧直接控制”与“看到一次后盲走”之间二选一：机器人位姿连续更新，目标可见时校正目标估计，暂不可见时预测并增加不确定性。局部演示可以使用 `odom`；增加地图并不是这个结构成立的前提。[REP-105](https://github.com/ros-infrastructure/rep/blob/master/rep-0105.rst)，[robot_localization 状态估计文档](https://github.com/cra-ros-pkg/robot_localization/blob/7dfb6aa97b2082185d2fac3420888ae8474bfc1a/doc/state_estimation_nodes.rst)

### Cited Findings

| 坐标系 | 官方含义及控制上的作用 |
|---|---|
| `base_link` | 固连于机器人机身；用于表示机器人当前相对目标方向。[REP-105](https://github.com/ros-infrastructure/rep/blob/master/rep-0105.rst#base_link) |
| `odom` | 世界固定的局部参考；机器人位姿应连续，但误差可长期无界累积，适合短程局部控制。[REP-105](https://github.com/ros-infrastructure/rep/blob/master/rep-0105.rst#odom) |
| `map` | 经全局观测修正的世界参考；可发生跳变，适合长期全局定位。[REP-105](https://github.com/ros-infrastructure/rep/blob/master/rep-0105.rst#map) |

- **可以没有 map：** `robot_localization` 明确说明，没有 `map_frame` 时可移除它，并将 `world_frame` 设为 `odom_frame`。仅融合连续的里程计、视觉里程计或 IMU 时，默认也使用 `odom`。[frame 参数文档](https://github.com/cra-ros-pkg/robot_localization/blob/7dfb6aa97b2082185d2fac3420888ae8474bfc1a/doc/state_estimation_nodes.rst#frame)
- **预测与观测修正是两步：** 当前 `robot_localization` rolling-devel `7dfb6aa97b2082185d2fac3420888ae8474bfc1a`，包版本 **3.10.1**。其 EKF 以运动模型向前预测，再用传感器观测修正；传感器超时时可以只预测。`smooth_lagged_data` 支持回到延迟观测发生前的历史状态再重放，前提是有正确时间戳和足够历史。[版本](https://github.com/cra-ros-pkg/robot_localization/blob/7dfb6aa97b2082185d2fac3420888ae8474bfc1a/package.xml)，[估计器及 sensor_timeout / smooth_lagged_data](https://github.com/cra-ros-pkg/robot_localization/blob/7dfb6aa97b2082185d2fac3420888ae8474bfc1a/doc/state_estimation_nodes.rst)
- **里程计记忆需要时效：** REP-105 明确要求按里程计精度丢弃过旧或过远的数据；高精度冗余编码器与转向反馈薄弱的平台不能采用同一数据保留时间。[REP-105 关于 odom 数据保留](https://github.com/ros-infrastructure/rep/blob/master/rep-0105.rst#transitions-between-maps)

### Inferences

- **建议的最小坐标链，属于设计推断：** 观测发生于 `t_m` 时，先用同一时刻的机身位姿与头部外参，把球位置变到局部坐标：`p_ball^odom = T_odom_base(t_m) · T_base_camera(t_m) · p_ball^camera(t_m)`。到控制时刻 `t_now` 再用 `T_base_odom(t_now)` 得到当前相对球位置。这个链条补偿机器人平移和头部运动；仅补偿图像旋转没有维护完整米制目标状态。[官方采用观测时刻目标变换、控制时刻机器人位姿的依据](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/src/following_server.cpp#L496-L558)
- **应区分两个估计器，属于设计推断：** 机器人估计器维护自身位置、朝向、速度；球估计器维护球位置、可选球速度、最后观测时刻与不确定性。普通未知球的相机相对位置不能直接冒充“机器人绝对位置”塞进自身 EKF。只有已知位置的地标和正确测量模型，才能反过来约束机器人定位。[机器人状态变量定义](https://github.com/cra-ros-pkg/robot_localization/blob/7dfb6aa97b2082185d2fac3420888ae8474bfc1a/doc/state_estimation_nodes.rst#sensor-config)，[目标位姿接口](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_msgs/action/FollowObject.action)
- **开环与预测不可混淆，属于工程推断：** 将前进指令乘以时间推算移动距离，是控制模型预测；有独立运动观测并反复修正的里程计，才提供反馈。允许估计器在短暂漏检期间继续计算，也不意味着允许机器人持续前进。滤波只能在模型与观测假设内组合信息，不能保证修复错误标定、偏置、实际转向响应或跌倒。[EKF 模型与控制输入的官方界限](https://github.com/cra-ros-pkg/robot_localization/blob/7dfb6aa97b2082185d2fac3420888ae8474bfc1a/doc/state_estimation_nodes.rst#use-control)，[odom 漂移界限](https://github.com/ros-infrastructure/rep/blob/master/rep-0105.rst#odom)
- **本项目上下文约束，尚非新验证：** 当前已有 `odom.position` 的运动学估计，可以作为待评估的输入；应保留其漂移和失效标志，不能当作仿真真值。优先明确坐标、时间戳、观测质量和测量误差，再选择是否使用 Kalman。这个建议由上述状态估计契约推导，本轮没有重新测量现有里程计。[设计依据](https://github.com/cra-ros-pkg/robot_localization/blob/7dfb6aa97b2082185d2fac3420888ae8474bfc1a/doc/state_estimation_nodes.rst)

### Gaps

- 当前相机协议没有曝光源时间戳的项目限制，不能通过“加 EKF”自动消除；本轮没有确定可接受的时钟误差、里程计协方差或目标距离误差。以上坐标链要求这些接口先被定义。官方延迟修正依赖观测时间与历史状态，不构成现有接收时间等同曝光时间的证据。[历史数据处理前提](https://github.com/cra-ros-pkg/robot_localization/blob/7dfb6aa97b2082185d2fac3420888ae8474bfc1a/doc/state_estimation_nodes.rst#smooth-lagged-data)

## 3. 循迹、目标跟随、寻球是否是同一个模块？

### Takeaway

路径跟踪的输入是路径，目标跟随的输入是变化的目标状态，寻球负责决定去哪里看与何时放弃。它们可以串联，但彼此不能替代。[Nav2 官方两种动态跟随方法](https://docs.nav2.org/rolling/tutorials/general_tutorials/navigation2_dynamic_point_following/navigation2_dynamic_point_following/)

### Cited Findings

- 官方教程给出两条路线：专用 Following Server；或在行为树中由 `GoalUpdater` 更新目标、`ComputePathToPose` 规划、`TruncatePath` 保留跟随间隔、`FollowPath` 跟踪路径。第二条路线的示例按 1Hz 重新规划，目标更新与路径跟踪是不同节点。[固定版本教程源码](https://github.com/ros-navigation/docs.nav2.org/blob/76849a4062be342499889f4442838d6ecca77046/docs/tutorials/general_tutorials/navigation2_dynamic_point_following/navigation2_dynamic_point_following.md)
- 专用服务器在到达期望距离后发布零速度，并继续等待目标移动；失目标搜索是可选择、受角度和时间限制的恢复过程。[到达处理](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/src/following_server.cpp#L249-L277)，[旋转搜索](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/src/following_server.cpp#L402-L476)

### Inferences

- **最小原型建议：** `视觉检测/距离观测 → 球状态估计`，与 `IMU/腿部运动观测 → 自身局部里程计` 并行；两者送入 `接近与搜索状态机 → 速度指令 → 现有步态控制`。空旷短程寻球先不需要路径规划、地图或 SLAM；若以后有障碍和指定路线，再加入规划与循迹模块。[结构依据](https://docs.nav2.org/rolling/tutorials/general_tutorials/navigation2_dynamic_point_following/navigation2_dynamic_point_following/)，[无 map 的 odom 模式](https://github.com/cra-ros-pkg/robot_localization/blob/7dfb6aa97b2082185d2fac3420888ae8474bfc1a/doc/state_estimation_nodes.rst#frame)
- **失目标分层建议：** 首次未建立目标则保持停止并寻球；已建立目标的短时漏检可零速度等待，同时预测位置和扩大不确定性；恢复观测必须确认身份一致；超时或不确定性过大则退出跟随，只有受控搜索可另行恢复。头部搜索与身体旋转应各自有停止条件，不能把“继续预测”当作无限前进的理由。[有界恢复与观测时效依据](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/src/following_server.cpp#L289-L342)，[数据保留时效依据](https://github.com/ros-infrastructure/rep/blob/master/rep-0105.rst#transitions-between-maps)
- **停车尺度建议：** 若要求“前脚到球面约 20cm”，应将目标中心距离、球半径、前脚几何和刹停运动联系起来，单独定义到达条件。目标状态估计、到达判断和步态稳定性分开验证；导航库的 base-to-object 距离不会自动等于该足端间隙。[Nav2 距离构造依据](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/src/following_server.cpp#L616-L629)

### Gaps

- 本研究只给出结构建议，没有证明现有步态能稳定执行所需角速度，也没有证明现有单目球定位与里程计足以维持 20cm 间隙。完成研究不等于允许扩大运动范围或宣称实物成功；需要分别取得自身运动、视觉米制位置、停车与步态响应的实验依据。[官方接口与实现的支持范围](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/src/following_server.cpp)
