# 让 MicroDuck 记住球，并持续校正方向

**现有方案适合作为有限条件下的视觉接近原型，但不足以承担完整寻球导航。** 球框中心可提供图像方向误差，面积可提供粗略远近线索；球持续可见时，这种反馈有合理依据。问题在于，把它扩大成“可靠判断脚前距离、经过遮挡仍知道球在哪里”的系统，缺少米制几何、状态记忆和恢复行为。你的“先确定位置、方向与距离”思路值得保留，但一次定位后停止视觉修正，会让步态误差、里程计漂移和球移动持续累积。建议下一步采用**局部 odom 中的球状态＋可见时视觉校正＋短时预测＋受控重搜**，先不引入完整 ROS2/Nav2/SLAM。以下是基于公开源码与现有记录的架构评估，未修改程序、运行试验或证明问题已解决。

## 面积接近有适用范围，丢球停车暴露了导航缺层

视觉伺服文献区分以图像特征形成误差的 IBVS，以及以视觉估计三维参数形成误差的 PBVS；因此，“看着球调整动作”本身不是错误方向。当前中心和面积反馈属于图像空间控制思路，但还不能等同于完整 IBVS 控制器。特征离开视野会破坏反馈，单纯调增漏检容忍时间也不会生成球位置。([视觉伺服原始论文](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf)、[作者对视野约束的说明](https://faculty.cc.gatech.edu/~seth/res.php?u=vs))

| 方案 | 合理用途 | MicroDuck 的关键限制 |
|---|---|---|
| 图像中心／面积持续反馈 | 球持续可见时的简单接近 | 面积不是足前间隙；失去检测后缺少位置记忆 |
| 一次定位后按旧目标行走 | 静止目标、可靠自身位姿和短程受控环境 | 球移动、脚滑、航向漂移无法靠旧观测纠正 |
| 米制球状态＋预测＋重新观测 | 短暂漏检后恢复接近与搜索 | 需要标定、时间对齐、里程计误差和有效性边界 |

表中适用性是本项目的工程判断，依据上述视觉反馈分类，以及 odom 连续但长期漂移的定义；第三种最符合当前目标，尚未实施。([REP-105：odom](https://github.com/ros-infrastructure/rep/blob/master/rep-0105.rst#odom))

最新 `035720` 记录是球左侧裁切／丢失后受控停车，记录为 `stopped_missing_ball`，且停车确认、保持站立；不能改述成跌倒，也不能认定已完成导航。当前链路是 RK3588 NPU 视觉、CPU 策略，Mac MuJoCo 提供身体与相机，**这不是实体机器人验证**。转向实际响应与配时仍须单独核实，增加状态估计不会自动修复步态转向。([本项目最新记录](</Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/ball_follow_npu_20261010_035720/summary.json>))

## 成熟系统记住坐标与可信度，而不是最后一次方向指令

B-Human 保存球位置、速度、协方差和最后观测时间；机器人运动后，用里程计把过去状态变换到当前身体坐标，再以新视觉更新。Bit-Bots 当前公开实现把三维球观测按时间戳变换到 odom，保存位置与协方差；漏检时增加不确定性，但这份实现没有球速度状态。可借鉴的是“估计仍在、可信度下降、行为另行决策”，不应把所有实现笼统称为完整运动 Kalman 跟踪。([B-Human 球模型](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Representations/Modeling/BallModel.h)、[球状态传播](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/Modeling/BallStateEstimator/BallStateEstimator.cpp)、[Bit-Bots 球滤波器](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_world_model/bitbots_ball_filter/bitbots_ball_filter/ball_filter.py))

对 MicroDuck，球状态至少需要位置、参考系、观测时间、不确定性和“观测／预测／丢失”标志。机器人状态则维护自身位置与朝向。**视觉球观测更新球估计，不能把可移动、未知世界位置的球当静态地标纠正机器人 odom。** 当前运动学 odom 也是待评估的估计，并非真值；把速度指令乘时间只是模型预测，不能替代实际运动反馈。局部 odom 足以承载短程目标，地图并非前提。([状态估计的坐标系与控制输入](https://github.com/cra-ros-pkg/robot_localization/blob/7dfb6aa97b2082185d2fac3420888ae8474bfc1a/doc/state_estimation_nodes.rst))

Nav2 的可借鉴点同样是带时间与坐标系的目标位姿、持续更新的机器人位姿，以及有界恢复。它没有替我们完成单目测距；本次核查的 Following Server 版本还强制关闭自身碰撞检测，不能直接当作完整避障保障。RoboCup 的搜索时长、Nav2 的超时和角速度都不能照搬到双足平台。([Nav2 目标变换与控制](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/src/following_server.cpp#L345-L399)、[碰撞检测限制](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/src/following_server.cpp#L61-L72))

## 70 毫米球能提供尺度，但标定与时序决定位置是否可信

理想固定成像下，面积约随距离平方的倒数变化，所以它能表达粗略尺度；裁切、遮挡、框偏差和相机姿态会破坏固定面积对应固定足前距离的假设。已知球直径 `D=0.070 m`，远处、近光轴且轮廓完整时，可由 `Z≈fD/d` 估计光轴深度；这仍不是脚到球面的距离，YOLO 框也未必贴合真实直径。应先标定内参、畸变，并在图像缩放后同步调整内参。([OpenCV 相机模型](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html))

另一条约束是球落地：地面为 `z=0` 时，球心位于 `z=0.035 m`，将去畸变后的球心射线与该高度平面求交。它需要相机离地高度及朝向；IMU 提供姿态信息，不能单独提供高度，头部关节与身体几何的正运动学还要给出相机外参。射线接近地平线、交点落在相机后方或球腾空时，应拒绝落地观测。直径法与落地法明显冲突时，应降权或判无效，先查几何误差。([B-Human 平面投影](https://raw.githubusercontent.com/bhuman/BHumanCodeRelease/master/Src/Tools/Math/Transformation.cpp)、[相机姿态敏感性](https://docs.b-human.de/master/perception/perception-infrastructure/))

这些变换必须使用**图像采集时刻**的头部与身体姿态，再用控制时刻的机器人位姿求当前相对球位置。现有 TCP 图像没有源采集时间戳，完整帧到达时间只能近似拍摄时间；头部摆动时，这个偏差会变成方向和距离误差。补帧序号、源时间及跨机时钟映射，应先于宣称滤波已解决延迟。([现有协议](</Users/ethan/Documents/RK3588 Microduck/work/probe_runtime_visual_load.py>)、[拍摄时刻姿态匹配](https://docs.b-human.de/master/perception/perception-infrastructure/))

## 先验证观测与自身运动，再决定预测能支撑什么动作

建议新增三层：米制球观测负责几何与质量，球状态负责记忆、预测和视觉更新，接近／搜索监督器负责对准、减速、停车、重获与完成。自身 odom 并行输入后两层；现有步态接受速度目标。2D tracker 主要解决目标关联，不能代替米制状态。若“循迹”指沿地面线走，应依据线的位置与方向；若指跟踪预定路径，应依据路径和自身位姿，两者都不能仅靠最后球位置。([Nav2 对目标更新、规划与路径跟踪的分层](https://github.com/ros-navigation/docs.nav2.org/blob/76849a4062be342499889f4442838d6ecca77046/docs/tutorials/general_tutorials/navigation2_dynamic_point_following/navigation2_dynamic_point_following.md))

验证先看离线静态几何：不同距离、头部姿态与边缘裁切下，比较两种测距和真实参考位置；仿真真值只作评分。再看静止遮挡与球移动，检查预测标志、过期失效和错误重获。然后独立核实直行、左右转、停走切换的实际位移、航向与 odom 误差，最后才评估运动中遮挡和足前停车间隙。这些是后续验收层次，本次未执行，也不要求重复旧的闭环测试。

现阶段可以保留丢球零速保护，同时继续估计并等待恢复；预测有效不等于获准继续前进。超时、位姿失效或身份不确定时退出接近；站稳后才进入受控扫头／转身搜索。有效期和动作范围应由观测延迟、漂移、转向响应及刹停证据决定，不预设“可盲走几秒”。([有界恢复依据](https://github.com/ros-navigation/navigation2/blob/3286119753ce1fab9d2ff411ea6d439b823fe18d/nav2_following/opennav_following/src/following_server.cpp#L289-L342))

## 结论

下一步的决策点应是：当前视觉几何和自身运动估计，能否支撑有界的局部目标状态。若证据不足，状态记忆仍可用于指导停止后的重搜；若证据成立，再讨论短时遮挡期间允许哪些动作。这样，每个新增能力都有独立验收条件，也避免把“换了架构”误当成“转向、测距与导航已经可靠”。
