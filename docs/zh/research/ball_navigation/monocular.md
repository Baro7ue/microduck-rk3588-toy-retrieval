# MicroDuck：单目球定位、视觉伺服与定位后导航

研究日期：2026-10-10。仅检索论文、官方文档和项目代码，未运行模拟、板端实验或修改产品。旧 `gait_response` 诊断脚本尚未启动。

## 1. 已知70毫米球，能否把“面积停车”升级为米制位置？

### Takeaway

可以建立有明确假设的米制球观测，但不能把 YOLO 框面积直接称作距离。最小路线是标定相机、恢复球方向，再利用已知直径和球落地约束估计位置，输出有效性与不确定性。[OpenCV 官方相机模型](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)；[B-Human 平面投影实现](https://raw.githubusercontent.com/bhuman/BHumanCodeRelease/master/Src/Tools/Math/Transformation.cpp)

### Cited Findings

- OpenCV 的针孔模型用内参、外参和畸变描述三维点到像素的映射；`undistortPoints` 可恢复去畸变的归一化像素坐标。图像缩放时内参也要相应缩放。[OpenCV 官方文档](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)
- B-Human 的 `imageToRobotHorizontalPlane` 将相机射线旋转到机器人参考系，并结合相机平移求与指定高度平面的交点；实现会检查射线的垂直分量。它并非仅按图像面积判距离。[B-Human 原始代码，61–84行](https://raw.githubusercontent.com/bhuman/BHumanCodeRelease/master/Src/Tools/Math/Transformation.cpp)
- B-Human 用躯干变换、机器人尺寸和当前关节角构造相机变换；官方文档提醒，小幅相机朝向误差会显著影响远处物体的地面投影。[B-Human 官方感知文档](https://docs.b-human.de/master/perception/perception-infrastructure/)

### Inferences

- **已知直径路线**：由针孔模型可推得，远距离、近光轴且轮廓完整时，`Z≈f·D/d`，本项目 `D=0.070 m`。这是光轴深度近似，不是相机到球的斜距，也不是脚到球表面的距离；球靠边或靠近时应使用球体轮廓模型。YOLO 框只能先作带误差的直径代理，不能假定框恰好贴住球。[推导依据：OpenCV 模型](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)
- **落地路线**：设地面为 `z=0`、球心高度 `R=0.035 m`，相机中心为 `c`、估计球心方向为 `r`，则 `p=c+λr`、`λ=(R-c_z)/r_z`。这需要相机位置和地面关系，IMU 姿态单独不提供相机离地高度。射线近乎平行地面、交点在相机后方、球腾空时应拒绝该观测。[推导依据：B-Human 平面交点代码](https://raw.githubusercontent.com/bhuman/BHumanCodeRelease/master/Src/Tools/Math/Transformation.cpp)
- **两条路线先互检**：直径深度与落地深度明显冲突时标记无效或降权，先不要无条件平均。固定面积阈值在理想固定成像条件下可代表大致尺度，但头部运动、截断和框偏差会破坏“同面积=同脚前距离”的假设。[依据：OpenCV 投影与 B-Human 相机几何](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)；[相机几何](https://docs.b-human.de/master/perception/perception-infrastructure/)

### Gaps

- 现有检测框对真实球轮廓的偏差、实物相机内外参、运动时相机离地高度误差尚未量化，因此不能承诺厘米精度。理论几何可计算位置，不等于本项目已验证米制测距。[标定需求依据](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)

## 2. 持续视觉伺服，还是看见一次后按位置导航？

### Takeaway

建议保留连续视觉修正，并增加短时球状态记忆；不建议看见一次后把目标永久冻结。视觉观测主要更新球的状态，不能把可能移动的球当固定地标去修正机器人的 odom。[视觉伺服原始教程](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf)

### Cited Findings

- Chaumette、Hutchinson 的教程区分：IBVS 用图像特征形成误差；PBVS 用视觉估计的三维参数形成误差。教程的基本推导主要面向相机速度控制，不能直接作为双足步态的稳定性证明。[原始论文，Part I](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf)
- 作者对混合视觉伺服的介绍指出，图像与位姿控制分别存在另一空间中的约束问题，特征离开视野可能造成失败；混合方案需明确切换条件。[Hutchinson 作者研究页](https://faculty.cc.gatech.edu/~seth/res.php?u=vs)
- B-Human 的投影基础设施使用关节角与图像拍摄时刻的匹配，而不仅是收到图像时的最新关节角。[B-Human 官方文档，BodyContour](https://docs.b-human.de/master/perception/perception-infrastructure/)

### Inferences

| 方案 | 适用条件 | 本项目的边界 |
|---|---|---|
| 仅图像中心/面积反馈 | 球持续可见，任务只要求接近某个视觉尺度 | 反馈依赖每帧检测；丢球时没有位置记忆，面积本身不表示脚前间隙 |
| 一次定位后导航 | 目标静止、机器人位移估计可靠、路径和终点可验证 | 球移动、脚滑或 yaw 漂移会使旧目标过时；一次观测不能消除后续误差 |
| 米制观测＋短时预测＋再次观测 | 允许短暂漏检，且预测不确定性可监控 | 更适合作为下一步；长时间不可见时应停车重获目标，不能把预测当新检测 |

上表是针对本项目的架构判断，依据上述视觉伺服分类与视野限制，不是已完成控制器的实验结论。[论文](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf)；[作者说明](https://faculty.cc.gatech.edu/~seth/res.php?u=vs)

- 使用 odom 把过去球观测变换到当前身体坐标，并在 **odom 坐标中纠正球估计**；不反向宣称球观测修正了机器人 odom。除非明确球静止且建立联合定位模型，否则两者不可混同。[坐标变换依据](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)
- 本项目 TCP 没有源拍摄时间戳，因此现有“按完整帧到达时刻配对姿态”只能近似拍摄时刻；头部会摆动，时间偏差也会成为方向和距离误差。应优先补帧序号/采集时刻及跨机时钟映射，不能把网络到达时间改名为拍摄时间。[项目源协议与限制](</Users/ethan/Documents/RK3588 Microduck/work/probe_runtime_visual_load.py>)；[图像/姿态匹配的项目参考](https://docs.b-human.de/master/perception/perception-infrastructure/)

### Gaps

- 尚无运动、转向和遮挡条件下的球位置误差、odom 漂移及时间偏差分布，不能规定“可盲行多少秒”或“必能停车在某个厘米范围”。短时预测边界应从数据确定。[误差来源依据](https://docs.b-human.de/master/perception/perception-infrastructure/)

## 3. 本项目最少新增什么，怎样分层验证？

### Takeaway

先新增球的米制观测与状态估计，再决定接近控制；现有 RK3588 NPU 检测、CPU 步态和 Mac MuJoCo 身体链路可保留。最新记录是丢球后受控早停，不能把它改述成跌倒，也不能靠位置记忆掩盖尚未验证的步态转向响应。[本项目最新记录](</Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/ball_follow_npu_20261010_035720/summary.json>)

### Cited Findings

- 当前记录为 `stopped_missing_ball`，`stop_confirmed=true`、`left_enabled_standing=true`；入口说明也明确没有航向 PI。该记录证明这轮受控停车，不证明导航成功。[项目记录](</Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/ball_follow_npu_20261010_035720/summary.json>)
- 已有球心/半径观测、坐标变换与视觉误差闭环是可以分离的层次；B-Human 分开描述球检测、相机几何，视觉伺服教程分开描述视觉特征与控制误差。[B-Human 球检测](https://docs.b-human.de/master/perception/ball-and-penalty-mark-detection/)；[相机几何](https://docs.b-human.de/master/perception/perception-infrastructure/)；[视觉伺服论文](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf)

### Inferences

以下是建议新增模块，尚未实现或验证；基于前述模型与控制分层。[几何依据](https://raw.githubusercontent.com/bhuman/BHumanCodeRelease/master/Src/Tools/Math/Transformation.cpp)；[控制依据](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf)

1. **BallMetricObservation**：输入原图球轮廓/框、相机标定、配对姿态及地面关系；输出身体或 odom 坐标的球心、距离/方位、时间、质量标记和不确定性。保留原始检测标签与“完整/截断/遮挡”状态。
2. **BallState**：保存最近有效球状态；用 odom 相对运动传播，用新球观测更新；显式区分 observed/predicted/lost。球运动与机器人自身运动分开建模；预测不确定性增长时停止推进并重获目标。
3. **ApproachSupervisor**：管理搜索、对准、接近、短暂遮挡、重获与完成；终点改为明确的身体/足前几何间隙，保留新鲜度、受控停车和姿态安全条件。目标位置估计不能保证步态实现所需转向，应单独核实实际响应。

建议验证顺序，均先采集证据再选择门限，不预设秒数或精度保证：[标定与投影依据](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)；[视野限制依据](https://faculty.cc.gatech.edu/~seth/res.php?u=vs)

- **离线静止位置**：覆盖不同已知球位置、头 yaw/pitch、完整/边缘/截断检测；分别测直径法、落地法及一致性。MuJoCo 真值只用于评分，不得进入部署算法。
- **静止遮挡**：保持机器人不动，遮球再出现或移动球；验证预测不会被记作新观测、错误关联可拒绝、状态过期能停车。
- **运动与转向**：分别验证直行、左右转、停走切换时的实际身体位移/航向响应与估计误差；再接入接近监督器。仅证明“发了正 yaw”不足以证明“身体向左转”。
- **运动中遮挡与重获**：统计预测误差、不确定性、重获关联、最终足前间隙及安全停止情况；在这些结果出来前，不批准无限期依赖旧目标继续走。

### Gaps

- 未完成单目米制观测标定、状态估计实现、运动响应验证或遮挡导航实验。本笔记给出研究支持的最小方向，不构成任何硬件或闭环成功声明。[当前项目边界记录](</Users/ethan/Documents/RK3588 Microduck/task/evidence/G4/ball_follow_npu_20261010_035720/summary.json>)
