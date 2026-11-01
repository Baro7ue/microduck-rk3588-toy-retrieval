# RoboCup 人形机器人如何寻球、记球与接近

研究日期：2026-10-10。仅阅读官方资料和源码；本轮没有修改 MicroDuck 产品，也没有运行板子、仿真闭环或调参。

## 版本与证据边界

- B-Human：官方 BHumanCodeRelease 的 2025 年度发布，固定提交 [`d89d603`](https://github.com/bhuman/BHumanCodeRelease/commit/d89d603f0388208cecd1c3144b6e8eb92aef4a93)，提交日期 2025-11-06。年度发布不能等同于团队此刻的内部比赛版本。[官方仓库说明](https://github.com/bhuman/BHumanCodeRelease)
- Bit-Bots：官方 bitbots_main 的 main 快照，固定提交 [`a9b6276`](https://github.com/bit-bots/bitbots_main/commit/a9b62760085e77f36e541ea00eacab71d5a4775a)，提交日期 2026-10-08。这是可核实的公开开发版本，未核实其所有分支是否用于某场比赛。[官方仓库](https://github.com/bit-bots/bitbots_main)
- 下列源码均于 2026-10-10 获取。完整来源清单（仓库、固定提交、文件路径、原文链接）在同目录 `robocup_sources.json`；原始副本没有放进公开仓库（它们属于 B-Human 和 Bit-Bots，各有自己的许可证），按清单里的 `url` 字段读原文。参数是所读代码的默认值或该份配置值，不是可直接复制到 MicroDuck 的指标。

## 关键问题一：一次看到球后，成熟机器人记住的是什么？

### Takeaway

它们保存有坐标系和可信度的球状态，后续视觉不断修正。图像漏检时仍能保留估计，但这不等于允许无限沿原指令前进。B-Human 明确建模球的运动；当前 Bit-Bots 这份滤波器主要保存静态位置并增长不确定性，两者不能混称同一种 Kalman 球追踪器。

### Cited Findings

- **B-Human 的表示**：`BallModel` 含机器人相对位置（mm）、速度（mm/s）、位置协方差、最后观测，以及 `timeWhenLastSeen` 和“应该看到却没看到”的 `timeWhenDisappeared`。估计状态与最近一次视觉测量分开保存。[BallModel.h](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Representations/Modeling/BallModel.h)
- **B-Human 的预测**：每帧用里程计逆变换，将静止球和滚动球假设更新到当前机器人坐标；滚动假设还推进位置、处理减速。视觉重现时再更新假设、选取结果；协方差也随运动和噪声传播。4 秒后新观测触发旧滤波器重置，不能解读为允许盲走 4 秒。[BallStateEstimator.cpp](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/Modeling/BallStateEstimator/BallStateEstimator.cpp)
- **Bit-Bots 的表示**：输入已经是 3D `BallArray`，通过带时间戳的 TF 变换到 `odom` 等滤波坐标系，保存位置和协方差。当前实现没有球速度状态；没有新测量时保留该坐标中的位置，定时增加不确定性。匹配观测后直接更新位置与测量协方差，因此不能仅凭模块名宣称它使用完整运动 Kalman 模型。[ball_filter.py](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_world_model/bitbots_ball_filter/bitbots_ball_filter/ball_filter.py)
- **Bit-Bots 的消费方式**：blackboard 保存场地球位置，按当前机身变换取得 `base_footprint` 中的前向/左向坐标，提供米制距离和 `atan2` 方位。获取一个返回位置不保证它仍可信，行为层需同时检查有效性。[world_model_capsule.py](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_blackboard/bitbots_blackboard/capsules/world_model_capsule.py)

### Inferences

对 MicroDuck，最小可借鉴形式是 `位置/方位 + 坐标系 + 观测时间 + 不确定性/有效性`。如果暂时只考虑静止球，保持一个短期静态球位置假设已经比“沿最后像素误差继续走”更有明确含义；还不需要直接复制球速度、多机器人通信或全场定位。

### Gaps

本轮没有审计 Bit-Bots 上游由图像生成 3D 球测量的全部实现，也没有验证 MicroDuck 当前里程计的误差、单位和时间对齐。没有这些验证，不能宣称已有可靠米制导航。

## 关键问题二：丢球多久还能预测，何时减速、停止或重搜？

### Takeaway

没有两队通用的“丢球 N 秒继续前进”常数。需要分开看估计器是否保留状态、行为层是否接受它，以及搜索动作是否允许机身移动。成熟实现也会在没看见球时转身、后退或去搜索位置，但这些是有条件的恢复动作，并非重复最后的接近指令。

### Cited Findings

| 实现 | 核实的触发条件 | 核实的动作与限制 |
|---|---|---|
| B-Human 近球恢复 | 执行处理球任务，球距 <700 mm；应见未见 >500 ms；排除障碍遮挡 | 扫头并短退；最多退 1 s，或退至少 0.5 s 且里程计后退超过 100 mm 后，转向最后球的方向。重新看到球即可退出。只适用于这个近球分支。 |
| Bit-Bots 球可信度 | 自身球协方差各对角项均 <配置值 2.0，或有有效队友球 | `BallSeen` 每次重评；无可信球进入角色相关搜索流程。并没有在该判断中使用固定最后观测秒数。 |

表中两条分别来自 [HandleStrikerLostBall.cpp](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/BehaviorControl/SkillBehaviorControl/Options/HandleStrikerLostBall.cpp)；[ball_seen.py](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_body_behavior/bitbots_body_behavior/behavior_dsd/decisions/ball_seen.py)、[blackboard](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_blackboard/bitbots_blackboard/capsules/world_model_capsule.py) 与 [body_behavior.yaml](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_body_behavior/config/body_behavior.yaml)。

- B-Human 先检查物理球投影及机身遮挡，才认为当前视野中应该看到球。默认可检查距离为 1 m，累计应见未见达到 7 次后记录消失；`700 ms` 是旧观测可用于速度计算的期限，不是通用运动许可。[BallLocatorTools.h](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Tools/Modeling/BallLocatorTools.h)、[估计器参数](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/Modeling/BallStateEstimator/BallStateEstimator.h)
- Bit-Bots 默认滤波频率 62 Hz，每个定时步骤给平面协方差加 0.002；若预测球在视野内却没有匹配检测，还将不确定性乘 1.5。因初始置信度、负观测频率及队友球不同，不能据此给出固定安全盲走秒数。[滤波配置](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_world_model/bitbots_ball_filter/config/ball_filter_parameters.yaml)、[filter_step 与负观测处理](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_world_model/bitbots_ball_filter/bitbots_ball_filter/ball_filter.py)
- Bit-Bots 的 `SearchBall` 分支先原地走 1 秒，再向最后球侧转 6 秒，随后去场地搜索点、转 15 秒、换位置，最后站立；角色不同可能采用其他分支。6/15 秒是搜索动作时长，不是盲目向球前进的容忍时间。转向动作本身仅发送旋转速度。[main.dsd](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_body_behavior/bitbots_body_behavior/behavior_dsd/main.dsd)、[turn.py](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_body_behavior/bitbots_body_behavior/behavior_dsd/actions/turn.py)
- B-Human 更广范围的搜索使用场地粒子/网格，结合已观察视野和遮挡选择下一个观察区域。搜索请求指向一个场地位置，体现主动改变观察方向和位置。[BallSearchParticlesProvider.cpp](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/BehaviorControl/BallSearchParticlesProvider/BallSearchParticlesProvider.cpp)、[BallSearch.h](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/BehaviorControl/StrategyBehaviorControl/BallSearch.h)
- Bit-Bots 导航控制器将平移速度设为剩余路径距离乘比例系数，并限速、平滑；这是接近目标时的距离减速，不是已核实的“按丢球时间减速”。取消规划也不自动停止走路，接口另有 `stop_walk`。[controller.py](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_navigation/bitbots_path_planning/bitbots_path_planning/controller.py)、[pathfinding_capsule.py](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_blackboard/bitbots_blackboard/capsules/pathfinding_capsule.py)

### Inferences

建议 MicroDuck 把三件事分开：短漏检时更新估计；不确定性越界时退出接近；重搜时主动转头或转身。当前缺少里程计可靠性证据时，维持现有丢球零速保护，再增加受控重搜，比单纯延长漏检容忍时间更容易验证。这是项目建议，尚未实现或实验。

### Gaps

本轮未追踪 B-Human 所有角色进入全场搜索的统一门限，也没有发现两队共享的统一“协方差 → 行走速度”公式。上述 500 ms 不能推广到所有 B-Human 行为；Bit-Bots 的配置注释有把协方差称为标准差的表述，应以代码实际比较量为准。

## 关键问题三：2D tracker 与米制球状态有什么不同，接近和头部怎样配合？

### Takeaway

2D 跟踪解决的是“这一帧的框是否延续上一帧目标”；米制状态解决的是“球相对机器人在哪里、机器人动了以后应在哪里”。前者可以改善关联和短时检测连续性；仅有像素框、面积与帧号，还无法定义足前到球 20 cm 的停车条件。

### Cited Findings

- B-Human 接近技能使用球预测终点与踢球偏移构造 `kickPose`，远处使用路径规划、近处转为局部走球避障；头部控制在近目标约 250 mm 时看球、超过 310 mm 时恢复其他观察，采用滞回。这些距离对应技能目标，不等于 MicroDuck 足前净距。[GoToBallAndKick.cpp](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/Ball/GoToBallAndKick.cpp)、[GoToBallHeadControl.cpp](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/Head/GoToBallHeadControl.cpp)
- Bit-Bots `GoToBall` 会持续计算并发布目标姿态；近球目标从当前机身相对球位置减去指定接近距离和侧偏置。配置中的 0.2 m 是其导航/踢球参考距离，不能直接等同于我们脚尖至球表面的 20 cm。[go_to_ball.py](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_body_behavior/bitbots_body_behavior/behavior_dsd/actions/go_to_ball.py)、[目标几何](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_blackboard/bitbots_blackboard/capsules/pathfinding_capsule.py)、[接近距离配置](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_body_behavior/config/body_behavior.yaml)
- Bit-Bots 头部 `TRACK_BALL` 对米制球点做 TF 变换并计算关节目标；搜索模式则生成平滑的左右/俯仰扫描轨迹。机身行为显式选择头部模式，头部扫描与机身导航分工。[move_head.cpp](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_motion/bitbots_head_mover/src/move_head.cpp)、[head_modes.py](https://github.com/bit-bots/bitbots_main/blob/a9b62760085e77f36e541ea00eacab71d5a4775a/src/bitbots_behavior/bitbots_body_behavior/bitbots_body_behavior/behavior_dsd/actions/head_modes.py)

### Inferences

| 层 | 对 MicroDuck 应回答的问题 | 仅有 2D 框能否完成 |
|---|---|---|
| 检测/关联 | 这是球吗，是否还是同一个目标？ | 可以承担主要部分，但仍有误检和漏检 |
| 几何与状态估计 | 球在机身前方几米、左侧几米？估计有多可信？ | 需要额外相机标定、姿态/外参、深度约束与时间信息 |
| 运动预测 | 我走或转以后，静止球相对我应怎样变化？ | 单纯像素框外推无法可靠区分球动与相机动；需要自身运动信息 |
| 行为/控制 | 继续接近、减速、站住，还是主动重搜？ | 应使用状态有效性和米制目标，不能由 tracker 单独决定 |

建议研究路线是先定义球位置的参考系及“足前净距”几何，再验证静态距离与自身运动补偿，之后讨论估计器和搜索状态机。RoboCup 的可借鉴重点是这些层的职责；具体门限、NAO/Bit-Bots 球尺寸、场地坐标及步态速度不能直接搬来。

### Gaps

这些公开实现说明设计机制，并不证明它们在 MicroDuck 的相机、步态和硬件上可直接工作。本研究没有对当前 MicroDuck 的失败试验作新的闭环验证，也没有评测 YOLO tracker、SLAM 或导航软件包的实际效果。
