# 部署配置总结

更新日期：2026-11-09（Asia/Shanghai）。本文件记录配置与最近实验结果，不代表当前硬件实时状态，也不授权自动回位或启动真机。

远程主机：4090-zhenji，用户：admin2026。
部署根目录：/home/admin2026/lianjh/real_experiment/deployment/。
下文路径均相对于部署根目录。

## 最近实际测试配置

EasyWAM bottle，训练步数8205，20Hz + 异步RTC + CUDA Graph优化版。
配置文件：configs/profiles/easywam_bottle_async_rtc_cuda_graph_20hz.yaml。
原有OpenWAM及EasyWAM sync配置保留，未被替换。

| 项目 | 配置 |
|---|---|
| 权重 | assets/checkpoints/easywam_bottle/step_008205.pt |
| 控制频率 | 20Hz |
| 执行方式 | 异步 + RTC |
| RTC | identity_vjp近似引导，不是完整网络VJP |
| 提交长度 / 前缀容量 | 14 / 24步 |
| 动作预测长度 | 32步 |
| 推理步数 | 请求10步，当前调度器实际执行11次动作去噪 |
| 图像 | 三相机拼接，384×320（高×宽），LANCZOS |
| 优化 | FlashAttention 2、编译缓存上限64、显式CUDA Graph |
| 臂速度 / 加速度比例 | 20 / 20 |
| 臂额外单步限幅 | 关闭；关节角度限位仍保留 |
| 手部单步限幅 | 2°/步 |
| 跟踪保护 | 臂8°、手15° |
| 超限确认等待 | 0ms |
| 时长 | duration=0，不限时，人工停止或保护终止 |
| 平滑 / 高度补偿 / 手部提前量 | 未启用 |
| 起点 | assets/references/bottle_demo_origin.json |
| 随机种子 | 真机配置不固定；离线对照曾用42 |

RTC按提交长度与前缀容量调度；该profile保留的prefetch_delay=16不是RTC的提交窗口。
20Hz下14步窗口为700ms。30Hz下同一窗口仅466.7ms，不能把20Hz验证结论套用到30Hz。

手部底层为位置模式，speed_coeff=0.6、current_coeff=0.6、skip_tactile=true、interpolate=false。
上述系数不是经过标定的接触力阈值。

## 权重与统计

| 模型 | badminton | top | bottle |
|---|---|---|---|
| OpenWAM | openwam_badminton，35705 | openwam_top，33050 | openwam_bottle，65640 |
| EasyWAM | easywam_badminton，4465 | easywam_top，4131 | easywam_bottle，8205 |

目录：assets/checkpoints/；登记表：assets/checkpoints/models.json。
EasyWAM top已有权重，但当前没有专用top测试profile。
EasyWAM bottle已补入models.json。

EasyWAM bottle专用文件：
- step_008205.pt：原始权重，未修改。
- bottle_easywam_context.pt：任务文本context；prompt为“unscrew the bottle cap and pour water into the cup”。
- normalization_stats_62.npy：由openwam_bottle/normalization_stats.npy中的eef统计投影到62个有效维度。
- config.yaml：保留的原始训练任务override，不是完整解析模型配置。
- deployment_model.yaml：由已有EasyWAM配置重建的部署模型结构；checkpoint action_config与参数形状核验一致，但不应将其称为完整训练配置原件。
- deployment_provenance.json：记录统计投影、来源与哈希。

有效80维索引：0:9、10:32、34:43、44:66（Python切片，上界不含）。
EasyWAM输出62维物理动作后映射回统一80维协议；OpenWAM使用80维表示。

## 主要保留配置

下列文件位于configs/profiles/。

| 文件 | 模式 |
|---|---|
| openwam_sync.yaml | badminton，30Hz sync，BILINEAR |
| openwam_top_sync.yaml | top，30Hz sync，BILINEAR |
| openwam_bottle_sync_lanczos.yaml | bottle，30Hz sync，LANCZOS |
| easywam_bottle_sync_lanczos.yaml | bottle，30Hz sync，LANCZOS |
| easywam_bottle_async_lanczos.yaml | bottle，30Hz普通异步，剩余24步时预取 |
| easywam_bottle_async_rtc_lanczos.yaml | bottle，30Hz RTC，原后端；时限门禁未通过 |
| easywam_bottle_async_rtc_cuda_graph.yaml | bottle，30Hz RTC优化版；时限门禁未通过 |
| easywam_bottle_async_rtc_cuda_graph_20hz.yaml | bottle，20Hz RTC优化版；已完成一次实测 |

另保留OpenWAM 20Hz、RTC、各任务LANCZOS及bottle关闭DiT cache的实验profile。
easywam_v1.yaml为历史badminton配置，不应误认为采用上述bottle的20/20、2°参数。
配置文件存在不代表通过真机验证或任务成功。

OpenWAM sync推理参数：
- 32步动作、9帧视频latent目标、384×320、10步去噪；不解码视频。
- DiT cache开启，cosine_threshold=0.99，max_skips=3。
- compile、prompt_embed_cache与worker context-cache开启。
- 关闭DiT cache版本另存独立profile。
EasyWAM没有沿用OpenWAM的DiT cache开关。CUDA Graph/编译优化不等同于跳过去噪计算。
sync的30Hz指块内发送节奏，换块等待会降低整体有效执行频率。

## 图像与执行链路

相机输入 → 按传感器安装旋转 → 主视角/左腕/右腕拼接 → 模型预处理与归一化 → 32步动作 → sync、普通异步或RTC队列 → EEF转IK关节目标 → 限位及手部步长处理 → 位置控制 → 反馈保护。

主视角旋转0°，左右腕实时输入旋转90°；录制视频已经旋转，top/bottle参考样本的重复旋转已修复。
LANCZOS布局：主视角320×256，左右腕各160×128，拼成320×384图像。
BILINEAR原路径仍保留；不自动将所有旧profile切换为LANCZOS。
RTC前缀来自已规划、将执行的物理动作，不是简单拼接原始预测。
模型服务使用回环ZMQ端口17200/17201/17202；启动时应避免多个模型实例占用同一组端口。

## 代码及数据位置

| 路径 | 作用 |
|---|---|
| scripts/launch.py、src/launcher.py | 统一入口、配置检查、进程管理、回位与停止确认 |
| vendor/bridge/zmq_chunk_worker.py | OpenWAM推理worker |
| src/easywam_adapter.py、src/easywam_worker.py | EasyWAM加载、预处理、归一化及推理服务 |
| src/easywam_rtc.py | EasyWAM部署层RTC引导 |
| src/easywam_cuda_graph.py | 可选CUDA Graph/FA2执行优化 |
| src/infra_state_client.py | sync与普通异步执行 |
| src/rtc_state_client.py、src/rtc_queue.py | RTC规划、前缀、期限与执行 |
| src/accepted_client.py、src/confirmed_client.py | 应用硬件profile的执行包装 |
| vendor/bridge/return_full_origin.py | 独立回位 |
| assets/references/ | 各任务起点与预热观测 |
| assets/checkpoints/、assets/base_models/ | 任务权重、统计、context及基础模型 |
| envs/openwam、envs/easywam、envs/robot | 项目内独立运行环境 |
| vendor/、sdk/ | 本部署依赖代码及硬件SDK |
| runs/ | 真机日志、effective_config、动作、反馈与停止结果 |
| offline_checks/ | 离线实验、性能验证、分析及门禁证据 |

运行链路使用deployment内资产与依赖；本次整理未修改遥操链路或全局环境。
runs/current_task.json和current_openwam_test.json记录最近选择/启动信息，不是实时运行状态；后者文件名是历史命名，也记录EasyWAM运行。

## 验证结果与未解决问题

最近模型实测目录：
runs/easywam_bottle_async_rtc_cuda_graph_20hz_20261108_194427/。

- 优化后离线24次RTC推理：中位491.98ms，P95 492.80ms，最大492.87ms；125项回归通过。
- 20Hz离线全链路100步、6次换块通过，通信推理约496ms，后续IK约16～18ms。
- 20Hz真机7次换块均按时完成；约5.6秒、112步后因手部误差16.09°停止，当时臂误差约1°。
- 该次模型运行停止记录确认双臂idle、双手closed；不是对当前硬件状态的实时保证。
- 左食指DIP接触后反馈停在约23.52°，目标继续到39.61°；尚未实现接触态握持控制。
- RTC改善接缝，但块内旋转目标波动及现场抖动仍未解决。
- 优化后端不保证数值等价：固定样本对照最大TCP目标差约4.1mm、手指约0.85°，未证明这些变化对所有任务无影响。
- 硬件越限故障曾反复出现；不能用放宽部署跟踪阈值代替硬件故障处理。
- 臂限位来源冲突：部署J1/J3/J5为±170°，URDF/回位默认±178°，SDK随附配置±170°；J4部署/回位为−145～60°，SDK文件为−140～78°。未取得控制器生效配置前未擅自统一或放宽。
- 手部22关节部署/回位范围逐项一致；2°/步与8°/15°跟踪阈值不是角度范围限位，均未取消。

证据目录：
- offline_checks/easywam_rtc_20261108/
- offline_checks/easywam_perf_20261108/
- offline_checks/easywam_rtc20_20261108/
- offline_checks/easywam_rtc20_jitter_20261108/
- offline_checks/limits_audit_20261108/

20Hz验证文件只支持对应实验条件下的人工监督测试；不代表长期可靠性、任务成功率或可无人值守运行。
