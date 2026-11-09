# Real Experiment Deployment

OpenWAM / EasyWAM 双臂与 Sharpa 手部署代码快照。支持 badminton（羽毛球）、top（拉绳陀螺）、bottle（拧盖倒水）的已接入配置。

**本仓库不包含模型权重、SDK 原生库、示范数据或参考图像，不能克隆后直接运行真机。** 先恢复环境与资源，再做离线检查。`--run` 会自动回起点并执行动作；`--worker-only` 只启动推理服务。

- [当前部署配置总结](deployment/DEPLOYMENT_CONFIG_SUMMARY.md)：最近使用的参数、测试结果与已知问题。
- [部署目录历史说明](deployment/README.md)：保留迁移记录；其中旧的模型数量和 RTC 描述以配置总结及本文件为准。
- [模型登记表](deployment/assets/checkpoints/models.json)：权重文件名、训练步数、SHA256。

## 1. 目录与准备条件

```text
requirements/                    # 原机器三个环境的版本快照
deployment/
  scripts/launch.py              # 统一入口
  configs/profiles/              # 模型、控制频率、硬件与任务组合
  configs/inference/             # OpenWAM 推理参数
  src/                          # 执行、队列、EasyWAM、RTC、CUDA Graph
  vendor/                       # 项目使用的模型/infra/bridge 源码副本
  assets/checkpoints/            # 任务权重及配套 stats/context/tokenizer
  assets/base_models/            # EasyWAM Cosmos 基础模型
  assets/robot/                  # URDF 和运动学代码
  assets/references/             # 起点及预热样本
  sdk/{tianji,sharpa}/            # 自行恢复厂商 SDK
  envs/{openwam,easywam,robot}/   # 本项目独立 Python 环境
  deps/{bridge,infra,observer}/  # 项目内补充依赖
  runs/                         # 每轮日志、PID、动作、反馈和停止确认
  offline_checks/               # 本机验证报告与实验门禁
```

原机器是 Linux x86_64 + RTX 4090，三个环境均为 **Python 3.10.12**；模型环境使用 **PyTorch 2.7.1+cu128**。需有可用 NVIDIA 驱动和 `nvidia-smi`。SDK 的 Python 扩展也必须与 Python ABI、CPU 架构匹配。macOS 可编辑代码，但不属于该真机运行环境。

```bash
git clone https://github.com/ljh-qh/real-experiment-deployment.git
cd real-experiment-deployment
export REPO_ROOT="$PWD"
export DEPLOY="$REPO_ROOT/deployment"
mkdir -p "$DEPLOY"/envs "$DEPLOY"/runs "$DEPLOY"/offline_checks
mkdir -p "$DEPLOY"/deps/{bridge,infra,observer}
```

后续命令均在仓库根目录执行。这里的环境变量只用于当前终端，不必改全局 shell、Conda 或 SDK 配置。

## 2. 安装独立环境

### 2.1 创建虚拟环境

```bash
python3.10 -m venv "$DEPLOY/envs/openwam"
python3.10 -m venv "$DEPLOY/envs/easywam"
python3.10 -m venv "$DEPLOY/envs/robot"
```

`requirements/*.txt` 是原机器已安装包的版本清单，**不是经过全新系统安装验证的跨平台 lockfile**。若版本在当前镜像不可用，请使用可获取该版本的官方源/已有 wheel 缓存，不要无记录地替换版本。

先安装模型环境的 CUDA 版 PyTorch，再安装剩余快照包；本地源码包不从 PyPI 拉取，FlashAttention 单独安装：

```bash
for env in openwam easywam; do
  "$DEPLOY/envs/$env/bin/python" -m pip install \
    torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
done

python3.10 - <<'PY'
from pathlib import Path
root = Path.cwd()
out = root / 'deployment/offline_checks/install_requirements'
out.mkdir(parents=True, exist_ok=True)
skip = {'torch', 'torchvision', 'flash-attn', 'mujoco-marvin', 'easywam', 'openwam', 'unirobot-infra'}
for env in ('openwam', 'easywam', 'robot'):
    lines = (root / 'requirements' / (env + '.txt')).read_text().splitlines()
    kept = [s for s in lines if s.split('==')[0].lower().replace('_', '-') not in skip]
    (out / (env + '.txt')).write_text('\n'.join(kept) + '\n')
PY

for env in openwam easywam robot; do
  "$DEPLOY/envs/$env/bin/python" -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r "$DEPLOY/offline_checks/install_requirements/$env.txt"
done
```

使用 EasyWAM CUDA Graph + FA2 优化 profile 时还需要匹配 PyTorch/CUDA/Python 的 FlashAttention 2。原机器版本为 `2.8.3.post2`：

```bash
# 优先使用匹配的官方 wheel；需要源码构建时，还需兼容 CUDA toolkit/nvcc 和编译工具。
"$DEPLOY/envs/easywam/bin/python" -m pip install ninja packaging wheel
"$DEPLOY/envs/easywam/bin/python" -m pip install \
  flash-attn==2.8.3.post2 --no-build-isolation
```

不要将三个环境合并：原 OpenWAM 与 EasyWAM 的 transformers 等依赖版本不同。

### 2.2 补充依赖与源码加载

原部署有通过 `--target` 放置的补充依赖，它们未全部出现在虚拟环境版本快照中：

```bash
"$DEPLOY/envs/robot/bin/python" -m pip install \
  --target "$DEPLOY/deps/infra" omegaconf==2.3.0 rich==14.2.0
"$DEPLOY/envs/robot/bin/python" -m pip install \
  --target "$DEPLOY/deps/observer" msgpack==1.2.4
"$DEPLOY/envs/openwam/bin/python" -m pip install \
  --target "$DEPLOY/deps/bridge" pyzmq==27.1.0 ruckig==0.12.2
```

上述安装命令适用于新建的空目录；不要直接覆盖已经验证的现网环境。

启动器和 profile 通过 `PYTHONPATH` 加载 `vendor/` 源码及 `deps/`，不需要额外 `pip install -e vendor/universal_robot_infra`。该 infra 快照的打包声明要求 Python 3.12，但原部署实际以源码导入方式运行在 Python 3.10；不能把它的打包安装流程与本仓库的运行方式混用。源码路径设置见 [deployment_paths.py](deployment/src/deployment_paths.py)。

## 3. 放置权重与配套资源

从有权限的模型发布源或原部署机恢复资源。本仓库不提供所有模型的统一下载链接，也不包含授权凭据。

| 模型 | 放置目录（相对于 deployment/） | 权重文件 | 启动支持 |
|---|---|---|---|
| OpenWAM badminton | `assets/checkpoints/openwam_badminton/` | `checkpoint_step_35705.safetensors` | 已接入 |
| OpenWAM top | `assets/checkpoints/openwam_top/` | `checkpoint_step_33050.safetensors` | 已接入 |
| OpenWAM bottle | `assets/checkpoints/openwam_bottle/` | `checkpoint_step_65640.safetensors` | 已接入 |
| EasyWAM badminton | `assets/checkpoints/easywam_badminton/` | `step_004465.pt` | `easywam_v1` |
| EasyWAM top | `assets/checkpoints/easywam_top/` | `step_004131.pt` | **只有权重登记，尚无 top adapter/profile** |
| EasyWAM bottle | `assets/checkpoints/easywam_bottle/` | `step_008205.pt` | 已接入 |

### OpenWAM

每个任务目录需要对应训练配置 `config.yaml`、`normalization_stats.npy`、权重，以及该配置指定的 tokenizer/结构资源。必须整套匹配，不要在任务之间借用统计文件。

现有 badminton 配置使用 `tokenizer/google/umt5-xxl/` 子目录；其他模型按自身 `config.yaml` 恢复。若配置采用自包含 Reason1，还需对应 `reason1/` 结构文件。**不要仅复制一个 safetensors 文件。**

### EasyWAM

```text
assets/
  base_models/Cosmos-Predict2.5-2B/   # 完整基础模型目录，非空占位
  checkpoints/
    easywam_badminton/
      config.yaml
      step_004465.pt
    easywam_common/
      normalization_stats_62.npy
      badminton_context.pt
    easywam_bottle/
      step_008205.pt
      config.yaml                   # 原训练任务 override
      deployment_model.yaml         # 实际部署加载的解析模型配置
      normalization_stats_62.npy
      bottle_easywam_context.pt
      deployment_provenance.json
```

bottle 的 62 维统计来源和投影索引记录在 provenance 文件中。不要用 badminton 的统计或 context 替代。EasyWAM top 的权重和 `top_context.pt` 即使恢复，也不等于已完成启动接入。

校验已恢复权重（只读文件，不连接硬件）：

```bash
python3.10 - <<'PY'
from pathlib import Path
import hashlib, json
root = Path('deployment/assets/checkpoints')
registry = json.loads((root / 'models.json').read_text())['models']
for name, item in registry.items():
    file = root / name / item['checkpoint']
    if not file.is_file():
        print(name, 'NOT RESTORED')
        continue
    digest = hashlib.sha256()
    with file.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == item['sha256'], f'{name}: SHA256 mismatch'
    print(name, 'SHA256 OK')
PY
```

### 起点与预热观测

起点 JSON 已包含在仓库；预热张量和图片未上传，需另行恢复：

| 任务 | 起点（assets/references/ 下） | 预热目录 |
|---|---|---|
| badminton | `badminton_demo_origin.json` | `fixed_observation_diagnosis/` |
| top | `top_demo_origin.json` | `top_observations/` |
| bottle | `bottle_demo_origin.json` | `bottle_observations/` |

每个预热目录按原结构恢复 `recorded_start/`、`recorded_mid/`、`live_end_reconstructed/`，样本内含 `state80.npy`、`main.png`、`wrist.png`、`wrist2.png`。具体加载范围由 worker 决定；离线回归会使用多个样本。top/bottle 的 `live_end_reconstructed` 是历史命名，实际是录制末段，不代表实时观测。

录制腕部图像已经旋转，不能再重复旋转。实时相机旋转由采集端负责。`prepare_*_assets.py` 依赖未上传的原始示范数据，不是下载脚本。

## 4. 放置 SDK 与适配机器路径

SDK 从厂商或已有授权部署获取，不包含在 Git 仓库中。

```text
sdk/
  tianji/                           # 原部署的平铺布局
    fx_robot.py
    ccs_m6.MvKDCfg
    libMarvinSDK.so
    libKine.so
    libInterfCheck.so
  sharpa/
    python/sharpa/
      __init__.py
      sharpa.cpython-310-x86_64-linux-gnu.so
    lib/
      libsharpa-wave-sdk.so
      libsharpa-discovery-delay.so   # 原部署的配套库；按授权发行包恢复
    config/tactile.json
    config.yaml
```

Tianji loader 也支持厂商的 `fx_robot.py + structure_data.py + lib/` 分拆布局，保留对应发行包的相对路径，不要混搭不同版本。Sharpa Python 扩展与原生库须配套；原实验使用 SDK 5.0.8、手固件 3.0.11，仅作为复现信息，不自动升级固件。

新机器至少检查以下文件：

- `vendor/universal_robot_infra/configs/robot_hardware/arm/tianji.yaml`：`sdk_dir`、`kine_cfg` 改为本机绝对路径，核实 `robot_ip`。
- `vendor/universal_robot_infra/configs/robot_hardware/end_effector/sharpa.yaml`：`sdk.vendor_root` 指向本机 `deployment/sdk/sharpa/python`；核实主机 IP、左右手 SN/IP。
- `src/infra_state_client.py`、`src/rtc_state_client.py`、`src/launcher.py` 及回位后端：仍包含原实验设备地址/相机序列号；不是修改一个 YAML 就完成全部移植。
- `assets/checkpoints/models.json` 中 `root`、`directory` 是原机器登记路径；上面的哈希脚本按仓库相对目录校验，移动部署时也应更新登记路径。

profile 中的 `$DEPLOY` 和 `$OUTPUT` 由启动器替换；**硬件 YAML 不会自动执行同样的变量替换**。不要把字面量 `$DEPLOY` 写进要求绝对路径的 SDK 字段。

原设备设置：机械臂 `192.168.1.190`；左/右手 `192.168.10.10/.20`；相机 main `241122305532`、左腕 `262422271713`、右腕 `260522271432`。部署时应核实本机网络和设备，不要为复用这些值而修改他人的全局网络设置。

## 5. 离线检查与仅启动模型

```bash
# --check 是默认行为：路径/profile 检查，不启动模型、不连接硬件。
"$DEPLOY/envs/robot/bin/python" "$DEPLOY/scripts/launch.py" \
  --profile openwam_sync --check

# 资源恢复后，可执行现有 CPU 回归；其中部分测试需要参考样本/本机验证文件。
CUDA_VISIBLE_DEVICES='' "$DEPLOY/envs/openwam/bin/python" -B \
  -m unittest discover -s "$DEPLOY/tests" -v

# 只加载 GPU 模型和预热；不会回位或发送机器人指令。Ctrl+C 结束。
nvidia-smi
"$DEPLOY/envs/robot/bin/python" "$DEPLOY/scripts/launch.py" \
  --profile easywam_bottle_sync_lanczos --worker-only
```

`--check` 不证明 SDK 可连接、统计内容匹配或任务能成功；目录存在也不代表目录中的资源完整。启动器发现 GPU 已有计算进程会拒绝竞争资源。模型默认占用回环 ZMQ 17200/17201/17202，不要同时启动冲突实例。

**普通异步/RTC 的验证文件未随代码导出**，因此相关 profile 的 `--check`、`--worker-only`、`--run` 都可能因缺少验证文件或门禁未通过而停止。这是预期行为：先用 sync 完成资源/模型检查，再根据 profile 指向的验证文件完成真实的本机推理、通信、IK、队列期限验证。仓库尚未提供一键重建所有历史验证文件的脚本，不要手工填一个 `hardware_ready=true` 绕过验证。

## 6. 各任务如何启动

以下是**真机命令模板**，这里只展示，不会自动执行。运行前需完成对应资源与设备核验、停止遥操/Pilot 占用并摆好场景。启动顺序为：模型加载/预热 → 对应任务回起点 → 执行。`duration=0` 表示持续执行至人工停止或保护触发，不会自动识别任务完成。

### OpenWAM

```bash
# badminton：35705，30Hz sync，BILINEAR，20/20，手部2°/步
"$DEPLOY/envs/robot/bin/python" "$DEPLOY/scripts/launch.py" --profile openwam_sync --run

# top：33050，30Hz sync，BILINEAR，20/20，手部2°/步
"$DEPLOY/envs/robot/bin/python" "$DEPLOY/scripts/launch.py" --profile openwam_top_sync --run

# bottle：65640，30Hz sync，LANCZOS，DiT cache开启
"$DEPLOY/envs/robot/bin/python" "$DEPLOY/scripts/launch.py" --profile openwam_bottle_sync_lanczos --run
```

可替换的已保存 profile：

| Profile | 差异 |
|---|---|
| `openwam_sync_20hz` | badminton，20Hz sync |
| `openwam_badminton_sync_lanczos` | badminton，LANCZOS |
| `openwam_top_sync_lanczos` | top，LANCZOS |
| `openwam_bottle_sync` | bottle，BILINEAR |
| `openwam_bottle_sync_no_dit_cache` | bottle，BILINEAR，关闭DiT cache |
| `openwam_bottle_sync_no_dit_cache_lanczos` | bottle，LANCZOS，关闭DiT cache |
| `openwam_async_rtc` | badminton的历史RTC实验；需要对应验证门禁，不代表已解决抖动 |

sync配置默认10步去噪、32步动作，DiT cache阈值0.99、最多跳过3次。sync换块仍有等待，30Hz不是整段无停顿的有效频率。

### EasyWAM

```bash
# badminton：4465；注意这是历史参数：50/50、手部3°/步、30Hz sync、运行30秒
"$DEPLOY/envs/robot/bin/python" "$DEPLOY/scripts/launch.py" --profile easywam_v1 --run

# bottle：8205，30Hz sync，20/20、手部2°/步、LANCZOS
"$DEPLOY/envs/robot/bin/python" "$DEPLOY/scripts/launch.py" --profile easywam_bottle_sync_lanczos --run

# bottle普通异步：30Hz，剩余24步时预取；需本机异步验证通过
"$DEPLOY/envs/robot/bin/python" "$DEPLOY/scripts/launch.py" --profile easywam_bottle_async_lanczos --run

# bottle RTC优化版：20Hz，commit14/prefix24；需本机RTC验证通过
"$DEPLOY/envs/robot/bin/python" "$DEPLOY/scripts/launch.py" --profile easywam_bottle_async_rtc_cuda_graph_20hz --run
```

**EasyWAM top 暂无启动命令**：当前 adapter 只支持 `v1`（badminton）和 `bottle`，不能只把checkpoint目录改成top就运行。

20Hz RTC优化版采用identity-VJP近似、FA2与CUDA Graph，不加输出平滑；原机器推理约492ms。它不是逐元素等价替换，三组固定样本相对原后端最大TCP目标差约4.1mm、手指约0.85°。20Hz已完成一轮时序测试，但仍有块内抖动及接触后手部跟踪误差问题。30Hz RTC profile仍未通过原提交窗口时限，不应直接套用20Hz验证文件。

### 日志与停止

每次运行自动创建 `runs/<profile>_<时间>/`；显式 `--output` 必须在deployment目录内且是尚不存在的新目录。可查看：

- `model_config.yaml`、`effective_config.json`：解析后与实际生效配置。
- `worker_ready.worker.log`、`client.log`、`return.log`：模型、执行、回位日志。
- `stage.json`、`result.json`、`stop_confirmation.json`：阶段、结果、停止确认。
- `runner.pid`、`runner_identity.json`：本轮启动器身份。

前台按 **Ctrl+C**，等待停止确认。后台只对核实了PID、启动时间及命令行的本项目启动器发送SIGINT；不要使用pkill/killall。停止后以实际idle/双手关闭记录为准，不能仅凭终端退出判定硬件状态。

## 7. 不上传的文件与许可

`.gitignore` 排除权重/张量、实验数据、日志、虚拟环境及SDK二进制；目录用`.gitkeep`保留。恢复大文件后，可先运行 `git status --short` 检查再提交，避免强制添加被忽略资源。

`vendor/` 保留已有LICENSE、NOTICE与README；本仓库不额外授予第三方源码、权重或SDK许可。部署代码发布不包含全局凭据，也不改写原机器的遥操链路。
