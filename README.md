# Group 14 电力数据异常检测

本项目包含四种独立的电力数据异常检测方法：LSTM Autoencoder、LSTM-VAE、RPCA 和 Multi-scale Matrix Profile。仓库已经按照配置、数据、模型代码、集群任务、运行结果和公共工具重新组织。四个模型保留各自的原生输出，同时生成统一的15分钟输出，供后续集成学习使用。

## 1. 目录结构

```text
group14reshape/
|-- configs/                         # 项目与模型配置
|-- data/
|   |-- raw/                         # 原始数据
|   |-- processed/meter_csvs/        # 每个电表通道一个合并CSV
|   `-- manifests/                   # 数据与集群任务清单
|-- docs/                            # 项目文档
|-- legacy/                          # 不参与当前流程的旧文件
|-- methods/
|   |-- lstm_autoencoder/
|   |-- lstm_vae/
|   |-- rpca/
|   `-- matrix_profile/
|-- runs/                            # 模型、日志、图片和检测结果
`-- shared/
    |-- data_tools/                  # 公共数据预处理
    `-- scripts/                     # 聚合与项目级脚本
```

重要配置：

- `configs/common.yaml`：统一时间范围和采样尺度。
- `configs/output_contract.yaml`：统一输出字段、校准规则和Matrix Profile分辨率声明。
- `configs/lstm_autoencoder.yaml`：LSTM Autoencoder可执行配置。
- `configs/lstm_vae.yaml`、`rpca.yaml`、`matrix_profile.yaml`：记录对应模型参数；部分参数仍由CLI或Slurm传入。

## 2. 原始数据格式

统一输入位置：

```text
data/processed/meter_csvs/<series_id>.csv
```

`series_id` 表示一个电表通道，例如 `DD03710131709_2`。CSV必须包含 `timestamp` 和模型所需的电力特征。

原始采样间隔为5分钟，正常自然日应包含：

```text
24 × 60 ÷ 5 = 288个时间点
```

时间戳读取后统一转换到 `Australia/Sydney`。

## 3. 统一训练与测试时间

项目采用左闭右开的时间范围，避免午夜同时进入训练集和测试集：

```text
训练/验证：timestamp < 2026-08-01 00:00:00
测试：     2026-08-01 00:00:00 <= timestamp < 2026-09-01 00:00:00
```

因此最后一个训练日期是2026-07-31，测试日期是2026-08-01至2026-08-31，`2026-09-01 00:00:00` 不属于测试集。

完整8月共有：

```text
31天 × 96个15分钟区间 = 2976个统一时间区间
```



## 4. 四个模型的输入、窗口和输出

### 4.1 LSTM Autoencoder

当前使用11个电力特征：

```text
powerFactor, pRealKw, pRealPositiveKw, pRealNegativeKw,
pReactiveKw, pReactivePositiveKw, pReactiveNegativeKw,
vRMSMin, vRMSMax, iRMSMin, iRMSMax
```

窗口规则：

```text
历史上下文：24小时 = 288个5分钟点
输出步长：  15分钟 = 3个5分钟点
```

24小时窗口只负责最后3个点对应的15分钟区间。模型计算最后3个点的逐点重构误差并输出：

- `max_score`：15分钟内最大误差，表示瞬时异常严重度。
- `mean_score`：15分钟内平均误差，表示异常持续性。
- `score_std`：3个点之间的误差波动。

不再把一个24小时窗口的单一分数平均摊回整个窗口，避免制造异常稀释和24小时拖尾。

主要输出：

```text
runs/lstm_autoencoder/aligned_august/
|-- best_model.pt
|-- scaler.joblib
|-- calibration_scores.npz
|-- metadata.json
|-- training_history.csv
|-- anomaly_windows.csv
|-- scores_15min.csv
|-- meter_summary.csv
`-- *.png
```

### 4.2 LSTM-VAE

默认电力特征为 `pRealKw`、`pReactiveKw`、`powerFactor`、`vRMSMax` 和 `iRMSMax`，还可以加入日序和星期序的周期编码。每个电表通道训练一个独立VAE。

窗口规则与LSTM Autoencoder一致：

```text
历史上下文：24小时 = 288点
输出区间：  最后15分钟 = 3点
滑动步长：  15分钟
```

VAE的KL项继续用于训练正则化，但时间定位使用逐点重构误差，KL值不会被平均分摊到24小时内的每个时间点。

主要输出：

```text
runs/lstm_vae/august/
|-- <series_id>_scores_15min.csv
|-- <series_id>_anomaly_periods.csv
|-- <series_id>_calibration_scores.npz
|-- <series_id>_calibration.json
|-- <series_id>_anomaly_diagram.png
|-- ALL_METERS_scores_15min.csv
|-- ALL_METERS_anomalies.csv
`-- _run_summary.csv
```

Slurm数组任务先生成分片结果，再由 `aggregate_array_results.py` 合并事件、运行状态和15分钟分数。

### 4.3 RPCA

RPCA使用 `pRealKw`、`pReactiveKw`、`vRMSMin`、`vRMSMax`、`iRMSMin` 和 `iRMSMax` 六个特征。每个自然日组织成 `288个5分钟点 × 6个特征`，训练期每日数据用于学习低秩正常子空间，测试数据投影后的残差作为异常分数。RPCA本身不使用24小时滑动窗口。

原生输出：

```text
<series_id>_point_scores.csv       # 5分钟分数
<series_id>_daily_scores.csv       # 日级分数
<series_id>_summary.csv            # 阈值与运行摘要
<series_id>_rpca_august.png
```

统一输出把每3个5分钟残差分数组合为一个15分钟区间，同时保留Max、Mean和Std：

```text
<series_id>_scores_15min.csv
<series_id>_calibration_scores.npz
<series_id>_calibration.json
```

### 4.4 Matrix Profile

Matrix Profile主要分析 `pRealKw` 的形状变化，并结合幅值、变化点、通道类别和部分电流信息判断事件。其内部有效时间尺度是30分钟：

```text
短窗口：2小时  = 4个30分钟点
中窗口：6小时  = 12个30分钟点
日窗口：24小时 = 48个30分钟点
原生步长：30分钟
```

8月任务使用固定7月基线：

```text
baseline：2026-07-01至2026-07-31
detection：2026-08-01至2026-08-31
history_mode：fixed
```

主要输出：

```text
events.csv, candidates.csv, channel_status.csv, day_status.csv
scores_30min.csv, scores_15min.csv, daily_summary.csv
ALL_EVENTS.csv, ALL_SCORES_30MIN.csv, ALL_SCORES_15MIN.csv
incident_summary.csv
```

`scores_30min.csv` 是原生连续输出。为了关联到统一15分钟主表，每个30分钟结果会关联到两个15分钟区间，但不会声称获得新的15分钟信息。兼容输出必须包含：

```text
source_resolution_minutes = 30
aggregation_method = repeat_from_native_30m
data_status = upsampled_from_30m
```

集成模型应使用这些字段识别重复值，避免把同一个30分钟结果当作两次独立观测。

## 5. 冻结校准与数据泄漏控制

四个模型使用训练期ECDF（经验累积分布）校准可比较分数：

```text
训练数据
  -> 生成训练分数
  -> 拟合ECDF
  -> 保存calibration_scores.npz和calibration.json
  -> 测试阶段只读取，不更新
```

禁止使用完整8月测试分数的均值、方差、极值或百分位进行校准，也不能随测试数据到来而静默更新归一化器。概念漂移应单独监控，并通过新的训练和 `calibration_version` 处理。

统一百分位范围为 `0.0` 至 `1.0`：

```text
max_percentile  = Max分数在冻结训练分布中的位置
mean_percentile = Mean分数在冻结训练分布中的位置
```

聚合器会拒绝带有 `rolling_history_not_frozen` 标记的结果。

## 6. 统一15分钟输出格式

每个模型的 `scores_15min.csv` 至少包含：

```text
model, series_id, interval_start, interval_end,
raw_score, max_score, mean_score, score_std,
max_percentile, mean_percentile, threshold, is_anomaly,
coverage_ratio, source_resolution_minutes,
aggregation_method, data_status, calibration_version, available
```

字段含义：

- `interval_start` 只能落在每小时的00、15、30、45分。
- `interval_end` 使用左闭右开的区间语义。
- `raw_score` 是模型原始主分数，当前通常等于 `max_score`。
- `max_score` 表示区间异常严重度；`mean_score` 表示持续性。
- `score_std` 表示区间内波动程度。
- `max_percentile`、`mean_percentile` 只能由训练期冻结ECDF计算。
- `coverage_ratio` 表示有效数据覆盖率。
- `source_resolution_minutes` 表示模型真正提供的信息分辨率。
- `data_status` 表示原生、上采样或其他数据质量状态。
- `available=false` 表示该模型没有可靠结果；缺失不能填成正常零分。

## 7. 对齐与聚合流程

```text
原始每日CSV或压缩包
        |
        v
生成每个电表通道的合并5分钟CSV
        |
        v
四个模型分别训练、冻结校准并推理
        |
        +-- LSTM Autoencoder -> scores_15min.csv
        +-- LSTM-VAE         -> ALL_METERS_scores_15min.csv
        +-- RPCA             -> *_scores_15min.csv
        `-- Matrix Profile   -> ALL_SCORES_15MIN.csv
        |
        v
build_ensemble_features.py 校验
        |
        +-- 检查字段和15分钟时间网格
        +-- 检查ECDF范围与冻结版本
        +-- 检查重复记录
        +-- 检查Matrix Profile分辨率声明
        `-- 保留缺失状态，不使用零填充
        |
        v
生成集成学习长表、宽表、事件表和运行摘要
```

在项目根目录运行：

```powershell
python shared/scripts/build_ensemble_features.py `
  --input runs/lstm_autoencoder/aligned_august `
  --input runs/lstm_vae/august `
  --input runs/rpca/august `
  --input runs/matrix_profile/results/matrix_profile_august_fixed_july/aggregate `
  --output-dir runs/ensemble/august
```

最终生成：

```text
runs/ensemble/august/
|-- all_models_scores_15min.csv     # 四模型统一长表
|-- ensemble_features_15min.csv     # 集成学习宽表
|-- all_models_events.csv           # 统一事件表
`-- run_summary.csv                 # 覆盖、异常与校准摘要
```

`ensemble_features_15min.csv` 以 `series_id + interval_start` 为主键，将四个模型的Max、Mean、百分位、可用状态、数据状态和源分辨率展开为独立特征列。

## 8. 旧结果兼容性

旧运行结果不能直接可靠地进入当前集成流程：

- 旧LSTM Autoencoder只有整窗分数，无法恢复最后3点重构误差。
- 旧LSTM-VAE只保存异常事件，没有完整连续时间分数。
- 旧RPCA有测试期5分钟分数，但缺少训练分数，无法重建冻结ECDF。
- 旧Matrix Profile只有事件和候选事件，没有正常区间连续分数。

因此需要使用当前代码重新运行或重新评分，再执行聚合脚本。不能根据旧测试结果后验计算百分位，否则会引入数据泄漏。

## 9. 路径说明

Python代码和YAML中的数据、输出路径已使用项目内相对路径，并由入口程序根据仓库根目录解析。部分Spartan Slurm脚本仍保留云端部署根路径：

```text
/data/projects/punim1257/Group14
/data/gpfs/projects/punim1257/Group14
```

这些根路径后续应改为统一的可覆盖环境变量；其下的数据、代码和结果目录已经遵循本README中的结构。

## 10. 本地运行 9 月 Matrix Profile

本地入口为 `shared/scripts/run_matrix_profile_local.py`，复用现有多尺度 Matrix Profile、分片汇总及 `build_ensemble_features.py` 的校验、特征和事件逻辑。完整命令、字段解释和缺测处理见 [本地运行说明](docs/matrix_profile_local.md)。

在 `group14reshape` 根目录，用已准备好的本地环境执行：

```powershell
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/run_matrix_profile_local.py --workers 4
```

默认读取 `AugSep_meters/` 中的 **2026-09-01（含）至 2026-10-01（不含）**，从原有 `data/processed/meter_csvs/` 提取 **2026-07-01 至 2026-07-31** 作为冻结基线。保留 5 分钟输入的 12 列格式，不修改原始数据或原有 8 月结果。准备后的输入写入 `data/processed/matrix_profile/september_fixed_july_local/`，结果写入 `runs/matrix_profile/results/september_fixed_july_local/`。

中断后用相同参数追加 `--resume`；参数、源文件大小/修改时间、代码或元数据变化时必须使用新的输出与输入目录。统一结果位于该结果目录的 `aligned/`，仍为长表、特征宽表、事件表和运行摘要四个 CSV。本次只运行 Matrix Profile，另外三个模型的 `available` 为 `False`，不能视为四模型集成完成。
