# Group 14 电力数据异常检测

本项目原有四种独立的电力数据异常检测方法：LSTM Autoencoder、LSTM-VAE、RPCA 和 Multi-scale Matrix Profile。仓库已经按照配置、数据、模型代码、集群任务、运行结果和公共工具重新组织。四个模型保留各自的原生输出，同时生成统一的15分钟输出，供后续集成学习使用。新增LOF已按相同输入与时间键接入，但保留批量拟合方式，作为独立比较输出，暂不加入冻结ECDF的四模型融合。

另保留独立的 Attention LSTM AE 新版本，提供单独的训练、冻结测试重放和新时间段推理入口；不覆盖原LSTM，也不自动加入原四模型融合。

## 1. 目录结构

```text
group14reshape/
|-- configs/                         # 项目与模型配置
|-- data/
|   |-- raw/                         # 原始数据
|   |-- processed/meter_csvs/        # 每个电表通道一个合并CSV
|   `-- manifests/                   # 数据与集群任务清单
|-- docs/                            # 项目文档
|-- legacy/                          # 本地备份，不参与运行/测试，不提交Git
|   |-- lof_import/                  # 上传的LOF仓库原件，仅作备份
|   `-- attention_lstm_import/       # 上传的Attention代码及压缩包，仅作备份
|-- methods/
|   |-- lstm_autoencoder/
|   |-- lstm_vae/
|   |-- rpca/
|   |-- lof/                        # LOF批量检测与独立对齐输出
|   |-- attention_lstm_autoencoder/ # 独立Attention版本及冻结新数据推理
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

## 11. 九月 Matrix Profile 告警减负与人工核验分流

新入口 `shared/scripts/triage_matrix_profile_alerts.py` 对已保存的正式事件做保守分流，不重新训练、不重新计算模型分数，也不删除原事件。配置在 `configs/matrix_profile_alert_triage.json`；目前是试验性的核验工作流，不替换生产检测器或四模型统一输出。

```powershell
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/triage_matrix_profile_alerts.py --workers 4 --output-dir runs/analysis/new_alert_triage
```

脚本读取原始九月结果和准备好的五分钟数据，冻结原始七月参考，再检查已完成的八月原始／注入配对验证。需要先有 `runs/analysis/matrix_profile_existing_injections_20261005/`；若没有，按 [本地运行说明](docs/matrix_profile_local.md) 先运行注入验证。新运行须使用空输出目录，不能覆盖已有结果。

完成的分流结果位于 `runs/analysis/matrix_profile_alert_triage_20261005_v2/`，入口为该目录的 `report.html`：

| 队列 | 原始事件数 | 合并后的核验单元 | 含义 |
| --- | ---: | ---: | --- |
| 优先告警 | 58 | 49 | 有实际功率偏差、持续区段、突变与佐证；仍不是已确认故障 |
| 运行变化待核验 | 1,581 | 1,484 | 未满足全部优先规则，不等于正常，不应直接删除 |
| 数据质量问题 | 289 | 172 | 平线等质量问题，与电力故障分开核验 |

原始 1,928 条事件全部保留，去重后共 1,705 个核验单元。**不是把模型检测结果从 1,928 条变成 49 条，而是优先队列缩小到 49 个单元。** 原来的 255,704 个超过候选阈值的十五分钟区间及其分数没有改变；不能把这些区间直接计为故障。

优先规则使用原始五分钟曲线，参考同类日期的历史同时段范围（允许 ±10 分钟排程差异）。超出历史范围的功率偏差门槛为 `max(0.5 kW, 10% × 历史绝对功率P95)`，需同方向连续至少15分钟、峰值达到门槛两倍，且该连续区段开始时发生超过历史跳变门槛的突变，并有短／中尺度共同高分或电流超历史范围佐证。只用真实满足条件的核心区段作为优先时间，不把原长事件或整窗全部涂成告警。电流佐证与模型高分都不是故障真值。

主要文件：

- `priority_alert_units.csv`：建议首先人工查看的49个核验单元。
- `priority_intervals.csv`：真实五分钟证据核心及其起止时间、方向、幅度和佐证；不是新的模型分数表。
- `operating_change_review_units.csv`、`data_quality_units.csv`：另外两个核验队列。
- `events_all_classified.csv`：完整事件台账，原ID、时间、分数、置信标签和证据均保留，追加队列、降级原因和核心区段。
- `event_memberships.csv`：原事件到核验单元的一对一归属关系；一个单元可以包含多个原事件。
- `injection_triage_comparison.csv`、`injection_downgraded_runs.csv`：减少优先告警后的已知修改覆盖与降级清单。
- `triage_summary.json`：固定规则、计数、输入／实现哈希和限制。

需要特别注意两项验证结果：

- 49个优先单元中34个来自太阳能通道；天气、云遮、季节和设备状态可能解释这些变化，目前没有天气真值，不能直接判故障。
- 这套严格规则在已有注入数据中生成 **0个优先事件**。许多修改表现为平线，仍由数据质量队列覆盖。在原路径上，全部原事件保留的改变点覆盖率为 `4,074 / 5,232 = 77.87%`；只看优先核心＋质量队列为 `4,036 / 5,232 = 77.14%`，另有38个改变点、4个连续改变区段只在待核验队列中保留。补漏路径受影响更大，详情见报告。

因此不能仅运行优先队列就丢掉其他记录，也不能宣称真实误报率降低。5个不可评分通道、1,060个改变点仍单独记录为不可评分，不能当正常。当前结果适合安排小规模人工核验，尚不适合以这套规则完全取代异常检测。

## 12. 与既有数据工具和统一输入输出衔接（2026-10-12）

新的流程入口是 `shared/scripts/align_dataset_workflow.py`，配置为 `configs/datasets/september_2026.json`。这是数据和结果的适配入口，**不重新训练、推理或校准模型**。八月的 `common.yaml`、各模型配置和旧结果保留；九月用独立的数据集配置，不把月份改成全项目的隐式默认值。

现有工具的分工如下：

| 工具 | 本次如何复用 |
| --- | --- |
| `shared/data_tools/build_meter_csvs.py` | 复用 `ArchiveReader` 读取目录／ZIP／TAR；历史数据首次合并仍用原来的多输入工具 |
| `shared/data_tools/meter_input.py` | 从原九月入口提取严格的12列、5分钟校验和基线／测试预处理；旧入口也调用同一实现 |
| `shared/scripts/build_ensemble_features.py` | 复用模型分数校验、宽表、候选区段、运行摘要与分数文件发现逻辑 |
| `shared/scripts/build_manual_review_set.py` | 复用模型时间线、原始曲线、人工标签表的提取与输出函数；新入口负责按三个核验队列抽样 |
| `shared/scripts/plot_manual_review_set.py` | 继续读取原有人工核验CSV格式；新增队列名称并明确只有一个模型可用时的显示口径 |

### 12.1 数据集角色与共享输入

```text
AugSep_meters/                          原始交付数据，只读，不移动旧目录
data/processed/meter_csvs/             原有完整历史数据，继续保留原训练语料
data/processed/matrix_profile/
  september_fixed_july_local/          已完成九月MP运行的原输入缓存，只读
data/processed/datasets/september_2026/
  meter_csvs/                         新的共享输入，每通道一个12列5分钟CSV
    _manifest.csv                     每通道基线、上下文、测试行数及缺失情况
    dataset_metadata.json             角色、路径、哈希、来源和数据集签名
data/injected_meters/                  人工修改数据，只用于独立配对评估
```

共享输入保留原七月／九月缓存的数值，另从原始交付数据读入8月31日上下文。没有插值、删除重复点、补零或重算正向／反向功率；重复时间戳、错误字段、错误时间网格会报错。缺失上下文和缺测日期会写入清单，不伪造完整观测。

| 角色 | 时间范围（悉尼时区，左闭右开） | 用途 |
| --- | --- | --- |
| MP冻结参考 | `[2026-07-01, 2026-08-01)` | 与已完成九月MP运行一致，不用测试分布再校准 |
| 神经网络上下文 | `[2026-08-31, 2026-09-01)` | 为24小时窗口提供紧邻测试前的观测，不进入训练／ECDF，也不输出为九月分数 |
| 九月测试 | `[2026-09-01, 2026-10-01)` | 模型结果和集成输出的目标时间范围 |
| 人工注入评估 | 既有八月原始／修改配对文件 | 独立评估来源，不混入生产数据或基线 |

这个派生数据集只有七月参考、测试前上下文和九月数据，**不能代替其他模型原来的完整训练语料**。继续使用已训练权重时，缩放器、阈值和ECDF也应读取原冻结版本；如需重新训练，应明确选择原历史数据和训练截止日期。

神经网络仍为24小时（288个5分钟点），每15分钟滑动（3点）；RPCA仍为每日288点；MP仍为2／6／24小时多尺度、30分钟原生步长。共享输入准备好不代表另外三个模型已经完成九月推理。尤其当前LSTM AE代码要求 `train_end == test_start`，且上下文来自基线尾部；七月训练截止与九月测试分离时，还需适配日期间隔与真正紧邻测试期的上下文。此入口不会偷偷修改其训练截止日期或自动重训。

当前RPCA生产入口的校准版本也根据 `test_start` 生成；直接改成九月测试，可能把训练截止推到九月。新适配入口要求其他三模型提供 `train_pre_2026-08-01_*` 的冻结版本声明，MP要求原七月固定版本；不符合配置的结果会拒绝混入。后续推理适配应把冻结训练截止和测试日期明确分离，而不是修改训练截止来绕过运行限制。

### 12.2 输出分层与统一键

已生成的对齐结果位于 `runs/ensemble/september_2026_aligned/`：

```text
all_models_scores_15min.csv            保留原分数、ECDF、is_anomaly和分辨率声明
ensemble_features_15min.csv            原四模型宽表，追加MP核验标记
all_models_events.csv                  原规则生成的连续候选阈值区段，不是确认故障
run_summary.csv                        原覆盖、候选阈值统计
model_readiness.csv                    明确哪些模型在这个数据集实际有结果
workflow_metadata.json                 来源、时间、窗口、输入哈希与输出口径
dataset_profile.json                   本次执行的配置快照
review/
  model_events.csv                    原1,928条正式事件＋统一键／分数字段＋核验分类
  review_units.csv                    三类核验单元，保留全部成员ID
  event_memberships.csv               原事件与核验单元的归属对应
  interval_flags_15min.csv             核验区间对齐到15分钟的独立附表
manual_review/
  review_cases.csv                    每类5个案例，共15个；人工标签留空
  review_timeline_15min.csv            与既有核验工具兼容的四模型时间线
  review_raw_5min.csv                  每个案例前后24小时的原始曲线
  review_manifest.json                抽样方式、来源与限制
```

分数和核验区间按 `model + series_id + interval_start` 对齐；正式事件按 `model + event_id` 回溯，同时提供 `series_id + start + end`。一个核验单元可以包含多个通道，保留 `series_ids` 和成员ID，不伪装成单一电表事件。

核验附表提供独立的 `priority_alert_overlap`、`operating_change_review_overlap`、`data_quality_overlap`。十五分钟区间可能同时覆盖不同类别的五分钟片段，因此三个字段不强制互斥。优先标记只覆盖真实五分钟核心，`priority_core_minutes` 为该十五分钟内实际覆盖的分钟数，不复制三十分钟分数或扩展到原长事件。

模型分数不可用时，核验标记留空、`triage_available=False`、`review_state=unavailable_model_score`；有分数但没有正式事件时为 `no_source_event`，**不等于正常标签**。`source_event_noncore_context` 表示原长事件存在，但该区间不是优先核心。原始 `is_anomaly` 不被核验规则覆盖，原模型置信标签也不改成故障概率。

附表布尔字段在CSV中为 `True / False / 空值`。用Pandas读取时，应指定可空布尔类型（`dtype="boolean"`），或复用新入口中的 `read_review_flags()`；不要直接对字符串使用 `bool()` 或 `astype(bool)`，否则字符串 `"False"` 也可能被当成真。`priority_core_minutes` 的空值同样表示不可用，不补成零。

本次只有MP的九月结果：1,267,200个标准区间、684,672个可评分区间、255,704个候选高分区间，与旧运行一致。其他三种方法没有九月结果，宽表的 `available=False`、分数缺失。三类核验仍为49／1,484／172个单元，没有把这次对齐解释成误报率改善。新增案例也不声称四模型一致，人工状态为 `unreviewed`。

### 12.3 后续如何调用

在 `group14reshape` 根目录执行：

```powershell
# 准备或校验共享输入，不运行模型
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/align_dataset_workflow.py --stage prepare

# 复用已保存的模型／核验结果，导出到新目录并抽取核验样本
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/align_dataset_workflow.py --stage all --output-dir runs/ensemble/september_2026_aligned_v2

# 已导出结果但尚未抽样时，只调用既有核验函数生成样本
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/align_dataset_workflow.py --stage review --output-dir runs/ensemble/september_2026_aligned_v2

# 可选：用原画图工具读取新核验格式，不重新评分
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/plot_manual_review_set.py --review-dir runs/ensemble/september_2026_aligned/manual_review
```

`--stage all` 已包含抽样；完成后不应再对同一目录运行 `--stage review`。非空结果目录／人工核验目录不会被覆盖，避免覆盖后续填写的标签。已有共享数据的哈希不匹配时会拒绝复用，不会强制覆盖。路径均由新入口相对于仓库根解析，可用于本地或Spartan；历史模型元数据中旧绝对路径保留作原运行记录，不拿它们定位新数据。

其他三个模型完成九月推理后，把各自的标准分数CSV、分片列表或目录填入数据集配置的 `models.<name>.score_input`，再导出到新目录。会复用原聚合工具的文件发现逻辑，但要求输入按 `series_id` 排序；重复键、错误校准声明或完全不含目标月份的旧文件会报错。**不能把八月分数直接当作九月结果，也不能把三类核验CSV当成第五个机器学习模型。**

此次只修改 `group14reshape`；原始交付、原历史数据、旧八月／九月生产结果及既有注入评估没有覆盖，尚未提交或推送GitHub。

## 13. LOF接入：统一输入输出，保留原批量算法

上传的LOF仓库原件已从顶层 `LOF/` 归档到 `legacy/lof_import/MeterAnomalyDetection-main/`，原文件内容保持不变，顶层不再保留重复目录。它也包含旧VAE、RPCA、Matrix Profile和融合工具，这些重复代码不覆盖当前模型。新增实现放在 `methods/lof/src/lof_batch.py`，测试放在 `methods/lof/tests/`，配置集中于 `configs/lof.json`，统一运行入口为 `shared/scripts/run_lof_local.py`，结果单独进入 `runs/lof/<run_name>/`。目录用途、完整字段和命令见 `methods/lof/README.md`。

`legacy/` 现在仅作本地备份，整个目录由 `.gitignore` 忽略。运行代码和自动测试都不读取、导入或依赖其中的文件；没有这个目录也能运行和测试。LOF通过独立的参考计算和固定特征／参数断言验证算法一致性，不再检查备份原件。新元数据记录当前实现位置和代码哈希；旧运行元数据保留当时的原路径及哈希，不回写历史结果。`methods/lof/` 是当前执行代码，不能当作旧原件删除。

保持原算法：每通道7个特征，`StandardScaler` 标准化，`LocalOutlierFactor(n_neighbors=20, contamination=0.005)`，使用 `fit_predict` 原生标签和 `-negative_outlier_factor_` 原生得分。没有加入滑动窗口、`novelty=True` 或七月训练模型。原脚本缺失括号、月份路径写死及绘图依赖问题由新入口解决，不修改上传原件。

输入复用第12节的12列5分钟共享CSV。默认九月配置只取 `[2026-09-01, 2026-10-01)` 拟合与检测，七月数据和8月31日上下文不进入LOF。缺少任一所用特征的行不评分但保留为不可用；有效目标行不足21个的通道不拟合。重复时间戳、错列、非5分钟网格和无限值通过原共享工具拒绝，不插值、补零或重算功率。

### 13.1 输出与时间窗口

- `channels/<series_id>/point_scores_5min.csv`：保留全量已观测目标行、原生得分和 `-1/1` 标签，以及不可用状态。
- `ALL_ANOMALIES_5MIN.csv`：全部原生异常点，单点尖峰仍保留。
- `ALL_SCORES_15MIN.csv`：完整15分钟网格，保存Max、Mean和总体Std；`raw_score= max_score`，每区间最多3个5分钟点。任一可用原生点被标记则区间被标记，不用新的聚合阈值重判原标签。
- `ALL_EVENTS_5MIN.csv`：连续5分钟异常点合并的原生区段，不跨缺失点或正常点连接。
- `ALL_EVENTS_NATIVE_MIN15MIN.csv`：原生连续异常至少15分钟的子集，短异常没有从其他输出中删除。
- `ALL_EVENTS_15MIN.csv`：用于时间对比的连续异常占用区间。**单个5分钟异常占据一个15分钟区间，不等于连续异常15分钟。**
- `run_summary.csv`、`run_metadata.json`：覆盖率、点／区间／事件数、原生阈值、输入与代码哈希、运行版本及批量拟合范围。

键与现有项目相同：`model=lof + series_id + interval_start`，事件提供 `model + event_id + series_id + start + end`。`coverage_ratio=valid_point_count/3`；`classification_complete` 表示三个点均有分数。部分覆盖区间的 `False` 只表示已评分点未被标记，不能说明缺失点正常。整段不可用时分数和 `is_anomaly` 留空，CSV导入应指定可空布尔类型。

### 13.2 为什么暂不加入原四模型融合

这是整批目标数据参与标准化和邻域拟合的回顾性离群检测，不是冻结模型推理，也不能当作无测试信息参与的训练／测试评估。`contamination=0.005` 是批量阈值参数，不是已验证故障比例。

因此百分位字段保持为空，明确记录 `calibration_version=lof_batch_not_frozen_v1`、`ensemble_eligible=False`。可以按统一时间键独立比较原生标签与四模型结果，但不加入冻结ECDF的模型清单、打分融合或投票。原四模型的验证器仍会拒绝这个未冻结输出。上传文件夹中的测试结果百分位排名工具不作为训练ECDF调用。未来若改为七月冻结训练／九月推理，需要另行确认算法模式变更。

### 13.3 本地调用

```powershell
# 先检查一个通道的完整目标月份，不把月份拆成小批拟合
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/run_lof_local.py --max-channels 1 --output-dir runs/lof/september_2026_smoke

# 确认后再显式运行全部通道；本次接入不会自动执行全量九月数据
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/run_lof_local.py --output-dir runs/lof/september_2026_batch
```

可用 `--series-id` 选择具体通道，`--dataset-profile` 切换独立数据集配置，`--plots` 额外生成五张英文统计图。相对路径统一从仓库根解析，非空结果目录拒绝覆盖，原始数据、旧结果和现有四模型功能保持不变。LOF依赖可复用当前本地Matrix Profile环境，不改装已有环境；其他机器可根据 `methods/lof/requirements.txt` 配置独立环境。

## 14. 独立Attention LSTM AE：代码结构与新时间段推理

队友代码的实际网络是LSTM编码器＋时间注意力＋GELU隐层投影＋LSTM解码器，已经独立接入 `methods/attention_lstm_autoencoder/`。原 `methods/lstm_autoencoder/`、其他模型、旧结果和默认四模型集合保持不变。旧LSTM权重不能直接加载到Attention结构；若已有与此结构匹配的训练包，可直接复用，不必重新训练。

上传的 `lstmaecodes/` 原件和 `codes.zip` 已完整归档到 `legacy/attention_lstm_import/`，只作本地备份，不参与运行或测试；当前执行代码全部位于新方法目录。原件归档前后逐文件哈希一致，没有删除原始代码。

| 位置 | 用途 |
| --- | --- |
| `methods/attention_lstm_autoencoder/src/` | 保留队友的网络、分片训练、恢复、独立标定和冻结测试重放；输出模型名与旧LSTM分开 |
| `methods/attention_lstm_autoencoder/project.py` | 相对路径、训练输入视图、窗口约束、数值C通道排序映射 |
| `methods/attention_lstm_autoencoder/inference.py` | 只读取冻结模型／缩放器／ECDF，检测新的时间段 |
| `methods/attention_lstm_autoencoder/train_by_category.py` | 按类别分组训练，每类独立输入视图、缓存和结果目录 |
| `methods/attention_lstm_autoencoder/jobs/` | 独立Slurm推理模板，不提交或更改现有作业 |
| `configs/attention_lstm_autoencoder.yaml` | 新训练实验的完整候选配置，不代表队友真实训练参数 |
| `configs/attention_lstm_autoencoder_inference.json` | 九月共享数据配置、冻结类别包和独立结果路径 |
| `shared/scripts/run_attention_lstm_autoencoder.py` | 项目级统一入口，可从任意工作目录调用 |

### 14.1 训练与推理必须分开

原生 `prepare/train/predict` 保留Linux文件锁、信号和分片恢复流程，仍要求 `train_end == test_start`，测试最初没有自身24小时历史时保留不可用区间。`predict` 只重放原冻结测试集，不能通过修改日期把七月训练模型直接挪到九月。训练入口创建排除 `_manifest.csv` 的只读链接视图，不移动或重写原数据，然后保留队友原生的重采样、重复处理及有界插值规则。

新增 `infer` 不调用训练、缩放器拟合或校准。默认读取第12节的共享数据，将七月冻结截止 `2026-08-01` 与九月检测 `[2026-09-01, 2026-10-01)` 分离；使用紧邻测试前的8月31日观测作为上下文，不拿七月尾部替代。窗口288个5分钟点、步长3点，只对最后3点计算Max、Mean和总体Std。首区间实际需要285个历史点（23小时45分钟），完整一天上下文足以提供这些点；不足时留空，不补零、不平摊24小时分数。

推理延续训练包声明的有界内部插值，保留原始观测覆盖率，不能把插值结果全写成真实观测。这是离线插值，不宣称无未来观测的严格在线检测。输入使用原共享12列格式，重复／错网格／无限值通过共享工具拒绝，不重算正向或反向功率。

### 14.2 队友需要提供的训练包

默认分别将类别1／2／4／5的真实产物放到：

```text
runs/attention_lstm_autoencoder/models/category_<id>/
  best_model.pt
  config.json
  scaler.json
  metadata.json
  calibration/max_sorted.npy
  calibration/mean_sorted.npy
```

元数据须为这份代码的完成训练产物，提供模型／缩放器／校准数组哈希、训练划分、冻结绑定、阈值、采样数量与近似标记。入口验证特征顺序、模型／评分参数、训练截止和文件完整性；权重采用安全加载及严格结构匹配，不用旧权重部分加载后留下随机注意力参数。原训练方法使用训练前的独立标定集，与旧LSTM训练分数ECDF不同，保留 `held_out_calibration` 声明，不伪装成相同标定方法。

当前预检查能识别440个共享通道，其中404个属于默认四个类别，36个类别未配置。真实权重包及当前环境PyTorch尚未提供，因此不能执行真实神经网络推理。支持类别缺包时在生成结果之前停止；未配置类别的状态和分数保持不可用，不视为正常。之前的 `lstm_results_new` 只保留原结果，不因字段相同就贴上Attention模型标签，仍需对应权重／配置绑定证明。

### 14.3 调用与输出

```powershell
# 只做预检查；不拟合、不创建结果，缺包／PyTorch时退出码为2
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/run_attention_lstm_autoencoder.py validate
```

拿到真实训练包并激活兼容的独立PyTorch环境后：

```bash
python shared/scripts/run_attention_lstm_autoencoder.py validate
python shared/scripts/run_attention_lstm_autoencoder.py infer --max-channels 1 --output-dir runs/attention_lstm_autoencoder/results/september_smoke
python shared/scripts/run_attention_lstm_autoencoder.py infer
```

默认结果在 `runs/attention_lstm_autoencoder/results/september_2026_frozen/`，提供逐通道分数、`ALL_SCORES_15MIN.csv`、`ALL_EVENTS.csv`、`run_summary.csv`、`channel_status.csv`、模型／输入／输出哈希和配置快照。共用原有分数／事件字段与时间键，但 `model=attention_lstm_autoencoder`，事件ID包含数据集、版本、电表和时间边界。不可用异常标记留空；类别未配置时连覆盖率也保留未知，不伪装缺测。校准版本保留训练包原哈希，训练截止、标定来源及近似状态另列。

训练需要Spartan/Linux/WSL；新增预检查不依赖PyTorch，新推理入口在有兼容PyTorch时可跨平台。原分片训练或类别训练只能明确调用，不会因为缺权重自动启动。配置、缓存、权重及输出均与旧模型分开，非空推理结果目录拒绝覆盖，运行／测试不依赖 `legacy/`。当前仍设置 `ensemble_eligible=False`，避免未经评估把新旧两个LSTM变体当作两个独立投票者。

专项测试使用合成输入／训练包和明确标注的NumPy评分桩检验适配流程；真实PyTorch前向／反向测试在当前缺依赖时明确跳过，不声称已经验证真实训练或模型效果。本次未安装依赖、未运行训练、未运行全量推理，也未提交或推送新的修改。
