# Local September Matrix Profile run / 本地九月运行

## 运行环境与命令

以下命令在 `group14reshape` 根目录的 PowerShell 中执行。入口中的相对路径按该仓库根目录解析，不依赖终端当前位置。

当前机器已有 `.venv/matrix_profile` 环境。新机器可创建独立环境：

```powershell
python -m venv .venv/matrix_profile
& ./.venv/matrix_profile/Scripts/python.exe -m pip install -r methods/matrix_profile/jobs/requirements-multiscale.txt tqdm
```

完整运行（默认值已经是九月）：

```powershell
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/run_matrix_profile_local.py --workers 4
```

断点续跑同一任务：

```powershell
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/run_matrix_profile_local.py --workers 4 --resume
```

少量通道试跑必须使用独立目录，避免与全量任务混用：

```powershell
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/run_matrix_profile_local.py `
  --channel-keys DD03710131709_0 DD13710148400_0 DD03710131908_0 `
  --workers 3 --output-dir runs/matrix_profile/results/september_local_smoke
```

也可直接使用 ZIP，不需要先解压；更换输入来源须使用新的结果目录：

```powershell
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/run_matrix_profile_local.py `
  --data AugSep_meters.zip --start 2026-09-01 --end 2026-10-01 `
  --baseline-start 2026-07-01 --baseline-end 2026-07-31 `
  --workers 4 --output-dir runs/matrix_profile/results/september_local_zip
```

## 输入、时间与基线

- 目标输入：`AugSep_meters/`，按日期文件夹选择九月 CSV，不读取其中的八月或十月目标数据。
- 历史输入：`data/processed/meter_csvs/` 及 `_manifest.csv`，只提取七月，原文件不追加、不覆盖。
- 通道集合：历史清单与九月实际 CSV 通道的并集。本数据为 440 个通道，其中 398 个九月有 CSV、42 个整月没有；398 个中 371 个完整、27 个部分缺失。只有 `.miss` 而从未有 CSV/历史记录的未知通道不新增。
- 准备输入：`data/processed/matrix_profile/<输出目录名>/`，每通道一个保留原 12 列的 CSV，以及覆盖率 `_manifest.csv`。可用 `--input-dir` 指定新目录。
- 时间单位：原始采样 5 分钟；统一时区 `Australia/Sydney`。九月检测范围为 `[2026-09-01, 2026-10-01)`，`--end` 不含当天；基线结束 `--baseline-end` 含当天。
- 算法：沿用现有多尺度实现，不改成神经网络的 96 点窗口。5 分钟功率先聚合为 30 分钟；2 小时、6 小时、24 小时子序列分别对应 4、12、48 点，原生分数输出在 30 分钟网格。工作日/周末按七月参考日分组，太阳能沿用现有日照筛选逻辑。
- 不需要神经网络权重。Matrix Profile 使用基线子序列和基线留一日校准；七月固定参考分布不会随九月更新。冻结标定缓存仅在同一通道内部复用。
- 基线须至少有 24 个有效日；有效日需至少 95% 的 288 个点。原有低活动、常量、元数据及覆盖率准入规则继续生效，因此“有九月文件”不代表该通道一定能评分。

输入 schema 仍为：

```text
timestamp,powerFactor,pRealKw,pRealPositiveKw,pRealNegativeKw,pReactiveKw,pReactivePositiveKw,pReactiveNegativeKw,vRMSMin,vRMSMax,iRMSMin,iRMSMax
```

此实现只把 `pRealKw` 和可选的 `iRMSMax` 送入现有检测器，其余列原样保留在准备输入中。重复时间戳、日期目录错配和 schema 不匹配会报错，不静默填成正常数据。

## 输出与统一格式

结果根目录默认为 `runs/matrix_profile/results/september_fixed_july_local/`：

| 文件 / 目录 | 内容 |
| --- | --- |
| `run_metadata.json` | 输入路径、基线/检测范围、代码指纹、软件版本、运行完成状态和数量 |
| `shards/part-*/` | 每通道检测结果和完成标记，用于恢复未完成任务 |
| `ALL_CHANNEL_STATUS.csv` | 每通道 detected / no_event / missing_target / low_activity 等状态和原因 |
| `ALL_DAY_STATUS.csv` | 每通道每日可用状态，部分缺测不能算 no_event |
| `ALL_EVENTS.csv` | 原生检测器正式事件，保留置信级别、证据和共同事件标记 |
| `ALL_CANDIDATES.csv` | 原生检测器候选事件，不能直接当作已确认故障 |
| `daily_summary.csv`, `incident_summary.csv` | 正式事件的每日分类统计和共同事件汇总 |
| `ALL_SCORES_30MIN.csv` | 全通道、全检测期原生网格，包括显式不可用行 |
| `ALL_SCORES_15MIN.csv` | 统一网格，每个 30 分钟分数重复到两个 15 分钟区间 |
| `aligned/all_models_scores_15min.csv` | 相同统一长表，模型字段为 matrix_profile |
| `aligned/ensemble_features_15min.csv` | 四模型接口宽表，另三个模型不可用，不产生伪造分数 |
| `aligned/all_models_events.csv` | 用统一分数标记连接连续区间生成的事件 |
| `aligned/run_summary.csv` | 每通道区间数、可用数量和异常标记数 |

统一分数保留 `model, series_id, interval_start, interval_end, raw_score, max_score, mean_score, score_std, max_percentile, mean_percentile, threshold, is_anomaly, valid_point_count, expected_point_count, coverage_ratio, source_resolution_minutes, aggregation_method, data_status, calibration_version, available` 等既有字段。

本模型 `raw_score/max_score` 是现有短窗、长窗和幅值**百分位分量的最大值**，不是电功率，也不是原始 MP 距离；`mean_score` 是这些有限分量的平均值，不是 15 分钟用电功率均值。`expected_point_count=3` 指三个评分分量，不是原始采样点数。每日原始距离另存 `daily_raw_score`，子序列最近邻距离保留在事件证据中。

15 分钟表严格保留 `source_resolution_minutes=30`、`aggregation_method=repeat_from_native_30m`、`data_status=upsampled_from_30m`，不声称新增 15 分钟分析分辨率。不可用行的 `available=False`、分数为 NaN、`is_anomaly=False`，另加 `availability_reason` 区分 missing_target、constant_signal 等。分析时必须先筛选 `available`，不能把不可用行的 False 理解为正常判断。

完整 440 通道九月网格为 **633,600 行原生 30 分钟表、1,267,200 行统一 15 分钟表**（不含表头），超过 Excel 单表行数上限。请用 Python 分析或按通道读取，不要在 Excel 中保存截断的大表。

`ALL_EVENTS.csv` 与 `aligned/all_models_events.csv` 不能混为一谈：前者包含原生多条件事件判定及质量事件；后者由统一分数超过 0.995 的区间合并，可能更多、更长。两者都是待核验的检测结果，不是已确认的电气故障。

## 此次适配与复现

1. 复用原有检测器，增加 Windows 本地多进程、分通道进度及断点续跑入口。
2. 固定基线留一日校准原本每个目标日重复计算，现改为同通道、同参考日组的缓存；缓存与未缓存分数通过等价测试。
3. 修复幅值百分位的近似相等值同时计入 lower/equal 导致超过 100% 的错误。此次固定校准版本为 `fixed_2026-07-01_2026-07-31_v2`；旧 v1 受影响行不能声称逐位一致，严谨的八九月比较应使用 v2 重跑八月。
4. 修复固定模式的缺失日状态，补全未评分通道的时间网格。原始数据没有补成虚构的零值。
5. 复用共享统一输出构建函数，并逐通道处理，避免百万行长表与宽表同时占用内存。
6. 全量运行发现两个不同粗候选经边界细化后落到相同通道、相同时间范围，生成相同事件 ID。汇总现在只合并身份、事件类型和置信级别都一致的记录，保留双方原始记录在 `evidence.refined_candidate_records`，严重度分量取最大值，通道事件计数同步修正；真正冲突的 ID 仍然报错。原始分片不删除、不改写。

本次 440 通道检测已先完成，随后才修复第 6 项汇总问题，因此检测签名与后处理代码签名单独记录。此目录需要重新汇总时使用下面的命令，不用重新评分，也不使用要求代码签名完全相同的 `--resume`：

```powershell
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/finalize_matrix_profile_local.py `
  --run-dir runs/matrix_profile/results/september_fixed_july_local
```

该入口要求全部通道分片已完成且原始签名一致，只重新生成派生结果，不允许把未完成任务当作完成。之后用当前代码新开的运行，正常使用 `run_matrix_profile_local.py` 和 `--resume` 即可。

续跑检查代码 SHA256、元数据内容、源文件大小/修改时间、日期范围、通道选择和输入目录。它不是原始 CSV 全内容哈希；若原始内容改变却刻意保留大小/时间，应使用新目录重跑。算法或参数改变也必须新建运行目录，不拼接不同版本结果。

断点续跑针对已完成输入准备后的检测分片。若首次运行在输入准备过程中中断、尚未生成 `input_signature.json`，请使用新的 `--output-dir`（或另加新的 `--input-dir`），保留原目录供核查，不覆盖无法验证的半成品。不同 `--output-dir` 如果目录名相同，会默认共享同名准备输入路径；这种情况请显式指定不同 `--input-dir`。

测试命令：

```powershell
& ./.venv/matrix_profile/Scripts/python.exe -m pytest methods/matrix_profile/tests -q
```

本次只新增九月 Matrix Profile 分析，没有运行 LSTM AE、LSTM-VAE 或 RPCA。不要把历史八月的其他模型分数混入九月集成表。

## 本次本地结果

440 个通道均完成检测流程，没有 processing_error 或 calibration_failed。242 个通道获得有效评分（239 个有正式事件、3 个无正式事件），其余按准入/缺测规则跳过：常量 69、低活动 63、整通道状态 missing_target 39、历史不足 27。输入整月缺失实际为 42 个，其中 3 个同时历史不足，因此通道级状态优先显示 insufficient_history；每日状态明确记录缺测，不漏计。

正式事件 1,928 个，候选事件 415 个；1 对相同细化区间的重复正式记录已合并，原始分片及两条证据保留。统一 15 分钟长表 1,267,200 行，其中有效 684,672 行、不可用 582,528 行。有效区间中 255,704 行超过候选阈值，约 37.35%。这是阈值触发比例，不是实际故障率，也不是正式事件数量；七月与九月的运行模式变化和窗口分数映射可能影响结果，尚需进一步核验。

`aligned/all_models_events.csv` 的阈值区间合并事件为 32,026 个，采用更宽的候选标记口径，不等于 `ALL_EVENTS.csv` 的 1,928 个原生正式事件。

验证包括 9 项自动化测试、文件夹与 ZIP 输入分数一致、试跑断点续跑分数文件哈希一致，以及全量输出的时间范围、每通道 2,880 行、唯一主键、百分位范围、缺测空分数、30 分钟复制标记和其他模型不可用状态检查。

## Isolated baseline sensitivity pilot

The exploratory entry point is `shared/scripts/experiment_matrix_profile_baselines.py`.
It reuses the existing multiscale scoring and event functions without editing the
production detector. The 2/6/24-hour windows, 0.995 threshold, weekday/weekend
reference selection, admission rules, and marked 30-to-15-minute repetition remain
unchanged. All historical references end before September.

```powershell
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/experiment_matrix_profile_baselines.py --workers 4
```

Five variants compare raw July, cleaned July, raw July–August, cleaned July–August,
and cleaned August. The pilot intentionally includes high-flag and middle-ranked
channels from four equipment categories, historical freeze cases, and previous
review cases. It is not a random fleet sample or a verified accuracy benchmark.

The conservative quality rule requires power, both voltage columns, and both
current columns to remain unchanged over contiguous five-minute readings for at
least two hours. Near-zero power is excluded. These are **suspected** multi-sensor
freezes, not confirmed communication faults and not a reproduction of the DOCX
report's unpublished algorithm. Historical rows are masked, never replaced by
zero, while target observations remain intact. The original valid-day and
minimum-history requirements are retained. Reduced coverage is reported
separately, not counted as an improvement.

The report compares threshold flags on the same available channel/time keys
across all five variants. Original raw-July scores and formal event IDs must be
reproduced, and SHA256 hashes of the production score/event files must remain
unchanged. Native and canonical scores, events, candidates, and coverage statuses
are saved in a separate experiment directory; experimental variants must not be
combined as independent models in the ensemble.

Controlled probes add a 15-minute spike, a six-hour level shift, or a four-hour
multi-sensor freeze to in-memory copies of real September days. Each variant gets
the same probe, with unchanged values outside the known injection interval.
These probes are not the existing `injected_meters` dataset. Background flags,
newly flagged intervals, and out-of-interval spread are recorded separately;
event overlap alone is not a recall estimate.

Default outputs are under
`runs/analysis/matrix_profile_baseline_pilot_20261005_v2/`, including `report.html`,
`aggregate_comparison.csv`, `trial_summary.csv`, `baseline_quality_audit.csv`,
`probes.csv`, and per-variant 30-minute/15-minute CSVs. Existing output directories
are never overwritten. To repeat an experiment, provide a fresh `--output-dir`.

### Completed pilot findings (2026-10-05)

The pilot completed 11 channels across five reference variants. Comparisons use
11,808 common native 30-minute intervals across nine channels, rather than
counting lost scoring coverage as an improvement. On those intervals, the raw
July flag rate was 60.81%, cleaned July 61.45%, raw July–August 33.43%, cleaned
July–August 35.58%, and cleaned August 39.09%. These rates describe a purposively
selected sample, not the full September fleet.

Cleaned July–August reduced threshold flags by 41.49% and common-coverage formal
events from 93 to 54. However, among ten identical eligible power-probe cases per
type, formal event overlap fell from four to two for 15-minute spikes and from
two to one for six-hour level shifts. Score-level overlap remained present in
all ten cases. Background events can already overlap a probe, so these are
sensitivity warnings, not measured fault recall or false-positive reductions.
The production reference should not be replaced on the basis of alarm counts.

Only nine of the 242 originally scored channels met the conservative July
freeze rule, and cleaning July alone did not reduce threshold flags. Two pilot
channels became unscorable under some alternative references due to the
`constant_signal` admission rule; missing scores must not be treated as normal.
The next checks should target amplitude/change-point event candidate paths,
sparse on/off load admission, and validation against known injected intervals.

Verification passed 19 tests, reproduced the original July-reference scores
and formal event IDs, checked all 55 variant-channel outputs and their explicit
30-to-15-minute repetition, and confirmed unchanged production score/event and
detector-code SHA256 hashes. The pre-experiment backup was pushed to
`origin/reshape1` as `3ae351b`; subsequent exploratory source changes remain local.

## Opt-in amplitude event experiment and existing-injection validation

The supplemental implementation is
`methods/matrix_profile/src/anomalies/multiscale/magnitude_events.py`.
It is deliberately **not enabled in the production detector**. It adds genuine
five-minute amplitude evidence to the existing 2/6/24-hour approach, without
altering scores, ECDF calibration, admission rules, or the original formal
events. Historical envelopes allow +/-10-minute schedule jitter. A minimum
absolute/relative power effect and current/jump/change-point evidence guard
against admitting every saturated percentile. The stricter
`transition_guarded` variant adds an onset/effect requirement. Suspected
multi-sensor freezes, quality intervals, and missing points are excluded.

```powershell
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/experiment_matrix_profile_event_policy.py --workers 4 --output-dir runs/analysis/new_event_policy_trial
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/validate_matrix_profile_injections.py --workers 4 --output-dir runs/analysis/new_injection_validation
```

Completed outputs are in `runs/analysis/matrix_profile_event_policy_20261005/`
and `runs/analysis/matrix_profile_existing_injections_20261005/` respectively.
Each contains `report.html` and machine-readable comparisons. The event policy
runner also supports `--finalize-only` when all channel checkpoints exist:
persisted original scores and legacy event IDs are revalidated before reporting.
New runs require fresh output directories. Comparisons are alternatives of one
model, not independent ensemble members.

On the two-reference September pilot's common coverage (13,248 native intervals,
ten channels), raw-July legacy produced 127 formal events and cleaned
July–August with the transition guard produced 102. This is a 19.69% count
reduction, not a demonstrated false-positive reduction. Within the cleaned
reference, the new path adds 24 events over the 78-event legacy result. Controlled
15-minute spike overlaps improve from 2/10 to 9/10, and six-hour shift overlaps
from 1/10 to 9/10. Background overlaps and out-of-interval coverage are reported
separately. These denominators differ from the earlier five-reference pilot.

The existing injection validation uses all 33 locally available August files,
checks embedded original power against independent originals, and freezes July
references. Only known edited points are replayed; signed-power modifications
outside those points never alter the historical reference or replay. Sources
remain untouched. Scoring is limited to paired dates containing edits, not a
full-month deployment run. Canonical 30/15-minute score fields retain their
usual meaning. In the September event-policy output, `event_is_anomaly` and
`supplemental_event_is_anomaly` separately describe event coverage; five-minute
event overlap is recalculated for each 15-minute half, not copied from 30 minutes.

Of 33 injection channels, 28 are admitted and five remain `constant_signal`.
There are 6,292 known changed points overall and 5,232 on scored channels.
Transition-guarded event coverage on scored changed points is 86.16%, versus
77.87% for legacy. Coverage over **all** changed points, including unavailable
channels, is 71.65%, versus 64.75%. Continuous edited-run overlaps improve from
29/94 to 48/94. A continuous changed run is not necessarily one designed anomaly.
Unmodified observations are not verified normal labels, so precision, F1, and
real-world false-positive rates are not claimed. Of 46 missed scored runs, 24
have fewer than three changed five-minute points, below the retained 15-minute
formal-event duration. Larger missed runs remain for investigation.

The new path is a tested optional experiment, not a default production change.
Sparse-load admission, sub-15-minute point alerts, full-month background burden,
and independently labelled normal/fault cases still need further validation.

Verification passed 32 automated tests, independently recalculated all admitted
injection-channel event/point totals, checked all 66 September canonical grids
and their point-based event flags, and confirmed unchanged hashes for all 66
original/injected input files and the original production scores/events/code.

## 保守告警分流：减少优先核验负担，不删除检测结果

新的 `shared/scripts/triage_matrix_profile_alerts.py` 读取现有九月正式事件，复用冻结七月历史参考，但不运行新的 Matrix Profile 评分或补漏路径。规则由 `configs/matrix_profile_alert_triage.json` 控制，所有参数在运行前固定，不根据九月全局分布或注入标签调节。

```powershell
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/triage_matrix_profile_alerts.py --workers 4 --output-dir runs/analysis/new_alert_triage
```

默认输入是原始九月运行目录、同名准备后输入目录，以及已有的 `matrix_profile_existing_injections_20261005` 配对验证。可用 `--run-dir`、`--input-dir`、`--validation-dir`、`--config` 修改相对路径。只支持固定历史参考模式；新运行必须使用空输出目录。其他月份使用相应运行元数据确定检测期，注入检查仍使用已保存的八月配对验证，不能解释为其他月份的真值评估。

完成的结果在 `runs/analysis/matrix_profile_alert_triage_20261005_v2/`，早期草稿目录不应作为最终结论。完整输出、规则与限制见 README 第11节。HTML可以单独重新渲染，不重新分流：

```powershell
& ./.venv/matrix_profile/Scripts/python.exe shared/scripts/triage_matrix_profile_alerts.py --finalize-only --output-dir runs/analysis/matrix_profile_alert_triage_20261005_v2
```

重新渲染保留原执行时的实现哈希，并另记报告渲染代码哈希；若分类／历史校准实现或配置已变化，会要求新运行。不能用重新渲染把旧结果冒充新规则的运行结果。

本次将1,928条原事件完整保留，分为58条优先事件（49个核验单元）、1,581条运行变化待核验（1,484个单元）及289条数据质量事件（172个单元）。总核验单元仍有1,705个；优先缩小不意味着全部异常变成49个。优先时间只取真实五分钟曲线满足幅度、持续、突变及佐证的连续核心，`priority_intervals.csv` 提供半开区间；如投影到十五分钟，应按实际区间重叠计算，不能从原三十分钟分数复制。它是核验附表，不替换统一模型分数 CSV。

局限已在配对验证中明确量化：已有注入文件没有事件进入优先队列，原路径31条质量事件覆盖4,036个有效修改点；另38个原有事件覆盖的修改点仅留待核验，对应4个连续修改区段。原路径的优先核心＋质量覆盖率为77.14%，与全部保留的77.87%分开报告。补漏路径仅留待核验的改变点更多（转变限制472个、较宽松475个）。全体改变点6,292个、其中1,060个不可评分，保持独立分母；未修改点没有已确认正常标签，不能计算真实精确率或误报率。

49个优先单元中34个来自太阳能、14个来自并网通道、1个来自照明与动力。天气和正常运行状态仍需外部确认。因此当前推荐用途是**排序与核验减负**，不是自动故障裁决，也不是删除待核验结果。原始输入、生产分数／事件和其他三个模型未改写。新增回归测试检查队列分离、原ID与字段保留、缺失／质量区段、同方向持续性、真实五分钟核心及十五分钟重叠；当前完整 Matrix Profile 测试集共51项通过。另独立核对全部1,928条事件的字段／成员关系、93个真实时间核心、84份注入路径输出及生产分数／事件哈希，均通过。
