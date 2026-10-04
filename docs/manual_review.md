# 8 月异常人工核验集

在项目根目录运行（新版本写入独立目录，不覆盖已有的 v1 核验记录）：

```bash
python shared/scripts/build_manual_review_set.py --output-dir runs/ensemble/manual_review_august_v2
```

脚本读取 `runs/ensemble/all_models_scores_15min.csv`、`runs/ensemble/all_models_events.csv` 和 `data/processed/meter_csvs/`。使用上面的命令写入 `runs/ensemble/manual_review_august_v2/`；旧的 v1 仍保留。如果目标目录已有文件，脚本会停止，不会覆盖已填写的人工标签。再次抽样时使用新的 `--output-dir`。

默认抽取约 41 个案例：保留原来的四模型一致 6 个、三模型标记 6 个、仅 Matrix Profile 标记 6 个（周末和工作日各半）、每模型最多 2 个至少持续 6 小时的事件；另增仅 LSTM AE、仅 LSTM-VAE、仅 RPCA 报警各 5 个。三个新增类别均要求该时段四模型都有结果，避免把“其他模型缺数据”误认为“其他模型没有报警”。v2 的前 26 个案例及编号与 v1 相同，新增案例从 MR027 开始。三模型一致案例优先覆盖四种不同的缺席模型组合。只选择前后上下文都在 8 月范围内的起点。选择尽量分散到不同设备，并避免同一通道 12 小时内重复抽取。候选抽样是确定性的；实际数量受可用候选和去重条件限制。

输出文件：

- `review_cases.csv`：一行一个待核验案例。`review_status`、`review_label`、`review_notes` 留给人工填写。建议标签为 `confirmed_anomaly`、`scheduled_operation`、`data_quality_issue`、`normal` 或 `uncertain`；这些不是模型自动生成的真值。
- `review_timeline_15min.csv`：每个案例起点前后各 24 小时的 15 分钟网格。包括四模型逐点分数、异常标记、可用状态，以及对应三个 5 分钟原始点的功率、电流、电压和功率因数统计。模型未输出的格子保持空白，不补零。
- `review_raw_5min.csv`：相同时间范围内未经重新缩放的原始 5 分钟测量值，保留源 CSV 的所有测量列。`minutes_from_anchor` 可用于比较事件发生前后变化。
- `review_manifest.json`：输入位置、抽样规则、候选总体和实际输出行数。

时间采用源文件的 `+10:00` 偏移，区间均为左闭右开。`anchor_start` 是人工查看的起点；长事件取连续标记的开始时间，48 小时上下文并不代表事件的完整持续时间。Matrix Profile 的源分辨率仍是 30 分钟，相邻两个 15 分钟格不是独立观测。`all_models_events.csv` 由连续逐点标记生成，因此这里的长事件也是**待核验候选**，不是已确认故障。

单模型案例采用有意均衡的核验配额，不代表四个模型在全部数据中的报警比例，也不能直接用这 41 个案例计算总体准确率。人工填写标签后，应先按 `source_model`、`category` 和 `review_label` 分组查看各类误报与漏报，再决定是否调整阈值或集成规则。

建议先看 `review_cases.csv`，按 `case_id` 到另外两张表过滤对应曲线。检查原始测量点是否缺失或恒定、异常是否是每天固定时间的正常运行、其他模型是否同时变化，然后再填写人工标签。不要把空白的模型分数解释为正常，也不要直接把 `review_label` 以外的模型标记当作真值。

## 案例图

运行下面的命令会在 v2 核验集的 `plots/` 下生成对应案例图和 `index.html` 浏览页：

```bash
python shared/scripts/plot_manual_review_set.py --review-dir runs/ensemble/manual_review_august_v2
```

每张图展示相同的前后 24 小时时间轴，分别绘制原始 5 分钟功率、原始电流、四模型最大百分位和逐时段报警标记；红色虚线表示选中的 15 分钟时段。图标题和索引明确显示红线处 `1/4`、`3/4` 或 `4/4` 模型报警，均为**待人工核验**，不是已确认故障。Matrix Profile 使用阶梯线并注明 30 分钟结果重复映射，缺失分数保留为线条间断。该命令也不会覆盖已有图片；重绘时用新的 `--output-dir`。

可调整抽取数量或上下文长度，例如：

```bash
python shared/scripts/build_manual_review_set.py \
  --per-category 8 --long-per-model 2 --single-per-model 5 --context-hours 24 \
  --output-dir runs/ensemble/manual_review_august_v3
```
