# 8 月异常人工核验集

在项目根目录运行：

```bash
python shared/scripts/build_manual_review_set.py
```

脚本读取 `runs/ensemble/all_models_scores_15min.csv`、`runs/ensemble/all_models_events.csv` 和 `data/processed/meter_csvs/`，默认写入 `runs/ensemble/manual_review_august_v1/`。如果该目录已有文件，脚本会停止，不会覆盖已填写的人工标签。需要重新抽样时，使用新的 `--output-dir`。

默认抽取约 26 个案例：四模型一致 6 个、四模型均可用但只有三模型标记 6 个、四模型均可用但只有 Matrix Profile 标记 6 个（周末和工作日各半）、以及每个模型最多 2 个持续至少 6 小时的长事件。三模型一致案例优先覆盖四种不同的缺席模型组合。只选择前后上下文都在 8 月范围内的起点。选择尽量分散到不同设备，并避免同一通道 12 小时内重复抽取。候选排序和 Matrix Profile 单独报警的抽样都是确定性的，重复运行可得到相同案例；实际类别数量受可用候选和去重条件限制。

输出文件：

- `review_cases.csv`：一行一个待核验案例。`review_status`、`review_label`、`review_notes` 留给人工填写。建议标签为 `confirmed_anomaly`、`scheduled_operation`、`data_quality_issue`、`normal` 或 `uncertain`；这些不是模型自动生成的真值。
- `review_timeline_15min.csv`：每个案例起点前后各 24 小时的 15 分钟网格。包括四模型逐点分数、异常标记、可用状态，以及对应三个 5 分钟原始点的功率、电流、电压和功率因数统计。模型未输出的格子保持空白，不补零。
- `review_raw_5min.csv`：相同时间范围内未经重新缩放的原始 5 分钟测量值，保留源 CSV 的所有测量列。`minutes_from_anchor` 可用于比较事件发生前后变化。
- `review_manifest.json`：输入位置、抽样规则、候选总体和实际输出行数。

时间采用源文件的 `+10:00` 偏移，区间均为左闭右开。`anchor_start` 是人工查看的起点；长事件取连续标记的开始时间，48 小时上下文并不代表事件的完整持续时间。Matrix Profile 的源分辨率仍是 30 分钟，相邻两个 15 分钟格不是独立观测。`all_models_events.csv` 由连续逐点标记生成，因此这里的长事件也是**待核验候选**，不是已确认故障。

建议先看 `review_cases.csv`，按 `case_id` 到另外两张表过滤对应曲线。检查原始测量点是否缺失或恒定、异常是否是每天固定时间的正常运行、其他模型是否同时变化，然后再填写人工标签。不要把空白的模型分数解释为正常，也不要直接把 `review_label` 以外的模型标记当作真值。

## 案例图

运行 `python shared/scripts/plot_manual_review_set.py` 会在核验集目录的 `plots/` 下生成 26 张案例图和 `index.html` 浏览页。每张图展示相同的前后 24 小时时间轴，分别绘制原始 5 分钟功率、原始电流、四模型最大百分位和逐时段异常标记；红色虚线表示选中的 15 分钟时段。Matrix Profile 使用阶梯线并注明 30 分钟结果重复映射，缺失分数保留为线条间断。该命令也不会覆盖已有图片；重绘时用新的 `--output-dir`。

可调整抽取数量或上下文长度，例如：

```bash
python shared/scripts/build_manual_review_set.py \
  --per-category 8 --long-per-model 2 --context-hours 24 \
  --output-dir runs/ensemble/manual_review_august_v2
```
