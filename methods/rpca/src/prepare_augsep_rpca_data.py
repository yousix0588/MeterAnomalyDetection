from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path("/data/projects/punim1257/Group14")

DAILY = ROOT / "dataset/AugSep_meters"
HIST = ROOT / "meter_csvs(till_Aug)"
OUT = ROOT / "meter_csvs_augsep_rpca"

OUT.mkdir(exist_ok=True)

TEST_START = pd.Timestamp("2026-08-01", tz="Australia/Melbourne")
TEST_END = pd.Timestamp("2026-10-01", tz="Australia/Melbourne")

# All date folders belonging to Aug + Sep only.
date_dirs = []

for p in DAILY.iterdir():
    if not p.is_dir():
        continue

    try:
        d = datetime.strptime(p.name, "%d_%m_%Y")
    except ValueError:
        continue

    ts = pd.Timestamp(d, tz="Australia/Melbourne")

    if TEST_START <= ts < TEST_END:
        date_dirs.append((ts, p))

date_dirs.sort()

# Unique channels present during Aug/Sep.
channels = sorted({
    f.stem
    for _, d in date_dirs
    for f in d.glob("*.csv")
})

print("Aug/Sep date folders:", len(date_dirs))
print("Aug/Sep unique channels:", len(channels))

manifest = []

for n, ch in enumerate(channels, 1):

    hist_file = HIST / f"{ch}.csv"

    if not hist_file.exists():
        manifest.append({
            "meter_channel": ch,
            "status": "no_history",
            "history_rows": 0,
            "test_rows": 0,
        })
        print(f"[{n}/{len(channels)}] {ch}: no history")
        continue

    # ----------------------------------------
    # Historical baseline: strictly before Aug 1
    # ----------------------------------------
    hist = pd.read_csv(hist_file)

    hist_ts = pd.to_datetime(
        hist["timestamp"],
        utc=True,
        errors="coerce"
    ).dt.tz_convert("Australia/Melbourne")

    hist = hist.loc[
        hist_ts.notna() & (hist_ts < TEST_START)
    ].copy()

    # ----------------------------------------
    # New Aug/Sep test data
    # ----------------------------------------
    pieces = []

    for _, d in date_dirs:
        f = d / f"{ch}.csv"

        if f.exists():
            pieces.append(pd.read_csv(f))

    if not pieces:
        manifest.append({
            "meter_channel": ch,
            "status": "no_test_data",
            "history_rows": len(hist),
            "test_rows": 0,
        })
        continue

    test = pd.concat(pieces, ignore_index=True)

    # ----------------------------------------
    # Combine baseline + new Aug/Sep test
    # ----------------------------------------
    combined = pd.concat(
        [hist, test],
        ignore_index=True
    )

    # Parse timestamps for sorting/deduplication.
    parsed = pd.to_datetime(
        combined["timestamp"],
        utc=True,
        errors="coerce"
    )

    combined = combined.loc[parsed.notna()].copy()
    combined["_parsed_timestamp"] = parsed[parsed.notna()].values

    combined = (
        combined
        .sort_values("_parsed_timestamp")
        .drop_duplicates(
            subset="_parsed_timestamp",
            keep="last"
        )
    )

    combined = combined.drop(
        columns="_parsed_timestamp"
    )

    out_file = OUT / f"{ch}.csv"
    combined.to_csv(out_file, index=False)

    manifest.append({
        "meter_channel": ch,
        "status": "ok",
        "history_rows": len(hist),
        "test_rows": len(test),
        "total_rows": len(combined),
    })

    print(
        f"[{n}/{len(channels)}] {ch}: "
        f"history={len(hist)}, "
        f"test={len(test)}, "
        f"total={len(combined)}"
    )

manifest_df = pd.DataFrame(manifest)

manifest_df.to_csv(
    ROOT / "augsep_merge_manifest.csv",
    index=False
)

# RPCA channel manifest: only channels with usable history.
ok = manifest_df[
    manifest_df["status"] == "ok"
][["meter_channel"]]

ok.to_csv(
    ROOT / "rpca_channels_augsep.csv",
    index=False
)

print("\n===== DONE =====")
print(manifest_df["status"].value_counts())
print("RPCA channels:", len(ok))
print("Output:", OUT)
