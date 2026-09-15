"""
Build one combined, chronologically-sorted CSV per meter-channel from the
raw dataset -- accepts a .zip, a .tar/.tar.gz/.tgz, OR a plain already-
extracted directory as input (see ArchiveReader below). No .rar support
is needed: extract the .rar locally (7-Zip/WinRAR/Archive Utility all read
.rar), then point --data at either the extracted folder directly, or a
.tar.gz you make from it -- both work without ever touching .rar on Spartan.

v4 changes: multiple separate archives (e.g. a July zip AND a separate
August zip) for the same dataset
------------------------------------------------------------------------
--data now accepts MULTIPLE paths. All archives are opened together and a
given meter's day-files from every archive are merged into ONE combined,
chronologically-sorted output CSV. This matters: running this script
separately once per archive would NOT append -- the second run would
overwrite each meter's CSV using only that second archive's contents,
silently discarding the first archive's data. Passing all archives in one
call is required to get a correct merged result.

v3 changes for a fixed train/test date split
----------------------------------------------
- --date_end: clip out any day-folder AFTER this date (e.g. discard
  September onward if you only need through August). Saves build time +
  disk space since discarded days are never read.
- --require_start / --require_end: only build a meter-channel if it has
  at least one day-file inside this window. Set this to your test period
  (e.g. July+August) so meters that never appear in that window are
  skipped entirely.
- Still streams/chunks writes and supports resume (--force to rebuild).

Usage
-----
    # single archive
    python build_meter_csvs.py --data full_data.zip --out_dir ./meter_csvs \
        --date_end 2026-08-31 --require_start 2026-07-01 --require_end 2026-08-31

    # July and August as SEPARATE zips, merged into one output per meter
    python build_meter_csvs.py --data july_data.zip august_data.zip --out_dir ./meter_csvs \
        --date_end 2026-08-31 --require_start 2026-07-01 --require_end 2026-08-31 --force

    # mixing formats is fine too (zip + tar.gz + extracted folder, any combination)
    python build_meter_csvs.py --data july_data.zip ./august_extracted/ --out_dir ./meter_csvs ...
"""

import argparse
import io
import re
import sys
import tarfile
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DATE_RE = re.compile(r"^(\d{2})_(\d{2})_(\d{4})$")


class ArchiveReader:
    """
    Uniform read-only interface over a .zip, a .tar/.tar.gz/.tgz, or a plain
    directory, so the rest of the script doesn't care which one it got.
    Does NOT support .rar (Python has no built-in rar reader, and getting
    unrar onto an HPC login node is more friction than it's worth) --
    extract .rar locally first, then use the resulting folder or re-pack
    it as .tar.gz.
    """
    def __init__(self, path: str):
        self.path = Path(path)
        if self.path.is_dir():
            self.kind = "dir"
        elif self.path.suffix == ".zip":
            self.kind = "zip"
            self._zf = zipfile.ZipFile(self.path)
        elif "".join(self.path.suffixes[-2:]) in (".tar.gz",) or self.path.suffix in (".tar", ".tgz"):
            self.kind = "tar"
            mode = "r:gz" if self.path.suffix in (".tgz",) or self.path.suffixes[-2:] == [".tar", ".gz"] else "r"
            self._tf = tarfile.open(self.path, mode)
        elif self.path.suffix == ".rar":
            raise ValueError(
                f"{path} is a .rar file -- Python can't read rar archives natively. "
                f"Extract it locally (7-Zip / WinRAR / Archive Utility) then pass either "
                f"the extracted folder or a re-packed .tar.gz to --data instead."
            )
        else:
            raise ValueError(f"Unsupported input: {path} (expected .zip, .tar[.gz]/.tgz, or a directory)")

    def namelist(self):
        if self.kind == "dir":
            return [str(p.relative_to(self.path)).replace("\\", "/") for p in self.path.rglob("*") if p.is_file()]
        if self.kind == "zip":
            return self._zf.namelist()
        if self.kind == "tar":
            return [m.name for m in self._tf.getmembers() if m.isfile()]

    def open(self, relpath: str):
        if self.kind == "dir":
            return open(self.path / relpath, "rb")
        if self.kind == "zip":
            return self._zf.open(relpath)
        if self.kind == "tar":
            return self._tf.extractfile(relpath)

    def close(self):
        if self.kind == "zip":
            self._zf.close()
        if self.kind == "tar":
            self._tf.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def parse_folder_date(folder_name: str):
    """'05_07_2026' -> datetime(2026, 7, 5). Returns None if unparseable."""
    m = DATE_RE.match(folder_name)
    if not m:
        return None
    day, month, year = m.groups()
    return datetime(int(year), int(month), int(day))


def find_date_in_path(path: str):
    """
    Find the DD_MM_YYYY date folder anywhere in a file's path, not just the
    first segment -- handles datasets exported with an extra wrapper folder,
    e.g. 'RACV+InchScape_25Oct-19Aug/01_07_2026/meter_0.csv' as well as the
    plain '01_07_2026/meter_0.csv' layout. Returns None if no segment matches.
    """
    for part in path.split("/"):
        d = parse_folder_date(part)
        if d is not None:
            return d
    return None


def build_meter_csvs(data_paths, out_dir: str, limit: int | None = None, force: bool = False,
                      date_end: str = None, require_start: str = None, require_end: str = None):
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    date_end_dt = pd.Timestamp(date_end).to_pydatetime() if date_end else None
    require_start_dt = pd.Timestamp(require_start).to_pydatetime() if require_start else None
    require_end_dt = pd.Timestamp(require_end).to_pydatetime() if require_end else None

    # Open ALL archives at once (e.g. the July zip AND a separate August zip),
    # so a meter's day-files from BOTH are combined into one output CSV in a
    # single pass. Running this script once per archive sequentially would
    # instead have the second run overwrite the first's output entirely,
    # since each run currently only knows about entries in its own archive.
    archives = [ArchiveReader(p) for p in data_paths]
    try:
        # every entry tracked as (archive_index, relpath) so we know which
        # archive to re-open it from later, while dates/basenames/grouping
        # work exactly as before regardless of how many archives contributed
        all_entries = [(i, n) for i, ar in enumerate(archives) for n in ar.namelist()]
        csv_entries = [(i, n) for i, n in all_entries if n.endswith(".csv")]
        miss_entries = [(i, n) for i, n in all_entries if n.endswith(".miss")]

        groups = defaultdict(list)
        for i, n in csv_entries:
            folder_date = find_date_in_path(n)
            if folder_date is None:
                continue
            if date_end_dt is not None and folder_date > date_end_dt:
                continue  # discard days after the cutoff (e.g. September onward)
            base = n.split("/")[-1]
            groups[base].append((i, n))

        if not groups:
            sample = [n for _, n in csv_entries[:5]]
            sys.exit(
                f"Found {len(csv_entries)} .csv entries across {len(data_paths)} archive(s), but NONE contained "
                f"a 'DD_MM_YYYY' date folder anywhere in their path (e.g. '05_07_2026/DD123_0.csv'), so zero "
                f"meter-channels were built. Sample paths actually found:\n  "
                + "\n  ".join(sample) +
                f"\nCheck the folder naming in your dataset against DATE_RE in this script."
            )

        miss_counts = defaultdict(int)
        for i, n in miss_entries:
            base = n.split("/")[-1].replace(".miss", ".csv")
            miss_counts[base] += 1

        meter_channels = sorted(groups.keys())

        # only keep meters that have at least one day-file inside the required window
        # (e.g. the test period) -- meters absent from that window are skipped entirely
        if require_start_dt is not None or require_end_dt is not None:
            kept = []
            for base in meter_channels:
                dates = [find_date_in_path(p) for _, p in groups[base]]
                has_required = any(
                    (require_start_dt is None or d >= require_start_dt) and
                    (require_end_dt is None or d <= require_end_dt)
                    for d in dates if d is not None
                )
                if has_required:
                    kept.append(base)
            print(f"{len(kept)}/{len(meter_channels)} meter-channels have data in the required window "
                  f"[{require_start}, {require_end}] -- only these will be built.")
            meter_channels = kept
            if not meter_channels:
                sample_dates = sorted({d.strftime("%Y-%m-%d") for paths in groups.values()
                                        for d in [find_date_in_path(p) for _, p in paths] if d})[:10]
                sys.exit(
                    f"0 meter-channels have data in the required window [{require_start}, {require_end}]. "
                    f"Dates actually present (sample): {sample_dates}\n"
                    f"Check that --require_start/--require_end fall inside that range."
                )

        if limit:
            meter_channels = meter_channels[:limit]

        print(f"found {len(groups)} meter-channels ({len(csv_entries)} csv files across {len(data_paths)} "
              f"archive(s), {len(miss_entries)} missing-day placeholders). Building {len(meter_channels)}...")

        manifest_rows = []
        for base in tqdm(meter_channels, desc="combining"):
            meter_id = base.replace(".csv", "")
            out_path = out_dir / f"{meter_id}.csv"

            if out_path.exists() and not force:
                continue  # resume support

            # sort this meter's day-files chronologically BEFORE reading (across
            # ALL archives together), so streamed writes land on disk in order
            paths_sorted = sorted(groups[base], key=lambda t: find_date_in_path(t[1]) or datetime.min)

            n_rows_total = 0
            n_days_used = 0
            n_bad_rows_total = 0
            start_ts, end_ts = None, None
            last_ts = None
            first_chunk = True

            with open(out_path, "w", newline="") as out_f:
                for archive_idx, p in paths_sorted:
                    with archives[archive_idx].open(p) as f:
                        chunk = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"))
                    if chunk.empty or "timestamp" not in chunk.columns:
                        continue

                    # Parse explicitly and robustly. Australian timestamps mix +10:00
                    # (AEST) and +11:00 (AEDT) across the DST boundary (Apr/Oct), which
                    # pandas refuses to auto-parse in one column without utc=True. We
                    # parse to UTC first (unambiguous instant for every row), then
                    # convert to Australia/Melbourne so local calendar-day boundaries
                    # (e.g. --train_end '2026-07-01') still line up correctly, DST and
                    # all. Any value that still fails to parse becomes NaT and is dropped.
                    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True) \
                                            .dt.tz_convert("Australia/Melbourne")
                    n_bad = chunk["timestamp"].isna().sum()
                    if n_bad:
                        n_bad_rows_total += int(n_bad)
                        chunk = chunk.dropna(subset=["timestamp"])
                    if chunk.empty:
                        continue

                    chunk = chunk.drop_duplicates("timestamp").sort_values("timestamp")

                    # guard against any timestamp overlap with the previous day's tail
                    # (also protects against the SAME day appearing in both archives,
                    # e.g. if the July and August zips overlap by a day at the boundary)
                    if last_ts is not None:
                        chunk = chunk[chunk["timestamp"] > last_ts]
                    if chunk.empty:
                        continue

                    chunk.to_csv(out_f, index=False, header=first_chunk)
                    first_chunk = False
                    n_rows_total += len(chunk)
                    n_days_used += 1
                    last_ts = chunk["timestamp"].iloc[-1]
                    if start_ts is None:
                        start_ts = chunk["timestamp"].iloc[0]
                    end_ts = last_ts

            if n_rows_total == 0:
                out_path.unlink(missing_ok=True)  # nothing usable; don't leave an empty file
                continue

            manifest_rows.append({
                "meter_channel": meter_id,
                "n_rows": n_rows_total,
                "n_days_present": n_days_used,
                "n_days_missing": miss_counts.get(base, 0),
                "n_bad_timestamp_rows_dropped": n_bad_rows_total,
                "start": start_ts,
                "end": end_ts,
            })
    finally:
        for ar in archives:
            ar.close()

    manifest_cols = ["meter_channel", "n_rows", "n_days_present", "n_days_missing",
                      "n_bad_timestamp_rows_dropped", "start", "end"]
    manifest = (pd.DataFrame(manifest_rows, columns=manifest_cols).sort_values("meter_channel")
                if manifest_rows else pd.DataFrame(columns=manifest_cols))
    manifest_path = out_dir / "_manifest.csv"
    # append to existing manifest if resuming a partial run, else write fresh
    if manifest_path.exists() and not force and len(manifest_rows):
        prior = pd.read_csv(manifest_path)
        manifest = (pd.concat([prior, manifest])
                      .drop_duplicates("meter_channel", keep="last")
                      .sort_values("meter_channel"))
    manifest.to_csv(manifest_path, index=False)
    print(f"\nDone. Combined CSVs -> {out_dir}")
    print(f"Manifest -> {manifest_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "--zip", dest="data", required=True, nargs="+",
                         help="One or more dataset paths (.zip, .tar/.tar.gz/.tgz, or an extracted directory). "
                              "Pass multiple paths (e.g. --data july.zip august.zip) to combine separate "
                              "archives for the same meters into one output per meter-channel.")
    parser.add_argument("--out_dir", required=True, help="Where to write combined per-meter CSVs")
    parser.add_argument("--limit", type=int, default=None, help="Only build the first N meter-channels (for testing)")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the output CSV already exists")
    parser.add_argument("--date_end", default=None, help="Discard any day after this date, e.g. 2026-08-31")
    parser.add_argument("--require_start", default=None, help="Only build meters with data on/after this date")
    parser.add_argument("--require_end", default=None, help="...and on/before this date (define your test window)")
    args = parser.parse_args()

    build_meter_csvs(args.data, args.out_dir, args.limit, args.force,
                      args.date_end, args.require_start, args.require_end)