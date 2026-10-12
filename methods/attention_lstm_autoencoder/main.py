"""Independent Attention AE entry point; paths resolve from group14reshape."""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from methods.attention_lstm_autoencoder import project


def parser():
    value=argparse.ArgumentParser(description=__doc__)
    commands=value.add_subparsers(dest="command",required=True)
    for name in ("validate","infer"):
        child=commands.add_parser(name)
        child.add_argument("--config",default="configs/attention_lstm_autoencoder_inference.json")
        child.add_argument("--series-id",action="append")
        child.add_argument("--max-channels",type=int)
        if name=="validate":
            child.add_argument("--verbose",action="store_true",help="Include every selected channel in the preflight report")
        if name=="infer":
            child.add_argument("--output-dir")
    for name in ("prepare","train","predict","export"):
        child=commands.add_parser(name)
        child.add_argument("--config",default="configs/attention_lstm_autoencoder.yaml")
        if name=="train":
            child.add_argument("--resume",action="store_true")
    child=commands.add_parser("train-by-category")
    child.add_argument("--config",default="configs/attention_lstm_autoencoder.yaml")
    child.add_argument("--repository",default="methods/matrix_profile/src/meter_repository.json")
    child.add_argument("--work-dir",default="data/processed/attention_lstm_autoencoder/by_category")
    child.add_argument("--output-root",default="runs/attention_lstm_autoencoder/training/by_category")
    child.add_argument("--prepare-only",action="store_true")
    child.add_argument("--resume",action="store_true")
    child.add_argument("--min-channels",type=int,default=20)
    return value


def main(argv=None):
    args=parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command in ("validate","infer"):
            from methods.attention_lstm_autoencoder.inference import make_plan,run_inference
            selected=dict(series_ids=args.series_id,max_channels=args.max_channels)
            if args.command=="validate":
                report=make_plan(args.config,**selected)[0]
                if not args.verbose:
                    from collections import Counter
                    report["channel_status_counts"]=dict(Counter(row["status"] for row in report["channels"]))
                    report.pop("channels")
                print(json.dumps(report,ensure_ascii=False,indent=2))
                return 0 if report["trained_packages_ready"] and report["torch_installed"] else 2
            print(run_inference(args.config,output_dir=args.output_dir,**selected))
            return 0
        config=project.load_training_config(args.config)
        if os.name!="posix":
            raise RuntimeError("Native preparation/training/replay needs Spartan/Linux/WSL; validate is cross-platform")
        if args.command=="train-by-category":
            work=project.resolve(args.work_dir)
            if not work.is_relative_to((project.ROOT/"data/processed"/project.MODEL).resolve()):
                raise ValueError("Category work directory must stay in separate Attention processed data")
            command=[sys.executable,str(ROOT/"methods"/project.MODEL/"train_by_category.py"),
                "--config",str(project.resolve(args.config)),"--repository",str(project.resolve(args.repository)),
                "--data-root",config["data"]["root"],"--project",str(ROOT/"methods"/project.MODEL),
                "--work-dir",str(work),"--output-root",str(project.output_path(args.output_root)),
                "--min-channels",str(args.min_channels)]
            if args.prepare_only: command.append("--prepare-only")
            if args.resume: command.append("--resume")
            return subprocess.run(command,cwd=ROOT).returncode
        config=project.training_input_view(config)
        pipeline=importlib.import_module("methods.attention_lstm_autoencoder.src.pipeline")
        if args.command=="prepare":
            from methods.attention_lstm_autoencoder.src.utils import StopRequest
            with StopRequest() as stop:
                manifest,_=pipeline.prepare_data(config,stop)
            print(f"Prepared {len(manifest['shards'])} native shards")
        elif args.command=="train":
            print(pipeline.train(config,resume=args.resume))
        elif args.command=="predict":
            print(pipeline.predict_saved(config))
        else:
            print(pipeline.export_scores(config))
        return 0
    except InterruptedError as exc:
        print(str(exc),file=sys.stderr); return 75
    except (OSError,ValueError,KeyError,RuntimeError,ImportError) as exc:
        print(f"Attention AE: {exc}",file=sys.stderr); return 2


if __name__=="__main__":
    raise SystemExit(main())
