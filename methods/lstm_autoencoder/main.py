from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.pipeline import predict, train
from src.utils import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "lstm_autoencoder.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "lstm_autoencoder" / "aligned_august"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LSTM Autoencoder 电表无监督异常检测")
    sub = p.add_subparsers(dest="command", required=True)
    train_p = sub.add_parser("train", help="训练、标定阈值并评估")
    train_p.add_argument("--config", default=str(DEFAULT_CONFIG))
    pred_p = sub.add_parser("predict", help="使用已训练模型检测当前 data 目录")
    pred_p.add_argument("--config", default=str(DEFAULT_CONFIG))
    pred_p.add_argument("--checkpoint", default=str(DEFAULT_OUTPUT / "best_model.pt"))
    pred_p.add_argument("--scaler", default=str(DEFAULT_OUTPUT / "scaler.joblib"))
    pred_p.add_argument("--metadata", default=str(DEFAULT_OUTPUT / "metadata.json"))
    return p


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = load_config(args.config)
    for section, key in (("data", "root"), ("output", "directory")):
        configured_path = Path(config[section][key])
        if not configured_path.is_absolute():
            config[section][key] = str(PROJECT_ROOT / configured_path)
    if args.command == "train":
        output = train(config)
        print(f"训练完成，结果位于: {output.resolve()}")
    else:
        metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
        target = predict(config, args.checkpoint, args.scaler, metadata["threshold"])
        print(f"检测完成: {target.resolve()}")


if __name__ == "__main__":
    main()
