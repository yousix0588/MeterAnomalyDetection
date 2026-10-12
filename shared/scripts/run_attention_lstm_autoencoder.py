"""Project-level alias for the independent Attention AE entry point."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from methods.attention_lstm_autoencoder.main import main

if __name__=="__main__":
    raise SystemExit(main())
