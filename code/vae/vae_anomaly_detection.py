"""
VAE-based unsupervised anomaly detection for electric meter time series.

Pipeline
--------
1. Load a single meter-channel's combined CSV (5-minute resolution).
2. Resample to a regular grid, interpolate short gaps, scale features
   (StandardScaler fit ONLY on the assumed-normal training slice).
3. Slice into overlapping sliding windows.
4. Train an LSTM sequence-to-sequence Variational Autoencoder (VAE) to
   reconstruct normal windows.
5. Score every window by its reconstruction error + KL term (negative
   ELBO). Map window scores back onto the timeline.
6. Flag anomalous timestamps via a threshold (percentile-based) and
   group consecutive flags into "anomaly periods".
7. Plot: (a) raw signal with anomaly periods shaded, (b) anomaly score
   over time with threshold line.

This is written per-meter (one model per meter/channel) because meter
behaviour is assumed NOT to be shared across meters -- re-run this
script (or loop over meters) with each meter's own CSV/scaler/model.

Usage
-----
    python vae_anomaly_detection.py --csv path/to/meter.csv \
        --features pRealKw pReactiveKw powerFactor vRMSMax iRMSMax \
        --window 48 --stride 1 --epochs 40 --out_dir ./out

Requires: torch, pandas, numpy, scikit-learn, matplotlib
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# --------------------------------------------------------------------------- #
# 1. Data loading & preprocessing
# --------------------------------------------------------------------------- #

def load_meter_series(csv_path: str, features: list[str], freq: str = "5min",
                       local_tz: str = "Australia/Melbourne") -> pd.DataFrame:
    """Load one meter's CSV, set a regular DatetimeIndex, interpolate short gaps.

    Timestamps are parsed via utc=True first, then converted to local_tz.
    This is required because Australian timestamps mix +10:00 (AEST) and
    +11:00 (AEDT) offsets across the DST boundary (Apr/Oct) -- pandas
    refuses to auto-parse a column containing both offsets otherwise
    (parse_dates=[...] alone raises "Mixed timezones detected" or silently
    leaves the column as plain strings, causing a plain Index with no .tz
    downstream). Parsing to UTC first is unambiguous for every row; the
    tz_convert back to local_tz keeps local calendar-day boundaries (e.g.
    --train_end '2026-07-01') aligned correctly, DST included.
    """
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(local_tz)
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")

    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz=df.index.tz)
    df = df.reindex(full_index)
    df.index.name = "timestamp"

    missing_mask = df[features].isna().any(axis=1)

    # interpolate short gaps (<= 6 steps = 30 min); longer gaps stay NaN
    # and are dropped from training but kept (as NaN score) for plotting.
    df[features] = df[features].interpolate(method="time", limit=6, limit_direction="both")

    df["is_imputed"] = missing_mask
    return df


def make_windows(values: np.ndarray, window: int, stride: int) -> np.ndarray:
    """values: (T, F) -> windows: (N, window, F)"""
    n = (len(values) - window) // stride + 1
    return np.stack([values[i * stride: i * stride + window] for i in range(n) if not np.isnan(values[i * stride: i * stride + window]).any()])


def make_windows_with_index(df: pd.DataFrame, features: list[str], window: int, stride: int):
    values = df[features].values.astype(np.float32)
    windows, start_idx = [], []
    for i in range(0, len(values) - window + 1, stride):
        chunk = values[i:i + window]
        if not np.isnan(chunk).any():
            windows.append(chunk)
            start_idx.append(i)
    return np.stack(windows), np.array(start_idx)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append cyclical calendar features (day-of-year, day-of-week) so a model
    trained across many months can represent seasonal/weekly structure
    explicitly, instead of having to relearn it purely from the raw signal.
    Meaningful once a series spans multiple seasons (e.g. 279 days), less
    useful on a short few-week series.
    """
    df = df.copy()
    doy = df.index.dayofyear.values.astype(float)
    dow = df.index.dayofweek.values.astype(float)
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    df["sin_dow"] = np.sin(2 * np.pi * dow / 7)
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7)
    return df


CALENDAR_FEATURES = ["sin_doy", "cos_doy", "sin_dow", "cos_dow"]


# --------------------------------------------------------------------------- #
# 2. LSTM Variational Autoencoder
# --------------------------------------------------------------------------- #

class LSTMVAE(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 64, latent_dim: int = 16, num_layers: int = 1):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder_rnn = nn.LSTM(n_features, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc_latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, n_features)

    def encode(self, x):
        _, (h_n, _) = self.encoder_rnn(x)
        h_last = h_n[-1]                       # (B, hidden_dim)
        return self.fc_mu(h_last), self.fc_logvar(h_last)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        h0 = torch.tanh(self.fc_latent_to_hidden(z)).unsqueeze(0)        # (1, B, hidden)
        h0 = h0.repeat(self.decoder_rnn.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        dec_input = h0[-1].unsqueeze(1).repeat(1, seq_len, 1)             # feed latent context at every step
        out, _ = self.decoder_rnn(dec_input, (h0, c0))
        return self.fc_out(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta: float = 1.0):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="none").mean(dim=[1, 2])  # per-sample
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) / x.size(1)  # per-sample, normalized by seq_len
    return recon_loss, kld, (recon_loss + beta * kld).mean()


# --------------------------------------------------------------------------- #
# 3. Training
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    beta: float = 0.5          # KL weight (beta-VAE); lower beta = prioritize reconstruction fidelity
    hidden_dim: int = 64
    latent_dim: int = 16
    train_frac: float = 0.7    # first N% of the timeline used as "assumed normal" training data


def train_vae(windows: np.ndarray, cfg: TrainConfig, device: str = "cpu", verbose: bool = True,
              _preinitialized_model: "LSTMVAE | None" = None):
    """
    If _preinitialized_model is given (e.g. warm-started from a previous
    retrain step's weights by the cluster pipeline), training continues
    from those weights instead of a fresh random init.
    """
    n_train = int(len(windows) * cfg.train_frac)
    train_x = torch.tensor(windows[:n_train])
    model = _preinitialized_model if _preinitialized_model is not None else \
        LSTMVAE(n_features=windows.shape[-1], hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    dataset = torch.utils.data.TensorDataset(train_x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(batch)
            _, _, loss = vae_loss(recon, batch, mu, logvar, beta=cfg.beta)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(f"epoch {epoch+1:3d}/{cfg.epochs} | train loss {epoch_loss/n_train:.5f}")

    return model


@torch.no_grad()
def score_windows(model: LSTMVAE, windows: np.ndarray, device: str = "cpu", n_samples: int = 8) -> np.ndarray:
    """
    Anomaly score per window = average reconstruction error over several
    stochastic latent samples (Monte-Carlo reconstruction error), a common
    VAE anomaly score that accounts for decoder stochasticity rather than
    relying on a single reparameterized sample.
    """
    model.eval()
    x = torch.tensor(windows).to(device)
    scores = torch.zeros(len(windows))
    for _ in range(n_samples):
        recon, mu, logvar = model(x)
        recon_err, kld, _ = vae_loss(recon, x, mu, logvar, beta=1.0)
        scores += (recon_err + kld).cpu()
    return (scores / n_samples).numpy()


# --------------------------------------------------------------------------- #
# 4. Map window scores -> per-timestamp scores, find anomaly periods
# --------------------------------------------------------------------------- #

def scores_to_timeline(df: pd.DataFrame, window_scores: np.ndarray, start_idx: np.ndarray, window: int) -> pd.Series:
    """Average overlapping window scores onto each timestamp."""
    acc = np.zeros(len(df))
    cnt = np.zeros(len(df))
    for s, idx in zip(window_scores, start_idx):
        acc[idx: idx + window] += s
        cnt[idx: idx + window] += 1
    cnt[cnt == 0] = np.nan
    timeline = acc / cnt
    return pd.Series(timeline, index=df.index, name="anomaly_score")


def find_anomaly_periods(
    score: pd.Series,
    threshold: float,
    min_duration_steps: int = 3,
    severity_multiplier: float = 1.5
) -> pd.DataFrame: 
    """Group consecutive above-threshold timestamps into discrete anomaly periods."""
    flag = (score > threshold).fillna(False)
    periods = []
    start = None
    for t, is_anom in flag.items():
        if is_anom and start is None:
            start = t
        elif not is_anom and start is not None:
            end = t
            periods.append((start, end))
            start = None
    if start is not None:
        periods.append((start, flag.index[-1]))

    rows = []
    for start, end in periods:
        seg = score.loc[start:end]
        n_steps = len(seg)
        max_score = seg.max()
        if (
            n_steps >= min_duration_steps
            and max_score >= threshold * severity_multiplier
        ):
            rows.append({
                "start": start,
                "end": end,
                "duration_minutes": (end - start).total_seconds() / 60,
                "max_score": max_score,
                "mean_score": seg.mean()
            })
    cols = ["start", "end", "duration_minutes", "max_score", "mean_score"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("max_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 5. Plotting
# --------------------------------------------------------------------------- #

def plot_results(df: pd.DataFrame, score: pd.Series, threshold: float,
                  periods: pd.DataFrame, signal_col: str, out_path: str, meter_id: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(df.index, df[signal_col], color="#2f6fed", linewidth=0.8, label=signal_col)
    for _, row in periods.iterrows():
        ax1.axvspan(row["start"], row["end"], color="red", alpha=0.25)
    ax1.set_ylabel(signal_col)
    ax1.set_title(f"Meter {meter_id} — raw signal with detected anomaly periods (shaded)")
    ax1.legend(loc="upper right")

    ax2.plot(score.index, score.values, color="#333333", linewidth=0.8, label="VAE anomaly score")
    ax2.axhline(threshold, color="red", linestyle="--", linewidth=1, label=f"threshold ({threshold:.3f})")
    for _, row in periods.iterrows():
        ax2.axvspan(row["start"], row["end"], color="red", alpha=0.25)
    ax2.set_ylabel("anomaly score\n(recon. err + KL)")
    ax2.legend(loc="upper right")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved diagram -> {out_path}")


# --------------------------------------------------------------------------- #
# 6. Main
# --------------------------------------------------------------------------- #

def run_pipeline(csv_path: str, out_dir: str, features: list[str] = None, signal_col: str = "pRealKw",
                  window: int = 48, stride: int = 1, epochs: int = 40, threshold_pct: float = 99.5,
                  make_plot: bool = True, save_model: bool = True, verbose: bool = True) -> dict:
    """
    Run the full VAE anomaly-detection pipeline for ONE meter-channel CSV.
    Importable so a batch runner can loop over many meters without spawning
    a subprocess per meter. Returns a small summary dict.
    """
    features = features or ["pRealKw", "pReactiveKw", "powerFactor", "vRMSMax", "iRMSMax"]
    os.makedirs(out_dir, exist_ok=True)
    meter_id = os.path.splitext(os.path.basename(csv_path))[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_meter_series(csv_path, features)
    windows, start_idx = make_windows_with_index(df, features, window, stride)
    if len(windows) < 20:
        raise ValueError(f"{meter_id}: only {len(windows)} valid windows (too much missing data / too short a series)")

    cfg = TrainConfig(epochs=epochs)
    n_train_windows = int(len(windows) * cfg.train_frac)

    scaler = StandardScaler().fit(windows[:n_train_windows].reshape(-1, windows.shape[-1]))
    windows_scaled = scaler.transform(windows.reshape(-1, windows.shape[-1])).reshape(windows.shape).astype(np.float32)

    model = train_vae(windows_scaled, cfg, device=device, verbose=verbose)

    window_scores = score_windows(model, windows_scaled, device=device)
    score_timeline = scores_to_timeline(df, window_scores, start_idx, window)

    train_scores = window_scores[:n_train_windows]
    threshold = np.percentile(train_scores, threshold_pct)

    periods = find_anomaly_periods(score_timeline, threshold, min_duration_steps=3, severity_multiplier=1.5)
    periods_path = os.path.join(out_dir, f"{meter_id}_anomaly_periods.csv")
    periods.to_csv(periods_path, index=False)

    plot_path = None
    if make_plot:
        plot_path = os.path.join(out_dir, f"{meter_id}_anomaly_diagram.png")
        plot_results(df, score_timeline, threshold, periods, signal_col, plot_path, meter_id)

    if save_model:
        torch.save({"model_state": model.state_dict(), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                    "features": features, "window": window}, os.path.join(out_dir, f"{meter_id}_vae.pt"))

    return {"meter_id": meter_id, "n_periods": len(periods), "threshold": threshold,
            "periods_path": periods_path, "plot_path": plot_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to one meter's combined CSV")
    parser.add_argument("--features", nargs="+", default=[
        "pRealKw", "pReactiveKw", "powerFactor", "vRMSMax", "iRMSMax"])
    parser.add_argument("--signal_col", default="pRealKw", help="Feature to plot on top panel")
    parser.add_argument("--window", type=int, default=48, help="Window length in steps (48 * 5min = 4h)")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--threshold_pct", type=float, default=99.5, help="Percentile of training scores for threshold")
    parser.add_argument("--out_dir", default="./out")
    args = parser.parse_args()

    result = run_pipeline(args.csv, args.out_dir, args.features, args.signal_col,
                           args.window, args.stride, args.epochs, args.threshold_pct)
    print(f"\nDetected {result['n_periods']} anomaly period(s). Saved -> {result['periods_path']}")
    if result["plot_path"]:
        print(f"saved diagram -> {result['plot_path']}")


if __name__ == "__main__":
    main()
