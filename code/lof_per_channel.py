import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

august_folder = "august"

features = [
    "powerFactor",
    "pRealKw",
    "pReactiveKw",
    "vRMSMin",
    "vRMSMax",
    "iRMSMin",
    "iRMSMax"

all_files = glob.glob(
    os.path.join(august_folder, "*_08_2026", "*.csv")
)

channels = {}

for file in all_files:
    channel = os.path.basename(file)

    if channel not in channels:
        channels[channel] = []

    channels[channel].append(file)

all_anomalies = []

for channel, files in channels.items():

    channel_data = []
    for file in files:
        df = pd.read_csv(file)
        df["source_file"] = file
        channel_data.append(df)
    df_channel = pd.concat(
        channel_data,
        ignore_index=True
    )


    X = df_channel[features].copy()
    # Remove rows with missing feature values
    valid_rows = X.notna().all(axis=1)
    X_valid = X.loc[valid_rows]
    if len(X_valid) < 21:
        print(f"Skipping {channel}: insufficient data")
        continue

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_valid)

    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.005
    )

    labels = lof.fit_predict(X_scaled)

    scores = -lof.negative_outlier_factor_

    results = df_channel.loc[valid_rows].copy()

    results["channel"] = channel.replace(".csv", "")
    results["lof_score"] = scores
    results["lof_label"] = labels

    anomalies = results[
        results["lof_label"] == -1
    ].copy()

    all_anomalies.append(anomalies)

    print(
        channel,
        "| rows:", len(results),
        "| anomalies:", len(anomalies)
    )
# Combine all anomalies

all_anomalies = pd.concat(
    all_anomalies,
    ignore_index=True
)

all_anomalies.to_csv(
    "august_lof_anomalies_per_channel.csv",
    index=False
)


df = pd.read_csv("august_lof_anomaly_events_15min.csv")

df["start_time"] = pd.to_datetime(df["start_time"])
df["end_time"] = pd.to_datetime(df["end_time"])

# Extract useful time variables
df["hour"] = df["start_time"].dt.hour
df["date"] = df["start_time"].dt.date

hourly = (
    df.groupby("hour")
      .size()
      .reindex(range(24), fill_value=0)
)

plt.figure(figsize=(10, 5))
plt.bar(hourly.index, hourly.values)

plt.xlabel("Hour of day")
plt.ylabel("Number of anomaly events")
plt.title("≥15-minute anomaly events by start hour")
plt.xticks(range(24))

plt.tight_layout()
plt.savefig("01_anomalies_by_hour.png", dpi=300)
plt.close()

daily = df.groupby("date").size()

plt.figure(figsize=(12, 5))
plt.plot(range(1, len(daily) + 1), daily.values, marker="o")

plt.xlabel("Day of July")
plt.ylabel("Number of anomaly events")
plt.title("≥15-minute anomaly events by day")
plt.xticks(range(1, len(daily) + 1))

plt.tight_layout()
plt.savefig("02_anomalies_by_day.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 5))

plt.hist(
    df["duration_minutes"],
    bins=30,
    edgecolor="black"
)

plt.xlabel("Duration (minutes)")
plt.ylabel("Number of anomaly events")
plt.title("Distribution of anomaly event duration")

plt.tight_layout()
plt.savefig("03_anomaly_duration.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 5))

plt.scatter(
    df["duration_minutes"],
    df["max_lof_score"],
    alpha=0.5
)

plt.xlabel("Event duration (minutes)")
plt.ylabel("Maximum LOF score")
plt.title("Anomaly severity vs duration")

plt.tight_layout()
plt.savefig("04_lof_vs_duration.png", dpi=300)
plt.close()


print("All plots saved!")


df["start_time"] = pd.to_datetime(df["start_time"])
df["hour"] = df["start_time"].dt.hour

lof_by_hour = (
    df.groupby("hour")["max_lof_score"]
      .mean()
      .reindex(range(24))
)

plt.figure(figsize=(10, 5))

plt.bar(lof_by_hour.index, lof_by_hour.values)

plt.xlabel("Start hour")
plt.ylabel("Average maximum LOF score")
plt.title("Average anomaly severity by start hour")

plt.xticks(range(24))

plt.tight_layout()
plt.savefig("07_lof_severity_by_hour.png", dpi=300)
plt.close()