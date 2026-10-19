"""
Build the NHANES 2003-2004 sensor cohort (report Sections 3.3, 4.4, 4.5).

Downloads the public NHANES files into src/data/nhanes_raw/ (git-ignored; the
accelerometer archive alone is 427 MB), then:
  1. Cohort  - examined adults aged 40+ with an ABI in at least one leg, excluding
               anyone with ABI > 1.40 in either leg; PAD = minimum ABI <= 0.90.
  2. Clinical - the 13 variables coded exactly as in PAD_dataset_50000.csv, so the
               frozen clinical ensemble can score these participants.
  3. Sensor  - the ten activity features from the ActiGraph minute counts, with
               the availability mask (>= 4 valid days of >= 10 h wear).

Writes src/data/PAD_nhanes_2003_2004_sensor.csv. Expected: 2,455 participants,
246 PAD; 1,919 with a valid recording (report Section 3.3.2).
"""
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.prediction.fusion import SENSOR_FEATURES  # noqa: E402
from src.train.accelerometer_features import participant_features  # noqa: E402

RAW = ROOT / "src" / "data" / "nhanes_raw"
OUT = ROOT / "src" / "data" / "PAD_nhanes_2003_2004_sensor.csv"
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/"

COMPONENTS = ["DEMO_C", "LEXAB_C", "BMX_C", "BPX_C", "L13_C", "L10_C",
              "SMQ_C", "DIQ_C", "BPQ_C"]
ACCELEROMETER = "PAXRAW_C.zip"

RACE = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
        4: "Non-Hispanic Black", 5: "Other/Multi-racial"}
SEX = {1: "Male", 2: "Female"}
DIABETES = {1: "Yes", 2: "No", 3: "Borderline"}
YES_NO = {1: "Yes", 2: "No"}


def download(filename: str) -> Path:
    path = RAW / filename
    if path.exists():
        return path
    RAW.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    print(f"downloading {filename}", flush=True)
    urllib.request.urlretrieve(BASE_URL + filename, partial)
    partial.rename(path)
    return path


def read_xpt(component: str) -> pd.DataFrame:
    frame = pd.read_sas(download(f"{component}.xpt"), format="xport")
    frame["SEQN"] = frame["SEQN"].astype(int)
    return frame.set_index("SEQN")


def smoking_status(smq: pd.DataFrame) -> pd.Series:
    """Never = under 100 cigarettes in life; otherwise current or former by SMQ040."""
    status = pd.Series(np.nan, index=smq.index, dtype=object)
    status[smq["SMQ020"] == 2] = "Never"
    status[(smq["SMQ020"] == 1) & smq["SMQ040"].isin([1, 2])] = "Current"
    status[(smq["SMQ020"] == 1) & (smq["SMQ040"] == 3)] = "Former"
    return status


def build_cohort() -> pd.DataFrame:
    demo, abi, bmx, bpx, lipids, hba1c, smq, diq, bpq = (read_xpt(c) for c in COMPONENTS)

    cohort = demo[["RIDAGEYR", "RIAGENDR", "RIDRETH1"]].join(
        abi[["LEXLABPI", "LEXRABPI"]], how="inner")
    cohort = cohort[cohort["RIDAGEYR"] >= 40]
    cohort = cohort[cohort[["LEXLABPI", "LEXRABPI"]].notna().any(axis=1)]
    cohort = cohort[~(cohort[["LEXLABPI", "LEXRABPI"]] > 1.40).any(axis=1)]

    # DBP of 0 is a recording artefact in NHANES, not a pressure.
    diastolic = bpx[["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]].where(lambda v: v > 0)
    frame = pd.DataFrame(index=cohort.index)
    frame["Age"] = cohort["RIDAGEYR"]
    frame["Sex"] = cohort["RIAGENDR"].map(SEX)
    frame["Race"] = cohort["RIDRETH1"].map(RACE)
    frame["BMI"] = bmx["BMXBMI"]
    frame["Smoking"] = smoking_status(smq)
    frame["Diabetes"] = diq["DIQ010"].map(DIABETES)
    frame["HbA1c"] = hba1c["LBXGH"]
    frame["SBP"] = bpx[["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]].mean(axis=1).round(1)
    frame["DBP"] = diastolic.mean(axis=1).round(1)
    frame["Total_Cholesterol"] = lipids["LBXTC"]
    frame["HDL"] = lipids["LBXHDD"]
    frame["TC_HDL_Ratio"] = (frame["Total_Cholesterol"] / frame["HDL"]).round(2)
    frame["Medical_History_Hypertension"] = bpq["BPQ020"].map(YES_NO)
    frame["ABI_Left"] = cohort["LEXLABPI"]
    frame["ABI_Right"] = cohort["LEXRABPI"]
    frame["ABI_Min"] = cohort[["LEXLABPI", "LEXRABPI"]].min(axis=1)
    frame["PAD"] = (frame["ABI_Min"] <= 0.90).astype(int)
    return frame


def accelerometer_minutes(participants: set[int]) -> pd.DataFrame:
    """Minute rows of PAXRAW_C for the cohort only, streamed from the zip archive.

    The full file is 72 million rows; filtering per chunk keeps memory small.
    """
    archive = zipfile.ZipFile(download(ACCELEROMETER))
    member = archive.infolist()[0].filename
    columns = ["SEQN", "PAXN", "PAXDAY", "PAXHOUR", "PAXSTAT", "PAXCAL", "PAXINTEN"]
    kept = []
    started = time.time()
    with archive.open(member) as stream:
        reader = pd.read_sas(stream, format="xport", chunksize=1_000_000)
        for index, chunk in enumerate(reader, start=1):
            # XPORT stores 0 as 5.4e-79; rounding restores the integer counts.
            chunk = chunk[columns].round().astype(np.int32)
            kept.append(chunk[chunk["SEQN"].isin(participants)])
            if index % 10 == 0:
                print(f"  {index:3d}M rows read ({time.time() - started:.0f}s)", flush=True)
    return pd.concat(kept, ignore_index=True)


def sensor_features(minutes: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    rows = {}
    for seqn, record in minutes.groupby("SEQN", sort=False):
        record = record.sort_values("PAXN")
        day_of_week = record["PAXDAY"].to_numpy()
        # A new calendar day starts whenever the day-of-week code changes.
        day_ids = np.concatenate(([0], np.cumsum(day_of_week[1:] != day_of_week[:-1])))
        reliable = ((record["PAXSTAT"] == 1) & (record["PAXCAL"] == 1)).to_numpy()
        features, valid_days = participant_features(
            record["PAXINTEN"].to_numpy(), day_ids, record["PAXHOUR"].to_numpy(), reliable)
        rows[seqn] = {"Sensor_Valid_Days": valid_days,
                      "Sensor_Available": int(features is not None),
                      **(features or {name: np.nan for name in SENSOR_FEATURES})}
    table = pd.DataFrame.from_dict(rows, orient="index").reindex(index)
    # Participants who never wore a device have no rows at all.
    table["Sensor_Valid_Days"] = table["Sensor_Valid_Days"].fillna(0).astype(int)
    table["Sensor_Available"] = table["Sensor_Available"].fillna(0).astype(int)
    return table


def main() -> None:
    cohort = build_cohort()
    print(f"cohort {len(cohort):,} | PAD {cohort.PAD.sum()} ({cohort.PAD.mean():.3f})")
    minutes = accelerometer_minutes(set(cohort.index))
    print(f"accelerometer rows kept {len(minutes):,} for "
          f"{minutes.SEQN.nunique():,} participants", flush=True)
    sensors = sensor_features(minutes, cohort.index)
    dataset = cohort.join(sensors)
    dataset.index.name = "SEQN"
    dataset.round(4).to_csv(OUT)
    available = dataset["Sensor_Available"] == 1
    print(f"valid recordings {available.sum():,} ({available.mean():.3f}); "
          f"PAD among them {dataset.loc[available, 'PAD'].sum()}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
