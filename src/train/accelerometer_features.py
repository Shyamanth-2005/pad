"""Free-living activity features from a seven-day hip accelerometer (report Alg. 4.1).

Input is one participant's minute-level ActiGraph counts with the NHANES
calibration and reliability flags. A day is valid when every minute is flagged
calibrated and reliable and the device was worn for at least 600 minutes;
participants with fewer than four valid days have no usable recording (m = 0).
For each valid day the ten features of Table 3.2 are computed over worn minutes
only, and the participant's feature vector is their mean over valid days.

Intensity cut-points and the non-wear rule follow Troiano et al. (2008).
"""
import numpy as np

from src.prediction.fusion import SENSOR_FEATURES

SEDENTARY_BELOW = 100       # counts/min; below this a minute is sedentary
MVPA_FROM = 2020            # counts/min; at or above this a minute is MVPA
MVPA_BOUT_MINUTES = 10      # MVPA counted in bouts only when it lasts this long
PEAK_MINUTES = 30           # peak-30: mean of the day's 30 most active minutes
MIN_WEAR_MINUTES = 600      # valid day: at least 10 hours worn
MIN_VALID_DAYS = 4

# Troiano non-wear: 60+ consecutive minutes of zero counts, tolerating
# interruptions of up to 2 consecutive minutes with counts of 1-100.
NONWEAR_MINUTES = 60
NONWEAR_SPIKE_MAX = 100
NONWEAR_SPIKE_MINUTES = 2

MORNING_HOURS = range(6, 12)     # 06:00-11:59
AFTERNOON_HOURS = range(12, 18)  # 12:00-17:59


def wear_mask(counts: np.ndarray) -> np.ndarray:
    """True for minutes the device was worn, by the Troiano non-wear rule."""
    n = len(counts)
    worn = np.ones(n, dtype=bool)
    i = 0
    while i < n:
        if counts[i] != 0:
            i += 1
            continue
        # A candidate non-wear window opens on a zero and closes on the last zero
        # before a count above 100 or a third consecutive non-zero minute.
        start = last_zero = i
        spike = 0
        j = i + 1
        while j < n:
            if counts[j] == 0:
                last_zero, spike = j, 0
            elif counts[j] <= NONWEAR_SPIKE_MAX and spike < NONWEAR_SPIKE_MINUTES:
                spike += 1
            else:
                break
            j += 1
        if last_zero - start + 1 >= NONWEAR_MINUTES:
            worn[start:last_zero + 1] = False
        i = last_zero + 1
    return worn


def run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of the consecutive runs of True in a boolean array."""
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return edges[1::2] - edges[0::2]


def day_features(counts: np.ndarray, worn: np.ndarray, hours: np.ndarray) -> dict:
    """The ten features for one day. Non-worn minutes break every run and transition."""
    worn_counts = counts[worn]
    sedentary = worn & (counts < SEDENTARY_BELOW)
    active = worn & (counts >= SEDENTARY_BELOW)
    mvpa = worn & (counts >= MVPA_FROM)

    # Transitions are read only between two consecutive worn minutes.
    both_worn = worn[:-1] & worn[1:]
    from_active = active[:-1] & both_worn
    from_sedentary = sedentary[:-1] & both_worn
    active_to_sedentary = np.count_nonzero(from_active & sedentary[1:])
    sedentary_to_active = np.count_nonzero(from_sedentary & active[1:])

    mvpa_runs = run_lengths(mvpa)
    active_runs = run_lengths(active)
    morning = counts[worn & np.isin(hours, MORNING_HOURS)].sum()
    afternoon = counts[worn & np.isin(hours, AFTERNOON_HOURS)].sum()

    values = {
        "Log_Total_Counts": np.log1p(worn_counts.sum()),
        "Sedentary_Share": np.mean(worn_counts < SEDENTARY_BELOW),
        "Light_Minutes": np.count_nonzero((worn_counts >= SEDENTARY_BELOW)
                                          & (worn_counts < MVPA_FROM)),
        "MVPA_Minutes": np.count_nonzero(worn_counts >= MVPA_FROM),
        "MVPA_Bout_Minutes": mvpa_runs[mvpa_runs >= MVPA_BOUT_MINUTES].sum(),
        "ASTP": (active_to_sedentary / np.count_nonzero(from_active)
                 if from_active.any() else np.nan),
        "SATP": (sedentary_to_active / np.count_nonzero(from_sedentary)
                 if from_sedentary.any() else np.nan),
        "Peak30_Counts": np.sort(worn_counts)[-PEAK_MINUTES:].mean(),
        "Longest_Active_Bout": active_runs.max() if active_runs.size else 0,
        # +1 keeps a day with no activity in one half finite.
        "Log_PM_AM_Ratio": np.log((afternoon + 1) / (morning + 1)),
    }
    return {name: float(values[name]) for name in SENSOR_FEATURES}


def participant_features(counts: np.ndarray, day_ids: np.ndarray, hours: np.ndarray,
                         reliable: np.ndarray) -> tuple[dict | None, int]:
    """Feature vector averaged over valid days, and the number of valid days.

    Arrays are one participant's minutes in recording order. `day_ids` labels the
    calendar day of each minute; `reliable` is True where the minute is flagged
    both calibrated and reliable. Returns (None, n_valid) when n_valid < 4.
    """
    worn = wear_mask(counts)
    days = []
    for day in np.unique(day_ids):
        in_day = day_ids == day
        if reliable[in_day].all() and np.count_nonzero(worn[in_day]) >= MIN_WEAR_MINUTES:
            days.append(day_features(counts[in_day], worn[in_day], hours[in_day]))
    if len(days) < MIN_VALID_DAYS:
        return None, len(days)

    averaged = {}
    for name in SENSOR_FEATURES:
        per_day = np.array([day[name] for day in days])
        finite = per_day[~np.isnan(per_day)]
        # A transition probability is undefined on a day with no active minutes;
        # if that holds for every day the value is left missing for the pipeline.
        averaged[name] = float(finite.mean()) if finite.size else np.nan
    return averaged, len(days)
