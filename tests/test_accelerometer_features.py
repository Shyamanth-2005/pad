import numpy as np
import pytest

from src.prediction.fusion import SENSOR_FEATURES
from src.train.accelerometer_features import (day_features, participant_features,
                                              run_lengths, wear_mask)

MINUTES_PER_DAY = 1440


def test_run_lengths():
    mask = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
    assert run_lengths(mask).tolist() == [2, 1, 3]
    assert run_lengths(np.zeros(4, dtype=bool)).size == 0


def test_sixty_zeros_are_non_wear_but_fifty_nine_are_not():
    counts = np.full(200, 500)
    counts[10:70] = 0      # 60 zeros
    counts[100:159] = 0    # 59 zeros
    worn = wear_mask(counts)
    assert not worn[10:70].any()
    assert worn[100:159].all()
    assert worn[:10].all() and worn[70:100].all()


def test_non_wear_tolerates_two_minute_spikes_up_to_100():
    counts = np.full(200, 500)
    counts[20:120] = 0
    counts[50:52] = [40, 100]      # tolerated interruption
    assert not wear_mask(counts)[20:120].any()

    counts[50:53] = [40, 100, 30]  # a third non-zero minute ends the window
    worn = wear_mask(counts)
    assert worn[20:50].all()       # only 30 zeros before the break
    assert not worn[53:120].any()  # 67 zeros after it


def test_spike_above_100_breaks_non_wear():
    counts = np.full(200, 500)
    counts[20:130] = 0
    counts[60] = 101
    worn = wear_mask(counts)
    assert worn[20:60].all()       # 40 zeros before the spike
    assert not worn[61:130].any()  # 69 zeros after it


def test_day_features_on_a_hand_built_day():
    counts = np.zeros(MINUTES_PER_DAY, dtype=int)
    hours = np.repeat(np.arange(24), 60)
    worn = np.zeros(MINUTES_PER_DAY, dtype=bool)
    worn[6 * 60:18 * 60] = True             # 12 h worn, 06:00-17:59
    counts[6 * 60:18 * 60] = 50             # sedentary baseline
    counts[8 * 60:8 * 60 + 12] = 3000       # 12-min MVPA bout in the morning
    counts[13 * 60:13 * 60 + 5] = 500       # 5-min light bout in the afternoon

    f = day_features(counts, worn, hours)
    assert set(f) == set(SENSOR_FEATURES)
    assert f["MVPA_Minutes"] == 12
    assert f["MVPA_Bout_Minutes"] == 12
    assert f["Light_Minutes"] == 5
    assert f["Sedentary_Share"] == pytest.approx((720 - 17) / 720)
    assert f["Longest_Active_Bout"] == 12
    assert f["Peak30_Counts"] == pytest.approx((12 * 3000 + 5 * 500 + 13 * 50) / 30)
    # Two active bouts, each ending in one active->sedentary transition.
    assert f["ASTP"] == pytest.approx(2 / 17)
    # The last worn minute has no worn successor, so 702 sedentary starts.
    assert f["SATP"] == pytest.approx(2 / (720 - 17 - 1))
    total = 703 * 50 + 12 * 3000 + 5 * 500
    assert f["Log_Total_Counts"] == pytest.approx(np.log1p(total))
    morning = 6 * 60 * 50 - 12 * 50 + 12 * 3000
    afternoon = 6 * 60 * 50 - 5 * 50 + 5 * 500
    assert f["Log_PM_AM_Ratio"] == pytest.approx(np.log((afternoon + 1) / (morning + 1)))


def test_short_mvpa_runs_do_not_count_as_bouts():
    counts = np.full(MINUTES_PER_DAY, 50)
    counts[100:109] = 2500    # 9 minutes: MVPA but not a bout
    f = day_features(counts, np.ones(MINUTES_PER_DAY, dtype=bool),
                     np.repeat(np.arange(24), 60))
    assert f["MVPA_Minutes"] == 9
    assert f["MVPA_Bout_Minutes"] == 0


def _recording(days: int, worn_minutes: int):
    """`days` calendar days, each worn for the first `worn_minutes` minutes."""
    counts = np.zeros(days * MINUTES_PER_DAY, dtype=int)
    for day in range(days):
        start = day * MINUTES_PER_DAY
        counts[start:start + worn_minutes] = 150
    day_ids = np.repeat(np.arange(days), MINUTES_PER_DAY)
    hours = np.tile(np.repeat(np.arange(24), 60), days)
    return counts, day_ids, hours, np.ones_like(counts, dtype=bool)


def test_four_valid_days_are_required():
    features, valid = participant_features(*_recording(days=4, worn_minutes=600))
    assert valid == 4 and features is not None

    features, valid = participant_features(*_recording(days=4, worn_minutes=599))
    assert valid == 0 and features is None


def test_unreliable_days_are_dropped():
    counts, day_ids, hours, reliable = _recording(days=5, worn_minutes=700)
    reliable[day_ids == 0] = False
    reliable[np.flatnonzero(day_ids == 1)[0]] = False   # one bad minute voids the day
    features, valid = participant_features(counts, day_ids, hours, reliable)
    assert valid == 3 and features is None
