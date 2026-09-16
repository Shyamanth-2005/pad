"""Ankle-brachial index interpretation.

The ABI is measured, not predicted: the clinical pipelines in this project are
trained without it. It is supplied by the user purely to grade severity.
Bands follow the standard reporting thresholds.
"""

ABI_MIN = 0.0
ABI_MAX = 3.0

# (upper_bound_inclusive, key, label, detail) - evaluated low to high.
_BANDS = [
    (0.40, "severe", "Severe PAD",
     "High risk of significant arterial blockages. Refer urgently."),
    (0.90, "moderate", "Mild to moderate PAD",
     "Reduced flow consistent with peripheral arterial disease."),
    (0.99, "borderline", "Borderline",
     "Just under normal. May suggest early changes."),
    (1.40, "normal", "Normal",
     "Normal blood flow and pressure."),
]
_ABOVE = ("noncompressible", "Non-compressible",
          "Stiff or calcified arteries, often seen in diabetes. "
          "ABI is unreliable here - consider toe-brachial index.")


def classify_abi(value: float) -> dict:
    """Grade a measured ABI value into its reporting band."""
    for upper, key, label, detail in _BANDS:
        if value <= upper:
            return {"value": value, "band": key, "label": label, "detail": detail}
    key, label, detail = _ABOVE
    return {"value": value, "band": key, "label": label, "detail": detail}


# The scale in the interface is drawn from 0 to this value. Past it the ABI is
# uninterpretable anyway, so the top band simply runs to the end.
SCALE_MAX = 1.6


def band_reference() -> list[dict]:
    """The full band table, for rendering the scale in the interface.

    `span` is the band's width on the drawn scale, measured edge to edge rather
    than from the printed range. The printed ranges leave a 0.01 gap at each
    boundary (0.40 then 0.41); summing those would make the track 1.56 wide
    while the needle is positioned against 1.60, drifting every marker left.
    """
    rows = []
    low = edge = ABI_MIN
    for upper, key, label, detail in _BANDS:
        rows.append({"band": key, "label": label, "detail": detail,
                     "low": low, "high": upper, "span": round(upper - edge, 2),
                     "range": f"{low:.2f} - {upper:.2f}"})
        low, edge = round(upper + 0.01, 2), upper
    key, label, detail = _ABOVE
    rows.append({"band": key, "label": label, "detail": detail,
                 "low": 1.41, "high": ABI_MAX, "span": round(SCALE_MAX - edge, 2),
                 "range": "Above 1.40"})
    return rows
