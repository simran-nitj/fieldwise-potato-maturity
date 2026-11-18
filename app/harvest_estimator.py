from datetime import date, timedelta


STAGE_NAMES = ["Vegetative", "Tuber Bulking", "Maturity"]


def estimate_harvest(planting_date, distribution, assumed_duration_days=None, duration_margin_days=14):
    """Transparent provisional calendar estimate; this is not a trained harvest model."""
    if not planting_date or not isinstance(distribution, dict):
        return {"available": False, "message": "Insufficient information for harvest estimation."}
    try:
        planted = date.fromisoformat(str(planting_date))
    except (TypeError, ValueError):
        return {"available": False, "message": "Enter a valid planting date to estimate a harvest window."}

    counts = {name: max(0, int(distribution.get(name, 0) or 0)) for name in STAGE_NAMES}
    total = sum(counts.values())
    if not total:
        return {"available": False, "message": "Insufficient information for harvest estimation: no growth-stage detections were observed."}
    try:
        duration = int(assumed_duration_days)
    except (TypeError, ValueError):
        return {"available": False, "message": "Choose an assumed crop duration to calculate a provisional window."}
    if not 60 <= duration <= 180:
        return {"available": False, "message": "Assumed crop duration must be between 60 and 180 days."}

    shares = {name: counts[name] / total for name in STAGE_NAMES}
    # The stage mix shifts the calendar midpoint conservatively by at most 21 days.
    maturity_signal = shares["Maturity"] + 0.5 * shares["Tuber Bulking"]
    stage_adjustment = round((0.5 - maturity_signal) * 42)
    midpoint = planted + timedelta(days=duration + stage_adjustment)
    margin = max(1, min(int(duration_margin_days), 30))
    earliest, latest = midpoint - timedelta(days=margin), midpoint + timedelta(days=margin)
    return {
        "available": True,
        "earliest": earliest.isoformat(),
        "latest": latest.isoformat(),
        "midpoint": midpoint.isoformat(),
        "assumed_duration_days": duration,
        "margin_days": margin,
        "stage_adjustment_days": stage_adjustment,
        "explanation": (
            f"Starting from planting date + {duration} assumed days, the observed stage mix "
            f"adjusted the midpoint by {stage_adjustment:+d} days; a ±{margin}-day planning margin was applied."
        ),
        "disclaimer": "Provisional planning estimate only. Validate crop duration, field conditions, and readiness with an agronomist; YOLO detects visible growth stages and does not predict days to harvest.",
    }
