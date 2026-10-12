"""Lossless review triage, not a fault classifier or replacement score.

Only frozen historical envelopes and observed five-minute evidence may admit
a priority interval. Percentile saturation alone is insufficient. Source event
IDs, scores, intervals and confidence labels are retained without alteration.
"""
from dataclasses import asdict, dataclass
from hashlib import sha1
import json

import numpy as np
import pandas as pd


QUEUES = ("priority_alert", "operating_change_review", "data_quality")


@dataclass(frozen=True)
class AlertTriageConfig:
    minimum_effect_kw: float = .5
    relative_effect: float = .10
    peak_effect_multiplier: float = 2.
    minimum_duration_minutes: int = 15
    minimum_reference_days: int = 6
    minimum_observed_ratio: float = .95
    current_confirmation_fraction: float = .5
    joint_percentile: float = 99.
    merge_gap_minutes: int = 0
    slot_radius_points: int = 2
    envelope_quantile: float = .01
    jump_quantile: float = .995

    def __post_init__(self):
        if self.minimum_effect_kw <= 0 or self.relative_effect <= 0:
            raise ValueError("Effect guards must be positive")
        if self.peak_effect_multiplier < 1 or self.minimum_duration_minutes < 5:
            raise ValueError("Invalid severity or duration guard")
        if self.minimum_duration_minutes % 5 or self.minimum_reference_days < 2:
            raise ValueError("Duration must use five-minute units and references need support")
        if not 0 < self.minimum_observed_ratio <= 1:
            raise ValueError("Invalid observation ratio")
        if not 0 < self.current_confirmation_fraction <= 1 or not 0 < self.joint_percentile <= 100:
            raise ValueError("Invalid confirmation guard")
        if self.merge_gap_minutes < 0 or not 0 <= self.slot_radius_points <= 6:
            raise ValueError("Invalid merge gap or historical jitter")
        if not 0 <= self.envelope_quantile < .5 or not 0 < self.jump_quantile <= 1:
            raise ValueError("Invalid historical quantiles")


def _number(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _json(value, fallback):
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _runs(mask, index):
    start = None
    for position in range(len(mask) + 1):
        flagged = position < len(mask) and mask[position]
        adjacent = position == 0 or (position < len(mask) and
                     index[position] - index[position - 1] == pd.Timedelta(minutes=5))
        if start is not None and (not flagged or not adjacent):
            yield start, position
            start = None
        if flagged and start is None:
            start = position


def interval_mask(frame, index, step_minutes=5, queue=None, priority_cores=False):
    """Half-open interval overlap, recalculated on the requested observation grid."""
    mask = np.zeros(len(index), dtype=bool)
    if queue is not None:
        frame = frame.loc[frame.review_queue.eq(queue)]
    for row in frame.to_dict("records"):
        intervals = (_json(row.get("priority_segments"), []) if priority_cores else
                     [dict(start_time=row["start_time"], end_time=row["end_time"])])
        for interval in intervals:
            start, end = pd.Timestamp(interval["start_time"]), pd.Timestamp(interval["end_time"])
            mask |= (index < end) & (index + pd.Timedelta(minutes=step_minutes) > start)
    return mask


def event_evidence(event, target, reference, policy=None, quality_mask=None):
    """Measure real point evidence; ground-truth labels are never consulted."""
    policy = policy or AlertTriageConfig()
    start, end = pd.Timestamp(event["start_time"]), pd.Timestamp(event["end_time"])
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError("Source events require valid timezone-aware half-open intervals")
    if len(target) != 288 or target.index.has_duplicates or not target.index.is_monotonic_increasing:
        raise ValueError("Triage requires a unique sorted 288-slot five-minute day")
    if not (np.diff(target.index.asi8) == pd.Timedelta(minutes=5).value).all():
        raise ValueError("Target day must be a five-minute grid")
    if max(reference.dates) >= start.date():
        raise ValueError("Triage calibration must precede the event")
    if start < target.index[0] or end > target.index[-1] + pd.Timedelta(minutes=5):
        raise ValueError("Source event extends outside its target day")
    index = target.index
    inside = (index < end) & (index + pd.Timedelta(minutes=5) > start)
    power = target.pRealKw.to_numpy(dtype=float)
    supported = (np.isfinite(power) & np.isfinite(reference.lower_kw) & np.isfinite(reference.upper_kw)
                 & (reference.support_days >= policy.minimum_reference_days))
    if quality_mask is not None:
        excluded = np.asarray(quality_mask, dtype=bool)
        if excluded.shape != supported.shape:
            raise ValueError("Quality exclusion shape mismatch")
        supported &= ~excluded
    excess = np.maximum(power - reference.upper_kw, reference.lower_kw - power)
    outside = inside & supported & (excess > reference.effect_kw)
    jumps = np.r_[np.nan, np.abs(np.diff(power))]
    current = np.zeros(len(index), dtype=bool)
    if reference.current_lower is not None and "iRMSMax" in target:
        values = target.iRMSMax.to_numpy(dtype=float)
        current_excess = np.maximum(values - reference.current_upper, reference.current_lower - values)
        current = (np.isfinite(values) & np.isfinite(current_excess)
                   & (current_excess > reference.current_effect))
    short, medium = _number(event.get("short_percentile")), _number(event.get("medium_percentile"))
    multiscale = (short is not None and medium is not None
                  and short >= policy.joint_percentile and medium >= policy.joint_percentile)
    cores, longest, has_peak, has_onset, has_confirmation = [], 0, False, False, False
    directional_runs = []
    for sign, sign_mask in (("up", outside & (power > reference.upper_kw)),
                            ("down", outside & (power < reference.lower_kw))):
        directional_runs.extend((first, stop, sign) for first, stop in _runs(sign_mask, index))
    for first, stop, sign in sorted(directional_runs):
        duration = (stop - first) * 5
        longest = max(longest, duration)
        peak = float(np.max(excess[first:stop]))
        onset = (first > 0 and supported[first-1] and np.isfinite(jumps[first])
                 and jumps[first] > reference.jump_kw)
        current_fraction = float(current[first:stop].mean())
        confirmation = multiscale or current_fraction >= policy.current_confirmation_fraction
        if duration >= policy.minimum_duration_minutes:
            has_peak |= peak >= policy.peak_effect_multiplier * reference.effect_kw
            has_onset |= bool(onset)
            has_confirmation |= confirmation
            # All guards must occur in the SAME continuous point run.
            if peak >= policy.peak_effect_multiplier * reference.effect_kw and onset and confirmation:
                cores.append(dict(start_time=index[first].isoformat(),
                                  end_time=(index[stop - 1] + pd.Timedelta(minutes=5)).isoformat(),
                                  duration_minutes=duration, peak_excess_kw=peak,
                                  onset_jump_kw=float(jumps[first]),
                                  current_confirmation_fraction=current_fraction,
                                  multiscale_agreement=bool(multiscale), direction=sign))
    observed_ratio = float(supported[inside].mean()) if inside.any() else 0.
    if observed_ratio < policy.minimum_observed_ratio:
        cores = []
    difference = power[inside & supported] - reference.median_kw[inside & supported]
    signed = float(np.nanmedian(difference)) if np.isfinite(difference).any() else None
    direction = "unknown" if signed is None else "up" if signed > 0 else "down" if signed < 0 else "flat"
    if cores:
        directions = {core["direction"] for core in cores}
        direction = next(iter(directions)) if len(directions) == 1 else "mixed"
    reasons = []
    if observed_ratio < policy.minimum_observed_ratio: reasons.append("insufficient_observed_point_support")
    if longest < policy.minimum_duration_minutes: reasons.append("no_sustained_historical_envelope_exceedance")
    if not has_peak: reasons.append("insufficient_peak_power_effect")
    if not has_onset: reasons.append("no_abrupt_onset_above_frozen_jump_guard")
    if not has_confirmation: reasons.append("no_multiscale_or_current_confirmation")
    if not cores and not reasons: reasons.append("guards_not_coincident_in_one_point_run")
    return dict(reference_end=str(max(reference.dates)), reference_days=len(reference.dates),
                effect_guard_kw=float(reference.effect_kw), jump_guard_kw=float(reference.jump_kw),
                observed_point_ratio=observed_ratio, outside_envelope_5m_points=int(outside.sum()),
                longest_outside_run_minutes=longest,
                peak_outside_envelope_kw=float(excess[outside].max()) if outside.any() else 0.,
                direction=direction, multiscale_agreement=bool(multiscale),
                priority_segments=json.dumps(cores),
                priority_core_minutes=sum(core["duration_minutes"] for core in cores),
                review_queue="priority_alert" if cores else "operating_change_review",
                triage_reasons=json.dumps(["sustained_effect_abrupt_onset_and_confirmation"] if cores else reasons))


def classify_events(events, targets, references, policy=None, unavailable_reason=None):
    """Partition source events once, retaining every original column and ID."""
    policy = policy or AlertTriageConfig()
    if events.event_id.isna().any() or events.event_id.duplicated().any():
        raise ValueError("Source event IDs must be non-null and unique")
    classified = []
    quality = events.loc[events.event_type.eq("data_quality")]
    for event in events.to_dict("records"):
        day = pd.Timestamp(event["start_time"]).date()
        evidence = _json(event.get("evidence"), {})
        result = dict(triage_policy="frozen_point_evidence_v1", priority_segments="[]",
                      priority_core_minutes=0, direction="unknown",
                      quality_issue=evidence.get("quality_issue", "unspecified"))
        if event["event_type"] == "data_quality":
            result.update(review_queue="data_quality", triage_reasons='["source_data_quality_event"]')
        elif unavailable_reason or day not in targets or day not in references:
            result.update(review_queue="operating_change_review", triage_reasons=json.dumps(
                [unavailable_reason or "missing_target_or_frozen_reference"]))
        else:
            excluded = interval_mask(quality, targets[day].index)
            result.update(event_evidence(event, targets[day], references[day], policy, excluded))
        classified.append({**event, **result})
    frame = pd.DataFrame(classified)
    if frame.empty:
        frame = events.copy()
        for column in ("review_queue", "triage_reasons", "priority_segments", "priority_core_minutes",
                       "direction", "quality_issue", "triage_policy"):
            frame[column] = pd.Series(dtype=object)
    # Triage is not allowed to rewrite scores, IDs, time ranges or source evidence.
    pd.testing.assert_frame_equal(frame[list(events.columns)].reset_index(drop=True), events.reset_index(drop=True))
    return frame


def build_review_units(classified, policy=None):
    """Merge adjacent same-channel episodes and existing common-mode review links.

Queues, directions, event types and quality subtypes never mix. Common-mode
links are review convenience, not proof of a single physical site fault.
"""
    policy = policy or AlertTriageConfig()
    if classified.event_id.duplicated().any():
        raise ValueError("Duplicate classified event ID")
    rows = classified.to_dict("records")
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i, j):
        parents[find(j)] = find(i)

    groups = {}
    for i, row in enumerate(rows):
        row["series_id"] = f"{row['device_id']}_{int(row['channel_idx'])}"
        cores = _json(row.get("priority_segments"), [])
        row["effective_start"] = min(pd.Timestamp(core["start_time"]) for core in cores) if cores else pd.Timestamp(row["start_time"])
        row["effective_end"] = max(pd.Timestamp(core["end_time"]) for core in cores) if cores else pd.Timestamp(row["end_time"])
        partition = (row["review_queue"], row["direction"], row["event_type"], row["quality_issue"])
        groups.setdefault(("channel", row["series_id"], *partition), []).append(i)
        incident = str(row.get("incident_id", ""))
        if incident and incident not in ("nan", "None") and not incident.startswith("event-"):
            groups.setdefault(("common", incident, row.get("organization"), row.get("category_id"), *partition), []).append(i)
    for key, members in groups.items():
        ordered = sorted(members, key=lambda i: rows[i]["effective_start"])
        for pos, i in enumerate(ordered):
            allowance = pd.Timedelta(minutes=policy.merge_gap_minutes if key[0] == "channel" else 0)
            for j in ordered[pos + 1:]:
                if rows[j]["effective_start"] > rows[i]["effective_end"] + allowance:
                    break
                # Existing common-mode clusters must still actually overlap.
                if key[0] != "common" or rows[j]["effective_start"] < rows[i]["effective_end"]:
                    union(i, j)
    clusters = {}
    for i in range(len(rows)):
        clusters.setdefault(find(i), []).append(rows[i])
    units, membership = [], []
    for members in clusters.values():
        ids = sorted(member["event_id"] for member in members)
        queue = members[0]["review_queue"]
        unit_id = "triage-" + sha1((queue + "|" + "|".join(ids)).encode()).hexdigest()[:16]
        start = min(member["effective_start"] for member in members)
        end = max(member["effective_end"] for member in members)
        intervals = []
        for member in members:
            cores = _json(member.get("priority_segments"), [])
            intervals.extend([(pd.Timestamp(c["start_time"]), pd.Timestamp(c["end_time"])) for c in cores]
                             if cores else [(member["effective_start"], member["effective_end"])])
        intervals.sort()
        has_gap, last_end = False, intervals[0][1]
        for begin, finish in intervals[1:]:
            has_gap |= begin > last_end
            last_end = max(last_end, finish)
        series = sorted(set(member["series_id"] for member in members))
        units.append(dict(review_unit_id=unit_id, review_queue=queue, start_time=start.isoformat(),
                          end_time=end.isoformat(), span_minutes=int((end-start).total_seconds()/60),
                          contains_gaps=bool(has_gap), source_event_count=len(ids), channel_count=len(series),
                          series_ids=json.dumps(series), event_ids=json.dumps(ids),
                          organization=members[0].get("organization"), category_name=members[0].get("category_name"),
                          event_type=members[0]["event_type"], direction=members[0]["direction"],
                          grouping_kind="common_mode_review" if len(series)>1 else "same_channel_episode"))
        membership.extend(dict(event_id=member["event_id"], series_id=member["series_id"],
                               review_unit_id=unit_id, review_queue=queue) for member in members)
    unit_columns = ["review_unit_id", "review_queue", "start_time", "end_time", "span_minutes", "contains_gaps",
                    "source_event_count", "channel_count", "series_ids", "event_ids", "organization", "category_name",
                    "event_type", "direction", "grouping_kind"]
    return pd.DataFrame(units, columns=unit_columns), pd.DataFrame(membership, columns=[
        "event_id", "series_id", "review_unit_id", "review_queue"])


def config_dict(policy):
    return asdict(policy)
