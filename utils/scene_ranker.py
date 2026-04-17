"""Scene ranking and adaptive selection.

Public API (unchanged):
    rank_scenes, select_top_scenes, select_by_threshold,
    adaptive_select_scenes, restore_timeline_order,
    get_ranked_scenes, save_selected_scenes, extract_scene_ids

Key improvements over the previous version
-------------------------------------------
* ``adaptive_select_scenes`` uses count-aware bounds:

      min_keep = max(8, ceil(n * 0.10), ceil(watched / 240))
      max_keep = min(35, max(min_keep, ceil(n * 0.20), ceil(watched / 90), 12))

  where n = total scenes passed in.

* Timeline-window coverage:
  - Splits chronological scene IDs into up to 8 equal windows.
  - Reserves the highest-scoring scene from each non-empty window first.
  - Fills remaining slots from the globally-ranked list (no duplicates).
  - Final output is always sorted by scene_id (chronological).
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional


# ── helpers ───────────────────────────────────────────────────────────────────

def load_scene_scores(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def rank_scenes(scene_data: list) -> list:
    """Return scene list sorted by final score descending."""
    return sorted(scene_data, key=lambda x: x["final"], reverse=True)


def select_top_scenes(ranked_scenes: list, top_n: int = 5) -> list:
    """Return the top-n scenes (backward-compat helper)."""
    return ranked_scenes[:top_n]


def select_by_threshold(ranked_scenes: list, threshold: float = 0.5) -> list:
    """Return all scenes whose final score ≥ threshold."""
    return [s for s in ranked_scenes if s["final"] >= threshold]


# ── timeline coverage ─────────────────────────────────────────────────────────

def _timeline_coverage(
    ranked_scenes: list,
    n_windows: int = 8,
) -> List[Dict]:
    """Return one representative scene per chronological window.

    Scenes in ``ranked_scenes`` are already sorted by score descending.
    We partition by scene_id order (chronological proxy) and pick the
    highest-scoring candidate from each non-empty window.  Returns a list
    of scene dicts (may be shorter than n_windows if some windows are empty).
    """
    if not ranked_scenes:
        return []

    all_ids  = sorted(s["scene_id"] for s in ranked_scenes)
    id_to_scene = {s["scene_id"]: s for s in ranked_scenes}

    n        = len(all_ids)
    n_wins   = min(n_windows, n)
    size     = math.ceil(n / n_wins)

    chosen: List[Dict] = []
    chosen_ids: set    = set()

    for win in range(n_wins):
        lo = win * size
        hi = min(lo + size, n)
        window_ids = set(all_ids[lo:hi])

        # Best candidate in this window (ranked_scenes is score-sorted)
        best = next(
            (s for s in ranked_scenes if s["scene_id"] in window_ids),
            None,
        )
        if best and best["scene_id"] not in chosen_ids:
            chosen.append(best)
            chosen_ids.add(best["scene_id"])

    return chosen


# ── adaptive selection ────────────────────────────────────────────────────────

def adaptive_select_scenes(
    ranked_scenes: list,
    watched_duration_sec: Optional[float] = None,
    keep_fraction: float = 0.55,
    n_windows: int = 8,
) -> list:
    """Select scenes using timeline coverage + score quantile, bounded by
    count-aware min/max.

    Args:
        ranked_scenes:       Scenes sorted by final score descending.
        watched_duration_sec: Seconds of video watched (drives bounds).
        keep_fraction:       Fraction of scenes to retain via quantile.
        n_windows:           Number of chronological windows for coverage pass.

    Returns:
        Unordered (score-ranked) selection; caller sorts by timeline.
    """
    if not ranked_scenes:
        return []

    n = len(ranked_scenes)

    # ── count-aware bounds ─────────────────────────────────────────────────
    if watched_duration_sec and watched_duration_sec > 0:
        dur = watched_duration_sec
        min_keep = max(8, math.ceil(n * 0.10), math.ceil(dur / 240))
        max_keep = min(
            35,
            max(min_keep, math.ceil(n * 0.20), math.ceil(dur / 90), 12),
        )
    else:
        min_keep = max(4, math.ceil(n * 0.10))
        max_keep = min(35, max(min_keep, math.ceil(n * 0.20), 12))

    # Clamp so we never request more than we have
    min_keep = min(min_keep, n)
    max_keep = min(max_keep, n)

    # ── timeline coverage seed (one scene per window) ──────────────────────
    seed = _timeline_coverage(ranked_scenes, n_windows=n_windows)
    selected_ids: set = {s["scene_id"] for s in seed}
    selected: list    = list(seed)

    # ── quantile shortlist ─────────────────────────────────────────────────
    scores      = sorted(s["final"] for s in ranked_scenes)
    cutoff_idx  = max(0, int(n * (1 - keep_fraction)))
    threshold   = scores[cutoff_idx]

    for s in ranked_scenes:
        if len(selected) >= max_keep:
            break
        if s["scene_id"] not in selected_ids and s["final"] >= threshold:
            selected.append(s)
            selected_ids.add(s["scene_id"])

    # ── pad to min_keep from globally ranked list ──────────────────────────
    if len(selected) < min_keep:
        for s in ranked_scenes:
            if len(selected) >= min_keep:
                break
            if s["scene_id"] not in selected_ids:
                selected.append(s)
                selected_ids.add(s["scene_id"])

    # ── trim to max_keep (by score) ────────────────────────────────────────
    if len(selected) > max_keep:
        selected = sorted(selected, key=lambda x: x["final"], reverse=True)
        selected = selected[:max_keep]

    return selected


# ── public API ────────────────────────────────────────────────────────────────

def restore_timeline_order(selected_scenes: list) -> list:
    """Sort selected scenes by scene_id (chronological proxy)."""
    return sorted(selected_scenes, key=lambda x: x["scene_id"])


def get_ranked_scenes(
    scene_data: list,
    threshold: float = 0.5,
    watched_duration_sec: Optional[float] = None,
    adaptive: bool = True,
) -> list:
    """Rank scenes and select the most important subset.

    Args:
        scene_data:           List of scene dicts with a 'final' score field.
        threshold:            Fallback fixed threshold (adaptive=False only).
        watched_duration_sec: Seconds watched; drives adaptive min/max caps.
        adaptive:             Use adaptive selection (default True).

    Returns:
        List of scene dicts in chronological (scene_id) order.
    """
    ranked = rank_scenes(scene_data)
    if adaptive:
        selected = adaptive_select_scenes(
            ranked, watched_duration_sec=watched_duration_sec
        )
    else:
        selected = select_by_threshold(ranked, threshold)
    return restore_timeline_order(selected)


def save_selected_scenes(data, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def extract_scene_ids(selected_scenes: list) -> list:
    return [
        int(s["scene_id"]) if isinstance(s, dict) else int(s)
        for s in selected_scenes
    ]


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    scene_data = load_scene_scores("data/fused_scores.json")

    selected_scenes   = get_ranked_scenes(scene_data)
    selected_scene_ids = extract_scene_ids(selected_scenes)

    save_selected_scenes(selected_scenes,    "data/selected_scenes.json")
    save_selected_scenes(selected_scene_ids, "data/ranked_scene_ids.json")

    print("Selected Scenes:")
    for scene in selected_scenes:
        print(scene)

    print("Saved selected scenes to data/selected_scenes.json")
    print("Saved ranked scene IDs to data/ranked_scene_ids.json")


if __name__ == "__main__":
    main()