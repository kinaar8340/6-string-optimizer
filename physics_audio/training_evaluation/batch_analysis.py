"""
Batch note clustering and key detection from real-audio fitting results.
"""

from __future__ import annotations

import json
import math
import numpy as np
from pathlib import Path
from typing import Optional

# Krumhansl-Kessler major/minor key profiles (pitch class 0=C)
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
KEYS_MAJOR = [f"{n} major" for n in NOTE_NAMES]
KEYS_MINOR = [f"{n} minor" for n in NOTE_NAMES]


def hz_to_midi(hz: float) -> float:
    if hz <= 0:
        return 0.0
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def midi_to_note(midi: float) -> str:
    n = int(round(midi)) % 12
    octave = int(round(midi)) // 12 - 1
    return f"{NOTE_NAMES[n]}{octave}"


def f0_to_pitch_class(hz: float) -> int:
    return int(round(hz_to_midi(hz))) % 12


def detect_key_from_f0_list(f0_hz_list: list[float]) -> dict:
    """
    Estimate musical key from a list of fundamental frequencies using
    Krumhansl-Schmuckler correlation with major/minor profiles.
    """
    if not f0_hz_list:
        return {'key': 'unknown', 'confidence': 0.0, 'chroma': [0.0] * 12}

    chroma = np.zeros(12, dtype=np.float64)
    for hz in f0_hz_list:
        if hz > 20:
            pc = f0_to_pitch_class(hz)
            chroma[pc] += 1.0

    if chroma.sum() < 1e-8:
        return {'key': 'unknown', 'confidence': 0.0, 'chroma': chroma.tolist()}

    chroma = chroma / chroma.sum()
    best_key, best_corr, best_mode = 'C major', -2.0, 'major'

    for shift in range(12):
        rolled = np.roll(chroma, -shift)
        corr_maj = float(np.corrcoef(rolled, _MAJOR_PROFILE)[0, 1])
        corr_min = float(np.corrcoef(rolled, _MINOR_PROFILE)[0, 1])
        if corr_maj > best_corr:
            best_corr, best_key, best_mode = corr_maj, KEYS_MAJOR[shift], 'major'
        if corr_min > best_corr:
            best_corr, best_key, best_mode = corr_min, KEYS_MINOR[shift], 'minor'

    return {
        'key': best_key,
        'mode': best_mode,
        'confidence': best_corr,
        'chroma': chroma.tolist(),
        'n_notes': len(f0_hz_list),
    }


def cluster_notes(
    summaries: list[dict],
    n_clusters: Optional[int] = None,
) -> dict:
    """
    Cluster batch results by (f0, B, mean damping, recon_mse).
    Uses sklearn KMeans when available; falls back to pitch-class bins.
    """
    if not summaries:
        return {'clusters': [], 'labels': [], 'method': 'none'}

    features = []
    for s in summaries:
        damps = s.get('learned_damps') or []
        mean_damp = float(np.mean(damps)) if damps else 0.0
        features.append([
            s.get('f0_est_hz', 0.0) / 500.0,
            s.get('b_est', 0.0) * 1000.0,
            mean_damp,
            s.get('recon_mse', 0.0),
        ])
    X = np.array(features, dtype=np.float64)

    try:
        from sklearn.cluster import KMeans
        k = n_clusters or max(2, min(5, len(summaries)))
        k = min(k, len(summaries))
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        method = 'kmeans'
    except ImportError:
        labels = np.array([f0_to_pitch_class(s.get('f0_est_hz', 0)) for s in summaries])
        k = len(set(labels))
        method = 'pitch_class'

    clusters = []
    for cid in sorted(set(labels)):
        members = [summaries[i] for i, lab in enumerate(labels) if lab == cid]
        f0s = [m['f0_est_hz'] for m in members]
        clusters.append({
            'cluster_id': int(cid),
            'count': len(members),
            'names': [m['name'] for m in members],
            'f0_mean_hz': float(np.mean(f0s)),
            'f0_median_hz': float(np.median(f0s)),
            'note_median': midi_to_note(hz_to_midi(float(np.median(f0s)))),
            'members': members,
        })

    return {
        'method': method,
        'n_clusters': len(clusters),
        'labels': labels.tolist(),
        'clusters': clusters,
    }


def analyze_batch_results(
    results: list[dict],
    output_dir: Path,
    n_clusters: Optional[int] = None,
) -> dict:
    """Run clustering + key detection; write reports to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    f0_list = [r['f0_est_hz'] for r in results if r.get('f0_est_hz', 0) > 20]
    key_info = detect_key_from_f0_list(f0_list)
    cluster_info = cluster_notes(results, n_clusters=n_clusters)

    per_note_keys = []
    for r in results:
        f0 = r.get('f0_est_hz', 0)
        per_note_keys.append({
            'name': r['name'],
            'f0_hz': f0,
            'note': midi_to_note(hz_to_midi(f0)) if f0 > 20 else 'unknown',
            'pitch_class': NOTE_NAMES[f0_to_pitch_class(f0)] if f0 > 20 else '?',
        })

    report = {
        'key_detection': key_info,
        'clustering': {
            'method': cluster_info['method'],
            'n_clusters': cluster_info['n_clusters'],
            'clusters': [
                {k: v for k, v in c.items() if k != 'members'}
                for c in cluster_info['clusters']
            ],
        },
        'per_note': per_note_keys,
    }

    report_path = output_dir / 'batch_analysis.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n--- Batch analysis ---")
    print(f"  Detected key: {key_info['key']} (confidence={key_info['confidence']:.3f})")
    print(f"  Clusters ({cluster_info['method']}): {cluster_info['n_clusters']}")
    for c in cluster_info['clusters']:
        print(f"    [{c['cluster_id']}] {c['count']} notes | f0≈{c['f0_median_hz']:.1f}Hz ({c['note_median']}) | {c['names']}")
    print(f"  Report: {report_path}")

    return report