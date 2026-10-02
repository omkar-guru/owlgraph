"""Render actual tracker output on a deterministic synthetic same-class swap.

No detector, trained checkpoint, dataset, or GPU is used. This illustrates the
association interface; it is not evidence of real-video tracking accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sggpipeline.tracking.tracker import AssociationTracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/demo/tracking.svg'))
    args = parser.parse_args()
    tracker = AssociationTracker()
    boxes = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=np.float32)
    detections = SimpleNamespace(boxes=boxes, labels=np.zeros(2, dtype=np.int64))
    appearances = [np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32)[::-1]]
    records = []
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="860" height="330" viewBox="0 0 860 330">',
           '<rect width="860" height="330" fill="#0f172a"/>',
           '<g font-family="sans-serif" fill="#e2e8f0">',
           '<text x="30" y="38" font-size="23">Same category, persistent identities</text>',
           '<text x="30" y="66" font-size="14">Synthetic inputs • actual AssociationTracker output • no detector inference</text>']
    colors = ['#38bdf8', '#fbbf24']
    for frame, features in enumerate(appearances):
        result = tracker.update(detections, features, float(frame))
        records.append({'timestamp': float(frame), 'track_ids': result.track_ids.tolist(),
                        'is_new': result.is_new.tolist()})
        origin = 30 + frame * 425
        svg.append(f'<text x="{origin}" y="110" font-size="18">Frame {frame + 1}</text>')
        for row, track_id in enumerate(result.track_ids):
            x = origin + row * 190
            color = colors[int(track_id) % len(colors)]
            svg.extend([f'<rect x="{x}" y="132" width="155" height="100" rx="10" fill="{color}" fill-opacity="0.12" stroke="{color}" stroke-width="3"/>',
                        f'<text x="{x + 14}" y="171" font-size="17">object · ID {track_id}</text>',
                        f'<text x="{x + 14}" y="203" font-size="14">feature {features[row].astype(int).tolist()}</text>'])
    svg.extend(['<text x="30" y="272" font-size="15">Appearance descriptors swap positions; IDs follow the supplied descriptors.</text>',
                '<text x="30" y="299" font-size="13">Illustrates association behavior only. Real-video identity validation remains pending.</text>',
                '</g></svg>'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(svg) + '\n')
    report = args.output.with_suffix('.json')
    report.write_text(json.dumps({'input': 'synthetic', 'frames': records}, indent=2) + '\n')
    print(json.dumps(records, indent=2))
    print(f'Wrote {args.output} and {report}')


if __name__ == '__main__':
    main()
