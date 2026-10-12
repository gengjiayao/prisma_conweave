#!/usr/bin/env python3
"""Independently match published weights, decisions and actual flow lifetimes.

Default main sample is fixed by protocol: first test seed and fit, every family,
both fabrics, learning enabled and disabled. --all-runs checks every online run.
"""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path


def audit(folder, delay):
    job = json.loads((folder / 'job.json').read_text())
    fcts = {}
    for line in (folder / 'fct_output_file.txt').read_text().splitlines():
        row = list(map(int, line.split()))
        fcts[tuple(row[:4])] = row
    by_node = {}
    during_flow = updates = 0
    first_update = None
    with (folder / 'transitions.csv.online-updates.csv').open() as handle:
        for row in csv.DictReader(handle):
            node, key = int(row['node']), int(row['key'])
            now, ack, version = (int(row[k]) for k in ['update_ns', 'ack_ns', 'updates'])
            times = by_node.setdefault(node, [])
            assert version == len(times) + 1
            assert now == ack + delay and (not times or now >= times[-1] + delay)
            assert float(row['delta_norm']) > 0
            weights = [float(v) for k, v in row.items() if k.startswith('w')]
            assert len(weights) == 36 and all(math.isfinite(v) and abs(v) <= 2 for v in weights)
            times.append(now)
            # Native ConWeave key encodes host IDs and transport ports.
            flow_key = ((key >> 32) & 65535, key & 65535,
                        (key >> 48) & 65535, (key >> 16) & 65535)
            flow = fcts[flow_key]
            during_flow += flow[5] < now < flow[5] + flow[6]
            updates += 1
            first_update = now if first_update is None else min(first_update, now)
    decisions = updated_decisions = redecisions = 0
    previous = {}
    with (folder / 'transitions.csv.online-decisions.csv').open() as handle:
        for row in csv.DictReader(handle):
            node, key, now, version = (int(row[k]) for k in ['node', 'key', 'ns', 'updates'])
            times = by_node.get(node, [])
            # Events at one timestamp have an order; either adjacent version is causal.
            assert bisect.bisect_left(times, now) <= version <= bisect.bisect_right(times, now), 'Future weights used'
            updated_decisions += version > 0
            pair = (node, key)
            redecisions += pair in previous and version > previous[pair]
            previous[pair] = version
            decisions += 1
    if job['arm'] == 'online':
        assert updates > 0 and updated_decisions > 0
    else:
        assert updates == 0 and updated_decisions == 0
    native = json.loads((folder / 'transitions.csv.online-audit.json').read_text())
    assert updates == sum(v['updates'] for v in native['leaves'])
    assert decisions == sum(v['decisions'] for v in native['leaves'])
    return dict(run=folder.name, updates=updates, decisions=decisions,
                decisions_after_update=updated_decisions,
                updates_while_same_flow_active=during_flow,
                flow_redecisions_with_new_weights=redecisions, first_update_ns=first_update)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=['smoke', 'main'], default='main')
    parser.add_argument('--all-runs', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.experiment_dir / 'protocol.json').read_text())
    paths = json.loads((args.experiment_dir / f'{args.phase}-jobs.json').read_text())
    selected = []
    for path in paths:
        folder = Path(path)
        job = json.loads((folder / 'job.json').read_text())
        if job['arm'] not in ['online', 'online_off']:
            continue
        if args.phase == 'main' and not args.all_runs and (job['seed'] != protocol['test_seeds'][0] or job['fit'] != 0):
            continue
        assert json.loads((folder / 'online-result.json').read_text())['all_complete']
        selected.append(folder)
    assert len(selected) == (8 if args.phase == 'smoke' else (432 if args.all_runs else 24))
    rows = [audit(folder, protocol['update_delay_ns']) for folder in selected]
    result = dict(phase=args.phase, checked_runs=len(rows), all_passed=True,
                  selection='All online runs' if args.all_runs else 'First test seed and fit, every family and topology',
                  runs=rows)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}))


if __name__ == '__main__':
    main()
