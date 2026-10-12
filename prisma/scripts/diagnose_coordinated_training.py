#!/usr/bin/env python3
"""Audit action sensitivity and training tradeoffs without changing trained models."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import coordinated_offline_evaluation as campaign


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, data):
    campaign.env.dump(Path(path), data)


def decisions(folder, cache):
    folder, cache = Path(folder), Path(cache)
    dest = cache / (folder.name + '.npz')
    if dest.exists():
        with np.load(dest) as f:
            return {k: f[k] for k in f.files}
    g = np.loadtxt(folder / 'transitions.csv.coordinated.csv', delimiter=',',
                   skiprows=1, usecols=(0, 2, 6, 8, 9, 10), dtype=np.int64).reshape(-1, 8, 6)
    assert np.all(g[:, :, 3].sum(axis=1) == 1)
    assert np.all(g[:, :, 1] == g[:, :1, 1])
    assert np.all((g[:, :, 2] >= 17) & (g[:, :, 2] <= 24))
    idx = np.arange(len(g))
    order = np.argsort(g[:, 0, 1])
    data = dict(flow=g[:, 0, 1], node=g[:, 0, 0],
                port=g[idx, np.argmax(g[:, :, 3], axis=1), 2],
                greedy=g[idx, g[:, 0, 4], 2],
                mask=np.sum(g[:, :, 5] * (1 << (g[:, :, 2] - 17)), axis=1),
                eligible=g[:, :, 5].sum(axis=1))
    data = {k: v[order] for k, v in data.items()}
    assert len(np.unique(data['flow'])) == len(g)
    np.savez_compressed(dest, **data)
    return data


def compare_actions(a, b):
    assert np.array_equal(a['flow'], b['flow'])
    assert np.array_equal(a['node'], b['node'])
    return dict(decisions=len(a['flow']),
                path_changed=int(np.sum(a['port'] != b['port'])),
                greedy_changed=int(np.sum(a['greedy'] != b['greedy'])),
                candidate_set_changed=int(np.sum(a['mask'] != b['mask'])),
                all_eligible_a=int(np.sum(a['eligible'] == 8)),
                all_eligible_b=int(np.sum(b['eligible'] == 8)))


def pair_worker(task):
    a, b, cache, meta = task
    x = compare_actions(decisions(a['folder'], cache), decisions(b['folder'], cache))
    x.update(meta)
    x.update(minus=a['name'], plus=b['name'],
             p95_log_ratio=float(np.log(b['p95_us'] / a['p95_us'])),
             p99_log_ratio=float(np.log(b['p99_us'] / a['p99_us'])))
    return x


def select(scores, fair, ports, gate, theta):
    idx = np.arange(len(scores))
    greedy = np.zeros(len(scores), dtype=int)
    for k in range(1, 8):
        greedy = np.where(scores[:, k] < scores[idx, greedy] - 1e-12, k, greedy)
    eligible = scores <= scores[idx, greedy, None] + theta[4] * gate[:, None] + 1e-12
    chosen = greedy.copy()
    first = np.ones(len(scores), dtype=bool)
    for k in range(8):
        better = ((fair[:, k] < fair[idx, chosen] - 1e-12) |
                  ((np.abs(fair[:, k] - fair[idx, chosen]) <= 1e-12) &
                   (ports[:, k] < ports[idx, chosen])))
        take = eligible[:, k] & (first | better)
        chosen = np.where(take, k, chosen)
        first &= ~eligible[:, k]
    return ports[idx, chosen], ports[idx, greedy], eligible


def score_delta(x, burst, theta):
    gate = burst / (burst + theta[5])
    trust = (1 - .75 * gate[:, None]) / (1 + x[:, :, 4] / theta[6])
    reserve = .25 + .75 * gate[:, None]
    score = (theta[0] * trust * x[:, :, 1] + theta[1] * reserve * x[:, :, 23] +
             theta[2] * reserve * x[:, :, 22] + theta[3] * x[:, :, 8])
    return score, gate


def variants(theta):
    for k in range(8):
        for sign in [-1, 1]:
            t = theta.copy()
            t[k] = np.clip(t[k] + sign * .1 * campaign.SCALE[k],
                           campaign.LOWER[k], campaign.UPPER[k])
            yield k, sign, t


def rescore_worker(row):
    folder = Path(row['folder'])
    audit = read(folder / 'transitions.csv.coordinated-audit.json')
    theta = np.array(audit['theta'])
    g = np.loadtxt(folder / 'transitions.csv.coordinated.csv', delimiter=',', skiprows=1,
                   usecols=(2, 6, 8, 9, 10, 11, 12, 16)).reshape(-1, 8, 8)
    native = np.loadtxt(folder / 'transitions.csv.candidates.csv', delimiter=',', skiprows=1).reshape(-1, 8, 29)
    assert np.array_equal(g[:, :, 0], native[:, :, 0])
    x, ports, burst = native[:, :, 5:], g[:, :, 1].astype(int), g[:, 0, 7]
    base_score, gate = score_delta(x, burst, theta)
    # Full-precision recorded scores anchor the replay. Candidate feature exports
    # have 12 significant digits; only the score difference uses those features.
    actual = ports[np.arange(len(g)), np.argmax(g[:, :, 2], axis=1)]
    reference, greedy, eligible = select(g[:, :, 5], g[:, :, 6], ports, gate, theta)
    assert np.array_equal(reference, actual), row['name']
    assert np.array_equal(eligible, g[:, :, 4].astype(bool))
    assert np.array_equal(greedy, ports[np.arange(len(g)), g[:, 0, 3].astype(int)])
    results = []
    for k, sign, t in variants(theta):
        score, new_gate = score_delta(x, burst, t)
        chosen, new_greedy, mask = select(g[:, :, 5] + score - base_score,
                                          g[:, :, 6], ports, new_gate, t)
        changed = new_greedy != greedy
        results.append(dict(parameter=k, sign=sign, before=float(theta[k]), after=float(t[k]),
                            decisions=len(g), path_changed=int(np.sum(chosen != reference)),
                            greedy_changed=int(np.sum(changed)),
                            greedy_changed_final_same=int(np.sum(changed & (chosen == reference))),
                            candidate_set_changed=int(np.sum(np.any(mask != eligible, axis=1)))))
    return dict(name=row['name'], family=row['family'], topology=row['topology_name'],
                checkpoint=int(row['name'].rsplit('_r', 1)[1]), decisions=len(g),
                all_eligible=int(np.sum(eligible.sum(axis=1) == 8)), variants=results)


def history(source, root, workers):
    cache = root / 'cache'
    cache.mkdir(exist_ok=True)
    tasks, tradeoffs, updates = [], [], []
    for rnd in range(8):
        rows = read(source / f'train-r{rnd}-results.json')
        record = read(source / f'train-round{rnd}.json')
        lookup = {r['name']: r for r in rows}
        for update in record['updates']:
            if update['arm'] != 'real':
                continue
            fit = update['fit']
            reward = np.array(update['raw_rewards']).reshape(4, 2)
            top = set(np.argsort(-reward.max(axis=1), kind='stable')[:2].tolist())
            updates.append(dict(round=rnd + 1, fit=fit,
                                scaled_update_norm=float(np.linalg.norm(update['normalized_delta'])),
                                before=update['before'], after=update['after']))
            for d in range(4):
                for a in rows:
                    if a['arm'] != 'train_real' or a['fit'] != fit or not a['name'].endswith(f'_d{d}_z-1'):
                        continue
                    b = lookup[a['name'][:-2] + '1']
                    meta = dict(round=rnd + 1, fit=fit, direction=d, family=a['family'],
                                topology=a['topology_name'], selected_direction=d in top)
                    tasks.append((a, b, str(cache), meta))
                    if a['family'] == 'alltoall':
                        better_plus = reward[d, 1] > reward[d, 0]
                        worse, better = (a, b) if better_plus else (b, a)
                        tradeoffs.append(dict(**meta, reward_gap=float(abs(reward[d, 1]-reward[d, 0])),
                                              p95_change_pct=100*(better['p95_us']/worse['p95_us']-1),
                                              p99_change_pct=100*(better['p99_us']/worse['p99_us']-1)))
    assert len(tasks) == 576 and len(tradeoffs) == 96
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pairs = []
        for item in pool.map(pair_worker, tasks):
            pairs.append(item)
            if len(pairs) % 48 == 0:
                print('PAIRS', len(pairs), flush=True)
    val = read(source / 'validation-results.json')
    selected = [r for r in val if r['arm'] == 'val_real' and r['fit'] == 0 and
                r['seed'] == 193501 and (r['name'].endswith('_r0') or r['name'].endswith('_r8'))]
    assert len(selected) == 48
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rescored = list(pool.map(rescore_worker, selected))
    refs = [r for r in val if r['arm'] in ['conweave', 'reserve_only']]
    curves = []
    for fit in range(3):
        for rnd in [0, 4, 8]:
            sub = [r for r in val if r['arm'] == 'val_real' and r['fit'] == fit and r['name'].endswith(f'_r{rnd}')]
            curves.append(dict(fit=fit, round=rnd, loss=campaign.objective(sub, refs)))
    family_changes = []
    for family in campaign.FAMILIES:
        for rnd in [4, 8]:
            sub = [r for r in val if r['arm'] == 'val_real' and r['family'] == family]
            initial = {(campaign.identity(r), r['fit']): r for r in sub if r['name'].endswith('_r0')}
            later = [r for r in sub if r['name'].endswith(f'_r{rnd}')]
            v = dict(family=family, round=rnd)
            for m in ['p95_us', 'p99_us']:
                v[m] = float(100*np.expm1(np.mean([np.log(r[m]/initial[campaign.identity(r), r['fit']][m]) for r in later])))
            family_changes.append(v)
    dump(root / 'history.json', dict(pairs=pairs, rescore=rescored, reward_tradeoffs=tradeoffs,
                                     updates=updates, validation_curves=curves, validation_family_changes=family_changes,
                                     source_selection=read(source/'selection.json')))
    print('HISTORY_COMPLETE', flush=True)


def native(source, root, workers):
    exp = root / 'native'
    p = campaign.init(exp)
    assert p['binary_sha256'] == read(source / 'protocol.json')['binary_sha256']
    assert p['library_sha256'] == read(source / 'protocol.json')['library_sha256']
    sel = read(source / 'selection.json')['models']['real'][0]
    theta = np.array(list(map(float, Path(sel['path']).read_text().split()[1:])))
    models = [(-1, 0, theta)] + list(variants(theta))
    jobs, plan = [], []
    for k, sign, t in models:
        model = campaign.export(t, exp / 'models' / f'probe_{k}_{sign}.coord')
        for family, seed, ranks, size in [('uniform',194001,0,0), ('alltoall',194002,48,131072)]:
            for topology in ['symmetric','asymmetric']:
                case = dict(family=family,ranks=ranks,size=size,load=.35)
                folder = campaign.prepare(exp,p,'diagnostic',case,seed,topology,'learned',0,model,
                                           suffix=f'_probe{k}_sign{sign}')
                jobs.append(folder)
                plan.append(dict(name=folder.name, parameter=k, sign=sign, theta=t.tolist(),
                                 family=family, topology=topology))
    assert len(jobs) == 68
    dump(root / 'native-plan.json', dict(purpose='Development-only one-parameter local sensitivity, not held-out generalization',
        reference_model=sel, step=(.1*campaign.SCALE).tolist(), jobs=plan, native_runs=68,
        parameters=p['parameters'], future_test_reuse_for_selection=False))
    campaign.execute(exp,jobs,workers,'diagnostic')
    print('NATIVE_COMPLETE', flush=True)


def analyze(source, root, workers):
    h = read(root / 'history.json')
    plan = read(root / 'native-plan.json')
    native_rows = read(root / 'native/diagnostic-results.json')
    assert len(native_rows) == 68
    meta = {x['name']: x for x in plan['jobs']}
    baseline = {(r['family'], r['topology_name']): r for r in native_rows
                if meta[r['name']]['parameter'] == -1}
    records, flows, max_error = [], 0, 0.
    cache = root / 'cache'
    for r in native_rows:
        folder = Path(r['folder'])
        job, status = read(folder/'job.json'), read(folder/'status.json')
        assert status['state'] == 'complete' and status['returncode'] == 0
        for field in ['model','coord_model','binary','config','traffic','topology','plan']:
            if job.get(field):
                assert campaign.env.sha(job[field]) == job[field+'_sha256']
        audit = read(folder/'transitions.csv.coordinated-audit.json')
        assert audit['parameter_updates'] == 0
        assert np.array_equal(audit['theta'], meta[r['name']]['theta'])
        values = np.loadtxt(folder/'fct_output_file.txt', dtype=np.int64)
        assert len(values) == r['offered'] == r['completed']
        flows += len(values)
        for metric, q in [('p95_us',95),('p99_us',99)]:
            error = abs(float(np.percentile(values[:,6]/1000,q))-r[metric])
            max_error = max(max_error,error)
            assert error < 1e-8
        m = meta[r['name']]
        if m['parameter'] == -1:
            continue
        b = baseline[r['family'],r['topology_name']]
        item = compare_actions(decisions(b['folder'],cache),decisions(r['folder'],cache))
        item.update(name=r['name'],family=r['family'],topology=r['topology_name'],
                    parameter=m['parameter'],sign=m['sign'],
                    p95_change_pct=100*(r['p95_us']/b['p95_us']-1),
                    p99_change_pct=100*(r['p99_us']/b['p99_us']-1))
        records.append(item)
    family = []
    for fam in campaign.FAMILIES:
        pairs = [v for v in h['pairs'] if v['family'] == fam]
        assert len(pairs) == 96
        path = [100*v['path_changed']/v['decisions'] for v in pairs]
        gap = [100*np.expm1(abs(v['p99_log_ratio'])) for v in pairs]
        family.append(dict(family=fam,pairs=len(pairs),paired_flow_decisions=sum(v['decisions'] for v in pairs),
                           mean_path_change_pct=float(np.mean(path)),
                           median_path_change_pct=float(np.median(path)),
                           zero_action_change_pairs=sum(v['path_changed']==0 for v in pairs),
                           p99_gap_quantiles_pct=np.percentile(gap,[0,25,50,75,100]).tolist()))
    frozen = []
    for rnd in [0,8]:
        for fam in campaign.FAMILIES:
            sub = [v for v in h['rescore'] if v['checkpoint']==rnd and v['family']==fam]
            for k in range(8):
                vs = [v for s in sub for v in s['variants'] if v['parameter']==k]
                total = sum(v['decisions'] for v in vs)
                changed = sum(v['greedy_changed'] for v in vs)
                frozen.append(dict(checkpoint=rnd,family=fam,parameter=k,decisions=total,
                    path_change_pct=100*sum(v['path_changed'] for v in vs)/total,
                    greedy_change_pct=100*changed/total,
                    changed_greedy_but_same_final_pct=(100*sum(v['greedy_changed_final_same'] for v in vs)/changed if changed else None),
                    all_paths_eligible_pct=100*sum(v['all_eligible'] for v in sub)/sum(v['decisions'] for v in sub)))
    chosen = [v for v in h['reward_tradeoffs'] if v['selected_direction']]
    conflicts = dict(selected_directions=len(chosen),
                     worse_alltoall_p99=sum(v['p99_change_pct']>1e-9 for v in chosen),
                     worse_alltoall_p99_over_half_pct=sum(v['p99_change_pct']>.5 for v in chosen),
                     worse_alltoall_p99_over_one_pct=sum(v['p99_change_pct']>1 for v in chosen),
                     worse_alltoall_p99_over_five_pct=sum(v['p99_change_pct']>5 for v in chosen))
    verification = dict(native_runs=len(native_rows),flow_instances=flows,max_quantile_error_us=max_error,
                        native_inputs_hash_verified=True,all_parameter_updates_zero=True,
                        rescore_reference_decisions=sum(v['decisions'] for v in h['rescore']),
                        rescore_reference_actions_exact=True)
    for filename, expected in read(source.parent/'immutable-inputs.json').items():
        assert campaign.env.sha(filename)==expected
    verification['original_immutable_inputs_unchanged']=True
    out = dict(family_pairs=family,frozen_state=frozen,native_probes=records,reward_conflicts=conflicts,
               validation_curves=h['validation_curves'],validation_family_changes=h['validation_family_changes'],
               validation_selection=h['source_selection']['scores'],verification=verification,
               limitations=['Trajectory action differences include feedback amplification, not just direct action sensitivity.',
                 'Frozen-state replay holds history, reservations, and predictions fixed; parameter7 only affects later states.',
                 'Native probes use one selected model, two development traffic seeds, two topologies, and local perturbations.',
                 'No extended training, convergence proof, global optimum, or new held-out generalization claim.'])
    dump(root/'diagnosis.json',out)
    plot_diagnosis(root,out)
    print('DIAGNOSIS_COMPLETE',verification,conflicts,flush=True)


def plot_diagnosis(root,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'savefig.dpi':170,
                         'axes.spines.top':False,'axes.spines.right':False})
    figures = root/'figures'
    figures.mkdir(exist_ok=True)
    def save(fig,name):
        for ext in ['png','pdf']:
            fig.savefig(figures/(name+'.'+ext),bbox_inches='tight')
        plt.close(fig)
    labels=['Uniform','Storage','Fast shift','Bursty','All-to-All','All-Reduce']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,values,title,ylabel,color in [
        (axes[0],[v['mean_path_change_pct'] for v in out['family_pairs']],
         'Different initial paths, same offered traffic','Mean changed flows (%)','#167f85'),
        (axes[1],[v['p99_gap_quantiles_pct'][2] for v in out['family_pairs']],
         'How much did completion time differ?','Median P99 gap (%)','#b9643e')]:
        bars=ax.bar(labels,values,color=color,width=.65)
        ax.bar_label(bars,fmt='%.2f',padding=4)
        ax.set_ylim(0,max(values)*1.18)
        ax.set_ylabel(ylabel);ax.set_title(title);ax.tick_params(axis='x',rotation=30)
        ax.grid(axis='y',alpha=.15)
    fig.suptitle('96 training perturbation pairs per family; P99 gap = larger / smaller − 1',fontweight='bold')
    save(fig,'action_vs_performance')
    contexts=[(f,t,z) for f in ['uniform','alltoall'] for t in ['symmetric','asymmetric'] for z in [-1,1]]
    pnames=['Remote weight','Destination debt','Port debt','Port history','Candidate window',
            'Burst threshold','Feedback age','Prediction scale']
    data=[]
    for metric in ['path','p99']:
        a=np.zeros((8,8))
        for v in out['native_probes']:
            j=contexts.index((v['family'],v['topology'],v['sign']))
            a[v['parameter'],j]=(100*v['path_changed']/v['decisions'] if metric=='path' else v['p99_change_pct'])
        data.append(a)
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    for i,(ax,a) in enumerate(zip(axes,data)):
        im=ax.imshow(a,aspect='auto',cmap='Blues' if i==0 else 'RdBu_r',
                     norm=None if i==0 else TwoSlopeNorm(vmin=-max(abs(a.min()),abs(a.max())),vcenter=0,vmax=max(abs(a.min()),abs(a.max()))),
                     **({'vmin':0,'vmax':100} if i==0 else {}))
        for row in range(8):
            for col in range(8):
                value = a[row,col]
                label = (f'{value:.1f}' if i == 0 or abs(value) >= .1 else
                         (f'{value:+.2f}' if abs(value) >= .01 else '<0.01'))
                ax.text(col,row,label,ha='center',va='center',fontsize=8,
                        color='white' if (a[row,col]>60 if i==0 else abs(a[row,col])>.7*np.max(np.abs(a))) else '#1c2733')
        ax.set_yticks(range(8),pnames)
        ax.set_xticks(range(8),[('Uniform' if f=='uniform' else 'A2A')+' / '+('sym' if t=='symmetric' else 'asym')+' / '+str(z) for f,t,z in contexts],rotation=45,ha='right')
        ax.set_title('Changed initial paths (%)' if i==0 else 'P99 change vs unchanged model (%)')
        fig.colorbar(im,ax=ax,shrink=.75)
    fig.suptitle('68 native runs: one parameter at a time (−1 / +1 = direction; <0.01 = magnitude)',fontweight='bold')
    save(fig,'native_parameter_sensitivity')
    fig,axes=plt.subplots(1,3,figsize=(17,4.5),layout='constrained')
    for fit,color in enumerate(['#157f85','#ac663d','#79689b']):
        v=[q for q in out['validation_curves'] if q['fit']==fit]
        axes[0].plot([q['round'] for q in v],[q['loss'] for q in v],marker='o',color=color,label=f'Model {fit+1}')
    group=[q for q in out['validation_selection'] if q['arm']=='real']
    axes[0].plot([q['round'] for q in group],[q['loss'] for q in group],ls='--',color='#293b46',label='Joint selection objective')
    axes[0].set_xticks([0,4,8]);axes[0].set_xlabel('Offline update rounds');axes[0].set_ylabel('Validation objective (lower is better)');axes[0].legend(fontsize=8);axes[0].grid(alpha=.15)
    for ax,metric in zip(axes[1:],['p95_us','p99_us']):
        for rnd,offset,color in [(4,-.18,'#9fb8be'),(8,.18,'#157f85')]:
            v=[next(q[metric] for q in out['validation_family_changes'] if q['family']==f and q['round']==rnd) for f in campaign.FAMILIES]
            ax.bar(np.arange(6)+offset,v,width=.34,color=color,label=f'Round {rnd}')
        ax.axhline(0,color='#66727b',lw=.8);ax.set_xticks(range(6),labels,rotation=30,ha='right');ax.set_ylabel(metric[:3].upper()+' change vs initialization (%)');ax.legend();ax.grid(axis='y',alpha=.15)
    fig.suptitle('Validation evidence: objective improves, All-to-All does not',fontweight='bold')
    save(fig,'training_dynamics')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=['history','native','analyze'],required=True)
    parser.add_argument('--workers',type=int,default=8)
    args = parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    globals()[args.stage](args.source,args.root,args.workers)


if __name__ == '__main__':
    main()
