import csv
from pathlib import Path
import numpy as np

root = Path('.')
coarse_path = root / 'out_quench_1000_K_coarse' / 'quench_log.dat'
fine_path = root / 'out_quench_1000_K_fine' / 'quench_log.dat'
out_root = root / 'validation' / 'quench_protocol_review_q1000'
out_root.mkdir(parents=True, exist_ok=True)

OBS = [
    'bulk','elastic','anchoring','total','radiality','avg_S','max_S',
    'defect_density_per_plaquette','xi_grad_proxy'
]

def load(path):
    data = np.genfromtxt(path, delimiter=',', names=True, dtype=float, encoding='utf-8')
    if getattr(data, 'size', 0) == 0:
        raise RuntimeError(f'empty log: {path}')
    return np.array(data, ndmin=1)

def series(data, *names):
    fields = set(data.dtype.names or ())
    for name in names:
        if name in fields:
            return np.atleast_1d(data[name]).astype(float)
    raise KeyError(names)

def crossing_time(time_s, temperature_K, tc_K):
    for idx in range(time_s.size):
        if np.isfinite(temperature_K[idx]) and abs(temperature_K[idx] - tc_K) <= 1e-12:
            return float(time_s[idx])
    for idx in range(1, time_s.size):
        T0 = temperature_K[idx - 1]
        T1 = temperature_K[idx]
        if not (np.isfinite(T0) and np.isfinite(T1)):
            continue
        if (T0 - tc_K) * (T1 - tc_K) < 0.0:
            t0 = time_s[idx - 1]
            t1 = time_s[idx]
            return float(t0 + (tc_K - T0) * (t1 - t0) / (T1 - T0))
    raise RuntimeError(f'No crossing for Tc={tc_K}')

def interpolate_at(time_s, values, target):
    if target < time_s[0] - 1e-18 or target > time_s[-1] + 1e-18:
        return float('nan')
    idx = int(np.searchsorted(time_s, target, side='left'))
    if idx < time_s.size and abs(time_s[idx] - target) <= 1e-18:
        return float(values[idx])
    if idx <= 0:
        return float(values[0])
    if idx >= time_s.size:
        return float(values[-1])
    t0 = time_s[idx - 1]
    t1 = time_s[idx]
    v0 = values[idx - 1]
    v1 = values[idx]
    if not (np.isfinite(v0) and np.isfinite(v1) and np.isfinite(t0) and np.isfinite(t1)):
        return float('nan')
    if abs(t1 - t0) <= 1e-30:
        return float(v1)
    w = (target - t0) / (t1 - t0)
    return float(v0 + w * (v1 - v0))

def rel_diff(a, b, floor=1e-18):
    if not (np.isfinite(a) and np.isfinite(b)):
        return float('nan')
    return abs(a - b) / max(abs(a), abs(b), floor)

coarse = load(coarse_path)
fine = load(fine_path)
ct = series(coarse, 'time_s', 'time')
ft = series(fine, 'time_s', 'time')
cT = series(coarse, 'T_K')
fT = series(fine, 'T_K')
cdt = series(coarse, 'dt_s', 'dt')
fdt = series(fine, 'dt_s', 'dt')

tc = 310.2
coarse_tc = crossing_time(ct, cT, tc)
fine_tc = crossing_time(ft, fT, tc)
horizon = min(float(ct[-1] - coarse_tc), float(ft[-1] - fine_tc))
offsets = [0.0]
for value in ct:
    off = float(value - coarse_tc)
    if off <= 1e-18 or off > horizon + 1e-18:
        continue
    if abs(off - offsets[-1]) > 1e-18:
        offsets.append(off)

summary = {
    'coarse_samples': int(ct.size),
    'fine_samples': int(ft.size),
    'coarse_final_time_s': float(ct[-1]),
    'fine_final_time_s': float(ft[-1]),
    'coarse_tc_crossing_s': coarse_tc,
    'fine_tc_crossing_s': fine_tc,
    'tc_crossing_abs_diff_s': abs(coarse_tc - fine_tc),
    'coarse_dt_min_s': float(np.nanmin(cdt)),
    'coarse_dt_max_s': float(np.nanmax(cdt)),
    'fine_dt_min_s': float(np.nanmin(fdt)),
    'fine_dt_max_s': float(np.nanmax(fdt)),
    'coarse_fixed_dt': bool(float(np.nanmax(cdt) - np.nanmin(cdt)) <= 1e-18),
    'fine_fixed_dt': bool(float(np.nanmax(fdt) - np.nanmin(fdt)) <= 1e-18),
    'shared_offset_count': len(offsets),
    'horizon_s': horizon,
}

comparison_rows = []
max_abs = {name: -1.0 for name in OBS}
max_rel = {name: -1.0 for name in OBS}
max_offset = {name: float('nan') for name in OBS}
for off in offsets:
    row = {'offset_s': off}
    for name in OBS:
        cval = interpolate_at(ct, series(coarse, name), coarse_tc + off)
        fval = interpolate_at(ft, series(fine, name), fine_tc + off)
        ad = abs(cval - fval) if np.isfinite(cval) and np.isfinite(fval) else float('nan')
        rd = rel_diff(cval, fval)
        row[f'coarse_{name}'] = cval
        row[f'fine_{name}'] = fval
        row[f'abs_diff_{name}'] = ad
        row[f'rel_diff_{name}'] = rd
        if np.isfinite(ad) and ad >= max_abs[name]:
            max_abs[name] = ad
            max_rel[name] = rd if np.isfinite(rd) else float('nan')
            max_offset[name] = off
    comparison_rows.append(row)

final_rows = []
for name in OBS:
    cval = float(series(coarse, name)[-1])
    fval = float(series(fine, name)[-1])
    final_rows.append({
        'observable': name,
        'coarse': cval,
        'fine': fval,
        'abs_diff': abs(cval - fval) if np.isfinite(cval) and np.isfinite(fval) else float('nan'),
        'rel_diff': rel_diff(cval, fval),
    })

with (out_root / 'summary.txt').open('w', encoding='utf-8') as fh:
    fh.write('Q1000 protocol review\n')
    fh.write('[summary]\n')
    for k, v in summary.items():
        fh.write(f'{k}={v}\n')
    fh.write('[max_abs_diffs]\n')
    for name in OBS:
        fh.write(f'{name}: abs={max_abs[name]} rel={max_rel[name]} offset_s={max_offset[name]}\n')
    fh.write('[final_state]\n')
    for row in final_rows:
        fh.write(f"{row['observable']}: coarse={row['coarse']} fine={row['fine']} abs={row['abs_diff']} rel={row['rel_diff']}\n")

if comparison_rows:
    with (out_root / 'offset_comparison.csv').open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

with (out_root / 'final_state_comparison.csv').open('w', encoding='utf-8', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=['observable', 'coarse', 'fine', 'abs_diff', 'rel_diff'])
    writer.writeheader()
    writer.writerows(final_rows)

print('[summary]')
for k, v in summary.items():
    print(f'{k}={v}')
print('[max_abs_diffs]')
for name in OBS:
    print(f'{name}: abs={max_abs[name]:.6g}, rel={max_rel[name]:.6g}, offset_s={max_offset[name]:.6g}')
print('[final_state]')
for row in final_rows:
    print(f"{row['observable']}: coarse={row['coarse']:.6g}, fine={row['fine']:.6g}, abs={row['abs_diff']:.6g}, rel={row['rel_diff']:.6g}")
