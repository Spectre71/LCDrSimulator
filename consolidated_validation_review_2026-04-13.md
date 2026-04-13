# Consolidated Validation Review And Research Conclusions - 2026-04-13

## Purpose

This is the single maintained consolidated validation review for the current repo state.

It replaces the temporary validation-folder duplicate and keeps the benchmark ladder, confined interpretation, production figure set, and paper-facing conclusion in one root-level document that can be carried directly into article drafting.

## Source Trail Used For This Consolidation

The narrative below is based on the dated notes already present in `validation/` together with the matching entries in `Changelog.md`, especially:

- `validation/sanity3_rate_sweep_merged_analysis/findings_2026-04-10.md`
- `validation/sanity3_rate_sweep_weak_anchor_screen/findings_2026-04-10.md`
- `validation/sanity3_rate_sweep_weak_anchor_screen/qsrvis_audit_2026-04-11.md`
- `validation/sanity3_rate_sweep_weak_anchor_W3em6_screen/findings_2026-04-11.md`
- `validation/kzm_prooving_ground_2026-04-11.md`
- `validation/kzm_bulk_ldg_2026-04-11.md`
- `validation/point1_point2_event_based_and_biaxial_probe_2026-04-11.md`
- `validation/point2_tc_and_droplet_normalized_reanalysis_2026-04-11.md`
- `validation/point3_confined_final_state_bridge_2026-04-11.md`
- `validation/point4_no_early_stop_finalhold_2026-04-11.md`
- `validation/point5_option2_transient_log_followup_2026-04-11.md`
- `validation/point6_window60_qtensor_followup_2026-04-11.md`
- `validation/point7_window60_3d_transient_followup_2026-04-11.md`
- `validation/point8_midplane_slab_followup_2026-04-11.md`
- `validation/point9_z_profile_followup_2026-04-11.md`
- `validation/point10_shell_depth_followup_2026-04-11.md`
- `validation/point11_shell_band_decomposition_2026-04-11.md`
- `validation/point12_focus_summary_and_window_audit_2026-04-11.md`
- `validation/point13_neighbor_window_stability_2026-04-11.md`
- `validation/point14_focus_exponent_2026-04-12.md`
- `validation/point15_dense_rate_ladder_refinement_2026-04-12.md`
- `validation/point16_anchor_strength_fixed_band_2026-04-12.md`
- `validation/point17_size200_matched_rate_shell_depth_2026-04-12.md`
- `validation/point18_size200_four_rate_ladder_2026-04-13.md`
- `Changelog.md`

## Executive Summary

The repo now supports three distinct statements, and they should not be mixed together.

1. The codebase can recover Kibble-Zurek-like scaling when the physics is appropriate.
   - The periodic `3D XY` proving-ground branch gives the first clean benchmark result.
   - The periodic bulk `LdG` branch then shows that a `Q`-tensor nematic model without confinement can also produce a clean, refinement-stable scaling readout.

2. The confined droplet is not just the bulk problem with finite resolution.
   - In the confined branch, whole-volume and late-final-state observables do not reproduce the clean bulk-like signal.
   - The dominant confined response is boundary-conditioned and geometry-conditioned.

3. The confined system still contains a measurable Kibble-Zurek-like signal, but not where the original whole-volume search looked for it.
   - The surviving signal is not a robust whole-droplet `3D` core observable.
   - It is a localized late-transient shell-subsurface signal.
   - The active dense confined baseline is the `100^3`, `W = 3e-6`, late-window annulus `[2,10)` with pooled exponent about `0.311`.

So the present story is regime separation rather than solver failure:

- clean benchmark KZ in unconfined systems,
- clean bridge behavior in periodic bulk nematic `LdG`,
- suppressed or redistributed global KZ behavior in confined droplets,
- and a localized shell-conditioned confined KZ signal that only becomes visible once the observable is redesigned around the real droplet geometry.

## Production Figure Ladder

### 1. Periodic XY Proving Ground

![Periodic XY benchmark](pics/production/01_periodic_xy_benchmark.png)

Caption. The final-state vortex-line density in the first non-smoke periodic `64^3` `XY` sweep scales with fitted slope `m = -0.529`, already close to the textbook `3D XY` Model-A expectation `m = -0.573`. The fixed `+0.4` after-`T_c` control is nearly flat, so the benchmark signal currently lives in the later topological readout rather than in an arbitrary early snapshot.

### 2. Periodic Bulk LdG Bridge

![Periodic bulk LdG benchmark](pics/production/02_periodic_bulk_ldg_benchmark.png)

Caption. The first seven-rate periodic bulk `LdG` scan already gives a clean final-state defect-line slope `m = -0.515` with strong monotonicity and log-log correlation. Fixed absolute-time windows remain too transient, while matched-order windows are smoother but weaker, so the final-state line-defect readout stays the current bridge observable between the periodic `XY` benchmark and the confined droplet branch.

![Periodic bulk LdG protocol convergence](pics/production/03_periodic_bulk_ldg_protocol_convergence.png)

Caption. The matched fixed-`dt` refinement checks preserve the same physical quench schedule and show that the bulk branch is numerically controlled. The standard pair keeps the final-state relative mismatches near `6.9e-5` in `avg_S`, `1.20e-3` in `xi_grad_proxy`, and about `7.4e-3` in the defect observables. The slower `ramp600` pair is even tighter at the final state.

### 3. Confined Global Controls And Localization

![Confined global transient controls](pics/production/04_confined_global_transient_control.png)

Caption. The whole-volume confined `3D` proxies do not survive as a robust late bridge signal. Both the core-density and line-density slopes collapse toward zero by about `42-44 ns` after `T_c`, which is why the confined continuation cannot be framed as a simple bulk-style final-state observable problem.

![Confined shell localization](pics/production/05_confined_shell_localization.png)

Caption. The localization analysis shows why the active confined readout is a fixed subsurface band rather than a whole-droplet metric. The narrow `[4,6)` and `[2,6)` bands are steeper but live on much smaller support. The broader `[2,10)` band keeps a still-positive mean exponent with much larger and more stable support, so it is the current production readout. The deep bulk region `[10,inf)` stays weak.

![Confined fixed-band exponent](pics/production/06_confined_fixed_band_exponent.png)

Caption. Once the confined readout is fixed to `[2,10)`, the dense five-window branch gives the current best exponent estimate: `alpha = 0.3108` for defect density, with normalized `95%` fit confidence width about `0.0389` and neighboring-window half-range about `0.0389`. The paired defect-share fit is weaker but still positive, reinforcing that the confined signal is real but boundary-conditioned.

![Anchoring-strength comparison](pics/production/07_confined_anchor_strength_comparison.png)

Caption. The fixed-band point-2 comparison confirms that the confined exponent is non-monotonic in anchoring strength. On the common `[2,10)` readout, the late-window exponent is about `0.096` at `W = 1e-6`, `0.311` at `W = 3e-6`, and `0.273` at `W = 1e-5`. The active dense `W = 3e-6` branch remains the strongest current confined baseline rather than an arbitrary middle choice.

### 4. Size-200 Controls

![Matched-rate size-200 shift](pics/production/08_confined_size200_matched_rate_shift.png)

Caption. The first matched-rate `200^3` control does not reinforce the same shell-adjacent annulus that carried the dense `100^3` confined bridge. Total late defect density rises modestly, but the `[2,10)` density, support fraction, and defect-share fraction all decrease relative to `100^3`. At the same time, the defect-weighted mean shell depth sits well deeper than the support-weighted mean, so the immediate size effect is bulkward redistribution rather than simple strengthening of the same subsurface signal.

![Sparse size-200 ladder](pics/production/09_confined_size200_sparse_ladder.png)

Caption. The first pruned `200^3` four-rate ladder still keeps the strongest rate-sensitive band shell-adjacent rather than bulk-dominated. The preferred sparse-ladder annulus remains `[2,6)`, but its support fraction shrinks from about `0.220` at sparse `100^3` to about `0.119` at sparse `200^3`. The fixed `[2,10)` exponent at `200^3` rises back near `0.509`, but the bulk exponent remains small at about `0.073`, so this is not evidence of bulk recovery. It is still a shell-conditioned signal, and the apparent return to `~0.51` remains entangled with the same sparse-ladder bias that point 15 already exposed at `100^3`.

## Consolidated Narrative

### 1. Why The Benchmark Branches Were Necessary

The confined `QSR` droplet solver contains too many ingredients at once for a clean first benchmark: first-order isotropic-nematic `LdG` dynamics, finite geometry, anchoring, shell ordering, and geometry-dependent post-processing.

That is why the repo first split the problem into cleaner branches instead of forcing the droplet solver to serve as the benchmark.

The periodic `3D XY` branch answers the narrow question of whether the infrastructure can recover a textbook Kibble-Zurek power law when the symmetry, transition, and topology observable are the right ones. It does.

The periodic bulk `LdG` branch then bridges back to the active `Q`-tensor setting without reintroducing confinement. It shows that even within a nematic `Q`-tensor model the repo can recover a clean, refinement-stable scaling story once boundaries and droplet geometry are removed.

Those two branches are why the confined result should now be interpreted as a real geometry-and-boundary problem rather than as generic solver failure.

### 2. What The Early Confined Search Actually Found

The original strong-confinement `sanity3` transient bridge search did not produce the canonical sign expected from a bulk-like picture. Slower quenches transiently retained more `3D` structure in the early post-`T_c` window, and the effect decayed later.

Weak anchoring at `W = 1e-5` opened a short canonical-sign window in permissive whole-volume metrics, but that result was not robust. The sign reversed quickly, the skeleton metric was even less stable, and the later audit showed strong morphology dependence.

The weaker `W = 3e-6` screen clarified the situation further: permissive whole-volume low-`S` metrics could look positive, but conservative interior-core definitions collapsed. That separated “a positive proxy can be manufactured” from “a morphology-robust confined scaling law exists.”

### 3. What Timing And Normalization Clarified

The repo also resolved two interpretation issues that would otherwise contaminate any paper conclusion.

First, `Tc_KZ` in the current workflow is not a computed freeze-out time `t_hat`. It is a transition-temperature marker used for snapshot placement and for aligning measurements relative to the logged `T_c` crossing extracted from `quench_log.dat`.

Second, droplet-normalized observables removed an important false positive. Some earlier whole-volume weak-anchor signals looked stronger in slower runs largely because the ordered droplet itself was still developing. Once the metrics were normalized by the actual ordered droplet volume, that apparent positive whole-volume core signal disappeared.

### 4. Why The Whole-Volume And Final-State Confined Observables Were Rejected

Several tempting confined observables were explicitly tested and rejected.

- Event-based timing alone did not rescue the early weak-anchor core result.
- Strict biaxial-core proxies could produce positive windows under permissive morphology handling, but those windows collapsed once the interior region was defined more physically by shell-excluded distance.
- The periodic bulk final-state bridge did not survive in confinement. Even after removing the early-stop loophole, the whole-volume droplet-normalized final `3D` metrics stayed flat or zero.
- In the cleaner late `50-60 ns` and then `45-65 ns` windows, the `2D` logged signal remained modestly positive, but the tested global `3D` low-`S` and strict-biaxial observables were already flat or zero.

That is the key negative result: the confined signal survives, but not as a global `3D` defect observable of the type originally targeted.

### 5. How The Correct Confined Observable Was Found

The later confined work succeeded because it stopped asking the whole droplet to behave like the periodic bulk branch and instead localized the observable where the confined signal actually lives.

The equatorial slab and full axial-profile analyses showed that the surviving late signal was not just one special slice, but they also showed that axial position was not the real discriminator. The strongest individual slices were shell-adjacent.

The shell-depth analysis then measured the signal directly in inward distance from the interface. That established two things at once:

- the strongest slopes are not in the outermost skin but a few layers inside the boundary,
- the deeper interior carries only a much weaker component.

Band decomposition turned that into a stable observable-design problem. The early sparse ladder selected `[2,6)` as the best common annulus. The later dense ten-rate ladder broadened the defensible band to `[2,10)` and simultaneously lowered the pooled exponent from about `0.510` to about `0.311`.

That dense-ladder update is one of the most important results in the repo. It showed that the original `~0.51` confined value was a sparse-ladder effect, not the final confined answer.

### 6. Anchoring Dependence And Size Dependence

Anchoring dependence is non-monotonic on the fixed `[2,10)` readout:

- `W = 1e-5`: about `0.273`
- `W = 3e-6`: about `0.311`
- `W = 1e-6`: about `0.096`

The preferred annulus also moves with anchoring strength. Stronger anchoring broadens the shell-adjacent signal, the intermediate branch gives the cleanest fixed `[2,10)` baseline, and the weakest branch leaves only a shallow near-shell remnant. That is strong evidence that the confined result is boundary-conditioned.

The first larger-droplet controls do not support bulk recovery either.

Point 17 showed that at matched rate the `200^3` run redistributes defects deeper inward while reducing the relative importance of the active shell-adjacent annulus.

Point 18 then showed that the rate-sensitive part of the `200^3` signal still remains shell-adjacent rather than bulk-dominated. The strongest sparse-ladder band stays `[2,6)`, support becomes smaller, and the bulk exponent remains small. The only superficially bulk-like feature is that the fixed `[2,10)` sparse-ladder exponent rises back near `0.509`, but point 15 already proved that sparse ladders can artificially support such values even at `100^3`.

So the correct reading is not “size restores textbook KZ.” The correct reading is that the signal remains shell-conditioned, its support becomes more selective at larger size, and the present `200^3` exponent story is still entangled with sparse-ladder bias.

### 7. Timing And Observation Windows To Quote Going Forward

The repo now supports a precise operational statement about timing.

In every branch, the observation time is defined relative to the logged crossing of the imposed temperature ramp through the reference transition temperature. The code does not directly compute the Kibble-Zurek freeze-out time `t_hat`.

The branch-specific readouts are therefore chosen by stability and observable coherence rather than by claiming a directly measured freeze-out time:

- periodic `XY`: the meaningful current readout is the later final-state vortex-line density,
- periodic bulk `LdG`: the meaningful current readout is the final-state bulk defect-line density,
- confined droplet: the meaningful current readout is a late transient shell-subsurface window rather than a whole-volume final-state observable.

### 8. Current Accomplishment And Active Conclusion

The present accomplishment of the repo is not merely that it found some nonzero confined fit. It established a full validation ladder from textbook-like benchmark physics to the real confined droplet problem.

The clean unconfined Kibble-Zurek signal is reproducible in periodic systems. In confined droplets, that signal is suppressed, redistributed, and localized rather than globally expressed in the same way.

The active paper-facing conclusion should therefore be carried forward in this order:

1. periodic `XY` proving ground: benchmark topology-aware KZ signal recovered,
2. periodic bulk `LdG`: the same overall observable philosophy survives in an unconfined nematic `Q`-tensor solver and passes fixed-`dt` refinement checks,
3. confined droplet: the strongest measurable rate-dependent response is not a whole-volume late-final-state bridge, but a localized boundary-conditioned late-window signal carried by the fixed annulus `[2,10)`.

The active confined baseline is still the dense `100^3`, `W = 3e-6` branch with fixed `[2,10)` readout and pooled exponent `alpha ~ 0.311`.

Future writing should treat the larger-size and anchoring studies as refinements of that confined-shell picture rather than as evidence that the droplet simply tends back to the bulk law.

## Reproduction

All production figures above were regenerated directly from reduced artifacts already present in the repo. No new simulation reruns were required for this figure set.

Use this command to regenerate the full production ladder, including the size-`200^3` controls:

```bash
"/media/spectre71/850 EVO/myFiles/Programming/VS/MAGISTERIJ/QSR/venv/bin/python" tools/generate_production_review_figures.py --output-dir pics/production
```

If you intentionally want only the benchmark-to-dense-`100^3` ladder, omit the size-`200^3` panels with:

```bash
"/media/spectre71/850 EVO/myFiles/Programming/VS/MAGISTERIJ/QSR/venv/bin/python" tools/generate_production_review_figures.py --output-dir pics/production --skip-size200
```