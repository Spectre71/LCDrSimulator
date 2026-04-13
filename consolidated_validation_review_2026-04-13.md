# Consolidated Validation Review And Research Conclusions - 2026-04-13

## Purpose

This file consolidates the dated validation notes and the relevant changelog context into one narrative that can be used as the starting point for paper writing, repo cleanup, and figure planning.

The goal here is not to replace the detailed point notes. It is to state, in one place, what the repo actually established, how the separate branches fit together, and what the present scientific accomplishment is.

## Source trail used for this consolidation

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

## Executive summary

The repo now supports three distinct statements, and they should not be mixed together.

1. The codebase can recover Kibble-Zurek-like scaling when the physics is appropriate.
   - The periodic 3D XY proving-ground branch gives the first clean benchmark result.
   - The periodic bulk Landau-de Gennes branch then shows that a Q-tensor nematic model without confinement can also produce a clean, refinement-stable scaling readout.

2. The confined droplet is not just the bulk problem with finite resolution.
   - In the confined branch, whole-volume and late-final-state observables do not reproduce the clean bulk-like signal.
   - The dominant confined response is boundary-conditioned and geometry-conditioned.

3. The confined system still contains a measurable Kibble-Zurek-like signal, but not where the original whole-volume search looked for it.
   - The surviving signal is not a robust whole-droplet 3D core observable.
   - It is a localized late-transient shell-subsurface signal.
   - The active dense confined baseline is the `100^3`, `W = 3e-6`, late-window annulus `[2,10)` with pooled exponent about `0.311`.

That means the repo no longer tells a story of solver failure. It tells a story of regime separation:

- clean benchmark KZ in unconfined systems,
- clean bridge behavior in periodic bulk nematic LdG,
- suppressed or redistributed global KZ behavior in confined droplets,
- and a localized shell-conditioned confined KZ signal that only becomes visible after the observable is redesigned around the real confined geometry.

## Consolidated narrative

### 1. Why the proving-ground and bulk branches were necessary

The confined `QSR` droplet solver contains too many ingredients at once for a clean first benchmark:

- first-order isotropic-nematic LdG dynamics,
- finite geometry,
- anchoring,
- boundary-controlled ordering,
- and geometry-dependent post-processing.

That is why the repo first split the problem into cleaner branches instead of trying to force the droplet solver to serve as the benchmark.

#### 1.1 Periodic XY proving ground

The periodic 3D XY branch exists to answer the narrow question: can the infrastructure recover a textbook Kibble-Zurek power law when the model, symmetry, and boundary conditions are the right ones?

What it established:

- The reduced smoke sweep validated the solver branch, configuration contract, topology observable, and rate-sweep tooling.
- The first non-smoke `64^3` periodic XY sweep showed that the final-state vortex-line observable has the correct sign, strong monotonicity, and a slope close to the expected 3D XY Model-A value.
- The matched coarse/fine protocol check showed that the benchmark branch is not just fitting noise.

What matters scientifically:

- The branch demonstrates that the framework can produce a recognizable Kibble-Zurek signal when the physical setting is a continuous transition in a periodic system.
- This removes the idea that the later confined difficulties must automatically mean the numerics are broken.

#### 1.2 Periodic bulk Landau-de Gennes branch

The bulk periodic LdG branch exists to bridge from the XY benchmark back to a Q-tensor nematic model without reintroducing confinement.

What it established:

- The periodic bulk branch orders properly once the post-ramp hold is long enough.
- In the first seven-rate sweep, the final-state `defect_line_density` already gives a strong monotonic slope against `tau_Q`, around `-0.515`, with strong log-log correlation.
- Two matched coarse/fine protocol checks showed that the final-state bulk readout is stable under refinement, with the only visibly larger mismatch occurring at the first discrete defect turn-on, not at the final state.

What matters scientifically:

- This branch shows that even within a nematic Q-tensor model, one can recover a clean scaling story when confinement and anchoring are removed.
- So the later confined result must be interpreted as a real geometry-and-boundary problem, not simply as a generic failure of LdG numerics.

### 2. What the early confined search actually found

The original strong-confinement `sanity3` transient bridge search did not produce the canonical sign expected from a bulk-like picture.

#### 2.1 Strong confinement branch

The merged strong-confinement analysis established that, in the original radial strong-boundary regime, slower quenches transiently retain more 3D structure in the early post-`T_c` window. That is the opposite sign from the canonical Kibble-Zurek expectation.

This was not a one-point accident:

- the slow-end bridge confirmed the trend beyond a single outlier,
- the effect was strongest around `3.4e-8` to `3.6e-8 s` after `T_c`,
- and it decayed by about `4.0e-8` to `4.4e-8 s`.

The important interpretation was already visible there: the active `sanity3` droplet regime is confinement-dominated. The shell and boundary structure compete with, and in that branch reverse, the simple bulk-like Kibble-Zurek ordering.

#### 2.2 First weak-anchoring screen

Weak anchoring at `W = 1e-5` was the first solver-supported boundary change that opened a short canonical-sign window in the whole-volume core metric near `34 ns` after `T_c`.

That was useful, but not yet trustworthy:

- the sign already reversed again by `36 ns`,
- the skeleton metric was even more short-lived,
- and the later audit showed that the positive `34 ns` signal depended strongly on morphology settings.

So the first weak-anchor screen mattered because it showed the right search axis, not because it already solved the confined problem.

#### 2.3 QSRvis audit and the `W = 3e-6` screen

The code-path audit confirmed that the 3D analysis implementation itself was consistent. What failed was the robustness of the observable interpretation.

Then the weaker `W = 3e-6` screen made the permissive whole-volume low-`S` signal look stronger, but the conservative preset collapsed to zero.

That result was crucial because it separated two claims:

- yes, the default low-`S` volume proxy can produce a positive early-time signal,
- no, that is not yet the same thing as a morphology-robust confined defect-scaling law.

### 3. What timing and normalization clarified

The repo then resolved two interpretation issues that would otherwise contaminate any later paper conclusion.

#### 3.1 `Tc_KZ` is not a computed freeze-out time

The current workflow does not compute the literature freeze-out time `t_hat`.

What it actually does is:

- use `Tc_KZ` as a transition-temperature marker for snapshot placement and post-processing alignment,
- determine the actual `T_c` crossing time from `quench_log.dat`,
- and then measure at fixed physical offsets after that crossing or at a branch-dependent alternative readout such as final state or matched order.

The note on `Tc_KZ = 310.2` also showed that this value is physically consistent with the model: it is just the rounded bulk coexistence temperature implied by the active coefficients, not an arbitrary parameter.

#### 3.2 Droplet-normalized observables removed a misleading positive signal

When the early weak-anchor observables were normalized by system volume, the slower branch could look artificially more defect-rich simply because the ordered droplet itself was still developing.

Once the observable was normalized by the actual ordered droplet volume, that apparent positive whole-volume core signal disappeared.

This mattered because it showed that part of the earlier positive trend was not true defect enrichment inside the ordered droplet. It was delayed droplet establishment.

### 4. Why whole-volume and final-state confined observables were rejected

At that stage, several tempting but misleading confined observables were explicitly tested and rejected.

#### 4.1 Event-based timing and strict biaxial cores

Reanalyzing the early weak-anchor screens at fixed `avg_S` milestones did not restore a robust canonical core trend.

The stricter low-`S` plus biaxiality proxy could produce a positive default result on the transient Q-tensor probe, but that result collapsed when the interior region was defined more physically by shell-excluded distance rather than permissive morphological dilation.

So event-based timing alone did not rescue the old core result, and the first strict-core success did not survive a stronger interior definition.

#### 4.2 Final-state confined bridge observable

The periodic bulk branch identified a clean final-state readout. That immediately raised the natural question of whether the same logic would survive in confinement.

It did not.

Point 3 showed that on the early-stop weak-anchor sweeps, the final retained confined 3D defect metrics were flat across rate or zero under the stricter preset.

Point 4 then removed the obvious loophole by running a true no-early-stop final-hold branch. Even then:

- the final logged times moved later in a real way,
- the 2D final midplane defect signal changed with rate,
- but the whole-volume droplet-normalized final 3D metrics stayed effectively flat or zero.

So the confined branch does not preserve the same kind of final-state bridge observable that works in the periodic bulk branch.

#### 4.3 Global 3D transient observables in the cleaner late window

Point 5 used log-window screening on the no-early-stop `W = 3e-6` branch to identify the cleaner post-critical band.

The very strong `40 ns` signal was rejected as too contaminated by ordering lag. The more defensible branch was the weaker but cleaner `50-60 ns` band.

Point 6 then created the snapshot-rich rerun needed to evaluate real 3D fields in that band.

Point 7 finally answered the question directly:

- the 2D logged signal remains modestly positive in the `50-60 ns` window,
- but the tested global 3D low-`S` and strict-biaxial observables are already flat or zero there.

That is the key negative result: the confined signal survives, but not as a global 3D defect observable of the type initially targeted.

### 5. How the correct confined observable was found

The later confined work succeeded because it stopped asking the whole droplet to behave like the periodic bulk branch and instead localized the observable in the geometry where the signal actually lives.

#### 5.1 Equatorial slab and full axial profile

Point 8 showed that the surviving late-transient signal is not just one special slice. A finite-thickness equatorial slab still carries a positive defect-vs-rate trend.

Point 9 then showed that axial thickness is not the real discriminator either. The slice-based signal remains positive even when integrated over the full `z` range, while the strongest individual slices are displaced toward shell-adjacent regions rather than sitting exactly at the midplane.

That shifted the problem from axial localization to radial localization.

#### 5.2 Shell-depth analysis and band decomposition

Point 10 measured the signal directly in inward distance from the droplet interface.

That result did two important things:

- it showed that the strongest slopes are not in the outermost skin but a few layers inside the boundary,
- and it showed that the deeper interior still carries a weaker positive component.

Point 11 then converted that profile into a stable observable design problem and selected the best common multibin focus annulus.

The conclusion was that the best confined signal is not the outer skin and not the deep bulk. It is the sub-surface annulus `[2,6)`.

Just as important, the defect-share diagnostics showed that the rate dependence is not a simple whole-profile amplification. It is a redistribution of defect weight outward toward that sub-surface annulus.

#### 5.3 Neighboring-window stability

Point 12 found a real blocker: after the first cleanup, the retained `window60` snapshots were too sparse to test `45/55/65 ns` honestly because the requested windows aliased onto the same files.

That led to the `window65` rerun and the one-command follow-up path.

Point 13 then showed that the `[2,6)` annulus remains the best common focus band across genuine `45/50/55/60/65 ns` windows. That turned the shell-localized signal from a two-window hint into a real late-window result.

### 6. How the confined exponent changed once the sampling was good enough

#### 6.1 First explicit confined exponent

Point 14 extracted the first exponent from the validated localized observable. On the original four-rate `window65` ladder, the `[2,6)` focus annulus gave a pooled exponent around `0.510`, with skin and bulk both clearly weaker.

That was the first explicit confined-bridge exponent and it was physically useful, but it still came from only four rates.

#### 6.2 Dense rate-ladder refinement changed the baseline

Point 15 is one of the most important notes in the repo because it showed that the original `~0.51` result was a sparse-ladder effect, not the final confined answer.

On the dense ten-rate ladder:

- the preferred annulus broadened from `[2,6)` to `[2,10)`,
- the pooled exponent dropped from about `0.510` to about `0.311`,
- and restricting the dense branch back to the original four runs reproduced the old `0.5095` almost exactly.

That means the old value was not wrong as a report of the sparse dataset. It was incomplete. The denser ladder resolved the turnover structure and changed the physically defensible baseline.

This is the active confined baseline now:

- branch: `validation/sanity3_rate_sweep_weak_anchor_W3em6_qtensor_window65_dense`
- preferred band: `[2,10)`
- preferred dense exponent: `alpha ~ 0.311`

### 7. What anchoring and size dependence revealed

#### 7.1 Anchoring-strength dependence is non-monotonic

Point 16 compared `W = 1e-5`, `3e-6`, and `1e-6` on the same fixed `[2,10)` readout.

The result is non-monotonic:

- `W = 1e-5`: about `0.273`
- `W = 3e-6`: about `0.311`
- `W = 1e-6`: about `0.096`

The auto-selected bands also move in informative ways:

- stronger anchoring broadens the signal toward a large shell-adjacent region,
- the intermediate branch keeps the clean `[2,10)` localization,
- and the weakest branch leaves only a shallow near-shell remnant.

This is strong evidence that the confined result is boundary-conditioned. The exponent and the preferred shell region both depend on anchoring.

#### 7.2 First larger-droplet controls do not support bulk recovery

Point 17 used a single `200^3` matched-rate run to ask a narrow localization question. That result showed:

- total late defect density increases modestly,
- but the relative importance of the active shell-adjacent `[2,10)` annulus decreases,
- and defect localization shifts deeper inward.

That by itself already argued against a simple “larger droplet just strengthens the same shell result” story.

Point 18 then ran the first pruned `200^3` four-rate ladder. That result is subtle but decisive in one respect:

- the strongest rate-sensitive band still remains shell-adjacent at `[2,6)`, not in the deep bulk,
- the bulk exponent remains small,
- so the larger droplet still does not behave like bulk recovery.

At the same time, the fixed `[2,10)` exponent at `200^3` comes out near `0.509`, which is numerically reminiscent of the old sparse `100^3` result. But point 15 already proved that sparse ladders can artificially support such values.

So the correct reading is not “size restores textbook KZ.” The correct reading is:

- the rate-sensitive signal remains shell-conditioned,
- its support becomes smaller at larger size,
- and the present `200^3` exponent story is still entangled with sparse-ladder bias.

## What this means for timing and observation windows

The repo now supports a precise and defensible statement about how the observation time was chosen.

### 1. What is and is not being measured

- `T_c` or `Tc_KZ` in the current workflow is a transition-temperature marker used to define the observation origin.
- The actual crossing time is reconstructed from `quench_log.dat`.
- The current solver and post-processing do not directly compute the literature freeze-out time `t_hat`.

So the observation rule is operational rather than claiming a direct measured freeze-out time.

### 2. Operational timing rule by branch

For all branches, the repo first identifies the crossing time `t_c` from the logged temperature history. The measurement is then taken at a branch-dependent readout chosen for physical rather than cosmetic reasons.

The branch logic is now clear:

- Periodic XY proving ground:
  - the early fixed-after-`T_c` window is too flat,
  - the first stable KZ-sensitive observable is the final-state vortex-line density,
  - so that becomes the benchmark readout.

- Periodic bulk LdG:
  - fixed absolute-time windows near onset are too fragile,
  - matched-order windows are cleaner but weaker,
  - the final-state bulk defect-line density is the strongest stable readout,
  - so that becomes the bridge readout.

- Confined droplet:
  - the early `40 ns` transient signal is too contaminated by ordering lag,
  - the cleaner late window is identified from the log-window scan first,
  - then the real snapshot-rich rerun validates the `50-60 ns` band,
  - then the neighboring-window rerun expands that to a stable `45-65 ns` late-window band,
  - and only after that is the signal restricted to the shell-subsurface region where it is actually localized.

### 3. Recommended paper language on timing

The safest paper-facing statement is:

"In all branches, the observation time was defined relative to the logged crossing of the imposed temperature ramp through the reference transition temperature. The analysis does not directly compute the Kibble-Zurek freeze-out time `t_hat`; instead it uses branch-specific post-critical observation windows chosen by stability and observable coherence. In the periodic benchmark and periodic bulk branches, the cleanest readout is the later final state. In the confined droplet, the cleanest readout is a late transient shell-subsurface window rather than a global final-state observable." 

That statement is aligned with what the code and notes actually do.

## How the pieces fit together physically

The repo now supports one coherent physical picture.

### 1. Unconfined systems

When confinement and anchoring are removed, the code can recover clean Kibble-Zurek-like behavior:

- first in the 3D XY proving ground,
- then in the periodic bulk Q-tensor nematic bridge.

This means the infrastructure is capable of seeing the expected physics when the geometry and boundary conditions do not dominate the dynamics.

### 2. Confined droplets

Once real boundaries and anchoring are present, the dominant regime changes.

The important point is not that the microscopic model becomes something else. The point is that the observed dynamics are co-determined by:

- finite spatial extent,
- shell ordering,
- anchoring strength,
- access of defects to the boundary,
- and geometry-conditioned annihilation and redistribution.

That is why:

- the whole-volume confined 3D observables fail,
- the final-state bridge fails,
- the deeper bulk remains weak,
- and the strongest rate-sensitive signal lives in a shell-subsurface annulus instead.

### 3. The confined result is therefore not "no KZ"

The confined result is more precise than a simple negative statement.

It is:

- not a clean whole-system bulk-like KZ law,
- not a morphology-robust early whole-volume core law,
- not a final-state bulk bridge,
- but yes, a localized late-transient shell-conditioned KZ-like scaling signal.

That is a stronger and more interesting conclusion than either "the solver failed" or "the confined system behaves just like bulk." 

## What these notes accomplished

Taken together, the validation notes accomplished six concrete things.

1. They validated the infrastructure in clean benchmark settings.
   - The repo can recover expected KZ-like behavior in periodic, unconfined systems.

2. They validated a nematic Q-tensor bridge without confinement.
   - The periodic bulk LdG branch behaves as a controlled bridge between the XY benchmark and the confined droplet problem.

3. They falsified several tempting but incorrect confined observables.
   - Early whole-volume low-`S` cores, strict-core claims, final-state confined 3D observables, and late global 3D transient proxies were all tested and either shown to be flat, zero, or morphology-sensitive.

4. They identified the correct confined observable and timing strategy.
   - The surviving confined signal is late-transient, localized, and shell-subsurface.

5. They quantified how that confined signal depends on sampling, anchoring, and size.
   - The rate ladder density matters.
   - Anchoring dependence is non-monotonic.
   - Increasing droplet size does not move the dominant signal into the bulk.

6. They established the active confined baseline that should be quoted going forward.
   - Dense `100^3`, `W = 3e-6`, real `45-65 ns` late windows, fixed annulus `[2,10)`, exponent about `0.311`.

## Current accomplishment

Based on the notes consolidated here, the present accomplishment of the repo is the following.

### 1. Benchmark accomplishment

The repo now contains a complete validation ladder from textbook-like benchmark physics to the confined droplet problem:

- periodic XY proving ground,
- periodic bulk Landau-de Gennes bridge,
- confined droplet branch.

That means the final confined interpretation is anchored by successful benchmark branches rather than by isolated droplet fits.

### 2. Confined-physics accomplishment

The confined droplet work has moved beyond the vague statement that confinement "makes things messy." It has identified what specifically happens:

- strong confinement reverses the naive whole-volume transient sign,
- weak anchoring alone does not restore a robust global defect law,
- the apparent positive early signal is largely tied to morphology sensitivity and droplet-establishment delay,
- a real confined scaling signal survives only after the observable is localized in the shell-subsurface region,
- and the active dense confined exponent is significantly smaller than the old sparse estimate.

### 3. Paper-level accomplishment

The most important paper-facing accomplishment is this:

The repo demonstrates that Kibble-Zurek-like dynamics can be recovered cleanly in unconfined systems, but in heavily confined droplets with real boundaries they are not globally expressed in the same way. Instead, the confined signal is suppressed, redistributed, and localized into a shell-subsurface late-transient annulus.

That is the result.

Not:

- "the solver could not find KZ," and not
- "the confined system just has the same bulk law with noise."

But rather:

- unconfined systems recover the expected benchmark behavior,
- confined droplets realize a boundary-conditioned dynamical regime,
- and the correct confined observable is local, not global.

### 4. Active conclusion to carry forward

If the repo needs one concise current conclusion before article writing begins, it should be this:

The current evidence supports a boundary-conditioned confined regime rather than bulk recovery. The clean unconfined Kibble-Zurek signal is reproducible in periodic systems, but in confined droplets the strongest measurable rate-dependent response lives in a shell-subsurface annulus, not in whole-droplet observables. The active confined baseline is therefore the dense `100^3`, `W = 3e-6`, `[2,10)` late-window readout with `alpha ~ 0.311`, and future writing should treat larger-size and anchoring studies as refinements of that confined-shell picture rather than as evidence that the droplet simply tends back to the bulk law.