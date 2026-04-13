# Changelog

This file tracks the main solver, validation, and tooling changes since the initial commit. Entries are ordered by version, with the latest changes at the bottom.

Historical output paths preserve the directory names that were in use when a branch was run. The surrounding prose describes the physical role of each branch so the workflow remains reproducible without relying on legacy shorthand labels alone.

## 07-04-2026

### v0.1.0

* Unified droplet geometry in physical space.
    * The droplet center now uses the true grid midpoint via the $(N-1)/2$ convention together with axis-aware physical semiaxes.
    * One shared geometry helper now defines the droplet instead of several slightly different index-space approximations.
* Reused the same geometry in all major droplet-dependent paths.
    * Weak anchoring, anchoring-energy evaluation, semi-implicit evolution masks, strong-boundary application, and radiality all use the same physical droplet definition.
    * Host-side initialization, shell-mask construction, average-$S$ sampling, and quench-instability diagnostics were updated to the same geometry model.

### v0.1.1

* Reworked weak-anchoring thickness and normalization so $W$ is tied to the same geometry used for the shell mask.
    * The normalization now uses $\delta_{eff} = V_{shell}/A_{ellipsoid}$, with fallback to the nominal shell-band width $2\delta_{half}$.
    * The same effective shell thickness is reused in shell diagnostics, anchoring-energy reduction, and weak-anchoring update paths.
* Added reproducible stochastic initialization support.
    * The solver now accepts `random_seed`, so paired validation runs no longer drift between reruns.
    * This also enabled stable coarse/fine weak-anchoring mesh checks.

### v0.1.2

* Enforced a fixed-`dt` contract for quench runs in the CUDA solver.
    * Legacy `enable_adaptive_dt` config values are now ignored with an explicit runtime note.
    * The old adaptive quench branch was removed from the instability guard, so the solver no longer mutates `dt` mid-protocol.
    * If the quench instability guard trips, the run aborts instead of trying to rescue the protocol by shrinking `dt`.
* Corrected the interpretation of this point update.
    * This change makes quench protocols physically interpretable and suitable for protocol-convergence testing.
    * Production Kibble-Zurek data still requires a paired fixed-`dt` refinement check rather than a single run.

### v0.1.3

* Added a reusable 2D defect-density convergence metric and promoted it into stopping logic.
    * Quench mode now uses energy plus defect-proxy stability for early stopping.
    * Single-temperature runs now use the same pattern.
    * Radiality remains logged and printable, but it is diagnostic only and no longer decides convergence by itself.
* Extended single-temperature logs to carry defect information.
    * If the chosen slice does not yet contain enough ordered plaquettes above `defects_S_threshold`, the defect proxy remains unavailable and stopping stays disabled instead of pretending topology has converged.

### v0.1.4

* Made shell order an explicit boundary policy instead of implicitly inheriting it from the initialization amplitude.
    * `S0` is now treated as an initialization parameter only.
    * Quench, sweep, and single-temperature modes all resolve shell order through the same policy.
    * Strong anchoring, weak anchoring, and anchoring-energy evaluation now see the same `S_shell` source.
* Added explicit boundary-order overrides.
    * `boundary_order_mode = equilibrium` uses the equilibrium shell order.
    * `boundary_order_mode = custom` uses an explicit `boundary_S` value.

### v0.1.5

* Moved all stable-`dt` estimation into one helper used by every simulation mode.
    * Quench, sweep, and single-temperature runs now share the same elastic and bulk stiffness assumptions.
    * In anisotropic LdG mode, the elastic cap is based on `max(|L1|, |L2|, |L1+L2|, |L1+2L2|, |L3| S_scale)` with one common anisotropic tightening factor.
    * The bulk cap is based on the same conservative `|A| + |B| S_scale + |C| S_scale^2` rate estimate across all modes.
* Preserved the semi-implicit quench path without breaking the shared estimator.
    * The quench path still relaxes only the explicit elastic remainder, but it now does so through the same helper.

### v0.1.6

* Upgraded the correlation-length guard to use anisotropic elasticity and the actual physical droplet geometry.
    * The guard now builds an anisotropic elastic $\xi$ band from positive $L_1/L_2$ combinations together with the $|L_3| S$ nematic envelope.
    * It evaluates resolution against each physical grid spacing and the actual droplet semi-axes.
    * Runtime reporting now focuses on the worst-resolved axis.
    * Non-interactive runs abort cleanly on guard failure instead of falling through to a prompt.

### v0.1.7

* Extended the reduced regression path so single-temperature and quench runs expose the same observable family.
    * Both single-temperature submodes now log bulk, elastic, anchoring, total, radiality, time, average $S$, max $S$, 2D defect density, and the 2D xi-gradient proxy.
    * Added reduced regression configs for a hedgehog relaxation and an across-transition quench.
* Unified the observable plumbing across workflows.
    * The xi-proxy implementation is shared between quench and single-temperature modes.

### v0.1.8

* Added a reproducible weak-anchoring mesh-convergence harness.
    * `tools/weak_anchor_mesh_check.py` now runs paired coarse/fine configs and writes `mesh_metrics.csv` plus `mesh_comparison.csv`.
    * The reduced validation configs use a fixed `random_seed` so the comparison is not polluted by stochastic initialization drift.
* Hardened the harness against single-temperature log-schema changes.
    * It now accepts both the legacy `free_energy,time` schema and the current `total,time` schema.

### v0.1.9

* Added a fixed-`dt` quench protocol-convergence harness for Kibble-Zurek work.
    * `tools/check_quench_protocol_convergence.py` runs an ordered coarse-to-fine set of quench configs, computes the $T_c$ crossing time from `quench_log.dat`, and compares post-crossing observables at the coarse run's recorded offsets.
    * The harness writes `protocol_metrics.csv`, `protocol_offsets.csv`, and `protocol_comparison.csv`.
* Added matched reduced protocol-convergence configs.
    * The fine validation case halves `dt` and doubles the relevant iteration counts so the physical cooling schedule is preserved.
    * Reduced validation confirms the tooling and the fixed-`dt` comparison workflow; the remaining production step is to repeat the full quench with the same refinement rule on the long-run configs.

### v0.1.10

* Brought the Python launcher and plotting stack back in sync with the rewritten CUDA backend.
    * `GUI.py` now exposes `random_seed`, `boundary_order_mode`, `boundary_S`, and `defect_density_abs_eps`.
    * The obsolete quench adaptive-`dt` control was removed from the GUI, and the help text now matches the fixed-`dt` quench contract and defect-aware stopping logic.
* Fixed run-output and plotting integration around current solver behavior.
    * The GUI now resolves the actual output directory by simulation mode, preserves the launch config outside overwriteable quench output folders, and stores a clean config copy back into the final run directory.
    * `QSRvis.py` now accepts both legacy and current single-temperature log schemas, handles the optional `anchoring` column, and chooses the correct single-temperature versus quench fallback path when plotting.

## 09-04-2026

### v0.2.0

* Added a whole-volume defect evaluation workflow for defect-rich quench analysis.
    * `QSRvis.py` now has a reusable `evaluate_volume_defects(...)` routine that combines 3D core/skeleton defect densities with per-slice x/y/z localization profiles.
    * The evaluation writes a text summary, a per-slice CSV, and a profile figure so whole-volume Kibble-Zurek candidates can be compared against boundary-localization diagnostics.
* Added a dedicated `Evaluation` page to the GUI.
    * `GUI.py` now exposes the whole-volume evaluator beside `Plot`, with controls for the 2D slice threshold, 3D core thresholds, and boundary-band diagnostic.
    * The new tab embeds the saved localization figure and report directly in the GUI instead of requiring ad hoc shell/Python postprocessing.
* Added explicit fixed-`dt` validation configs for the defect-rich confined-droplet baseline.
    * `configs/validation/sanity3_protocol_coarse.cfg` and `configs/validation/sanity3_protocol_fine.cfg` preserve the same baseline physics while making the coarse/fine time-step refinement explicit.

### v0.2.1

* Added a transient Kibble-Zurek sweep baseline for the defect-rich confined-droplet branch.
    * `configs/validation/sanity3_transient_kz_base.cfg` keeps the validated baseline physics and boundary conditions fixed while switching to `snapshot_mode = 2` with an early-stop KZ capture window.
    * This workflow is intended for measuring 3D defect observables at fixed offsets after `T_c`, before late-time coarsening drives all runs to the same final anchored state.
* Hardened 3D KZ aggregate output naming.
    * `QSRvis.aggregate_kz_scaling_3d(...)` now includes the chosen defect proxy in its CSV/plot filenames so skeleton-based and core-based transient analyses no longer overwrite each other.

### v0.2.2

* Added a dated merged-analysis record for the dense confined transient-bridge study.
    * `validation/sanity3_rate_sweep_merged_analysis/findings_2026-04-10.md` summarizes the reverse-sign transient scaling result, the active measurement window, and the next simulation priorities for finding the canonical KZM branch.
* Added a reproducible figure helper for the merged transient bridge analysis.
    * `tools/plot_confined_transient_bridge_summary.py` renders the per-run crossover and cohort-level slope decay into `pics/confined_transient_bridge_summary_2026-04-10.{png,pdf}`.

### v0.2.3

* Added a weak-anchoring transient KZ screening baseline for the canonical-sign search.
    * `configs/validation/sanity3_transient_kz_weak_anchor_base.cfg` keeps the dense transient capture and defect-rich baseline material parameters, but switches from strong radial confinement to solver-supported weak anchoring (`W = 1e-5`) so the next screening branch can test whether reduced shell locking restores the standard KZM ordering.

### v0.2.4

* Added a dated weak-anchoring analysis record for the first canonical-sign screening branch.
    * `validation/sanity3_rate_sweep_weak_anchor_screen/findings_2026-04-10.md` records that weak anchoring opens a narrow canonical-sign whole-volume window around `after_Tc_s = 3.4e-8`, but the sign reverses again by `3.6e-8`, so the branch is promising but not yet conference-ready as a stable textbook KZM result.
* Pruned obsolete strong-confinement raw payloads after reduction to durable analysis artifacts.
    * Raw outputs from the completed strong-anchoring transient sweeps and the defect-rich baseline protocol-convergence pair were deleted while preserving generated configs, sweep plans, reduced CSVs, and dated findings notes.

### v0.2.5

* Audited the QSRvis 3D transient-analysis path used for the weak-anchor screen.
    * Fixed aggregated CSV run labeling for sweep layouts where raw data live in per-case `output/` subdirectories, so future 2D/3D aggregation tables keep unique case labels instead of repeated `output` rows.
    * Added `scikit-image` to `requirements.txt`, since the whole-volume skeleton proxy depends on 3D skeletonization.
    * Added `tools/analyze_kz_3d_screen.py` so future transient screens can be summarized reproducibly with explicit morphology presets instead of one-off terminal snippets.
* Added weaker-anchoring transient baselines below the original `W = 1e-5` branch.
    * `configs/validation/sanity3_transient_kz_weak_anchor_W3em6.cfg` and `configs/validation/sanity3_transient_kz_weak_anchor_W1em6.cfg` preserve the dense transient KZ capture while reducing the radial shell penalty for the next anchoring-strength screen.

### v0.2.6

* Added a dated audit record for the QSRvis 3D weak-anchor analysis path.
    * `validation/sanity3_rate_sweep_weak_anchor_screen/qsrvis_audit_2026-04-11.md` records that the stored default weak-anchor CSVs are numerically reproducible, but the apparent `34 ns` canonical-sign window is not robust under a stricter morphology preset.
* Added the first sub-`1e-5` weaker-anchoring findings note.
    * `validation/sanity3_rate_sweep_weak_anchor_W3em6_screen/findings_2026-04-11.md` records that `W = 3e-6` strengthens the permissive default low-`S` volume signal, but the conservative interior-core proxy goes identically to zero at `34 ns` and `36 ns`, so the branch still does not provide a morphology-robust canonical KZ defect result.

### v0.2.7

* Made the active transient-KZ baselines self-contained with an explicit `T_star = 308.0`.
    * This removes a hidden dependence on the solver default and makes the origin of `Tc_KZ = 310.2` reproducible from the config itself.
* Added droplet-normalized 3D KZ proxies to `QSRvis.py` and the scripted screen analyzer.
    * `aggregate_kz_scaling_3d(...)` can now separate whole-system low-`S` volume effects from defect content inside the ordered droplet via `core_droplet` and `skeleton_droplet` metrics.
* Added a dated note tying the `Tc` sanity check to the point-two observable reanalysis.
    * `validation/point2_tc_and_droplet_normalized_reanalysis_2026-04-11.md` records that `Tc_KZ = 310.2` is consistent with the solver’s bulk `T_NI`, but the positive weak-anchor whole-system signal disappears once the observable is normalized by droplet volume.

### v0.2.8

* Added optional transient Q-tensor snapshots for KZ runs via `save_qtensor_snapshots = true` in `QSR.cu`.
    * In `snapshot_mode = 2`, the solver can now write `Qtensor_output_iter_*.dat` alongside `nematic_field_iter_*.dat`, which is required for transient biaxial-core analysis.
* Extended `QSRvis.aggregate_kz_scaling_3d(...)` and `defect_line_metrics_3d_from_field_file(...)` for strict biaxial-core proxies.
    * New proxy families include `core_biaxial`, `core_biaxial_droplet`, `skeleton_biaxial`, and `skeleton_biaxial_droplet`.
    * Snapshot selection now works with either nematic-field or transient Q-tensor files.
* Extended `tools/analyze_kz_3d_screen.py` for event-based transient analysis.
    * The scripted screen analyzer now supports fixed `avg_S` milestones (`--measure-mode avg_s`) in addition to fixed `after_Tc_s` offsets.
* Added a targeted weaker-anchoring transient-Q-tensor probe config.
    * `configs/validation/sanity3_transient_kz_weak_anchor_W3em6_qtensor_probe.cfg` runs the `W = 3e-6` branch with transient Q-tensor snapshots enabled.
* Added a dated validation note covering point 1 and the first strict biaxial-core probe.
    * `validation/point1_point2_event_based_and_biaxial_probe_2026-04-11.md` records that event-based timing alone does not restore a positive weak-anchor core trend, while a strict biaxial-core proxy reveals a later positive window that is still sensitive to morphology handling.

### v0.2.9

* Added a distance-based shell-excluded interior-region mode for transient 3D core observables.
    * `QSRvis.defect_line_metrics_3d_from_field_file(...)` and `aggregate_kz_scaling_3d(...)` now accept `core_region_mode` and `shell_exclude_layers`, and they export those settings in the per-run CSV metadata.
    * `tools/analyze_kz_3d_screen.py` now exposes `--core-region-mode` and `--shell-exclude-layers` for reproducible batch reanalysis.
* Added a dated follow-up result for the `W = 3e-6` transient Q-tensor probe under the stronger interior mask.
    * `validation/point1_point2_event_based_and_biaxial_probe_2026-04-11.md` now records that the default strict positive window collapses under shell-excluded distance masking: the strict biaxial-core proxy is zero at `avg_S = 0.1` for all rates and survives only in the slowest run at `avg_S = 0.15`, so no defensible power-law fit remains.

## 10-04-2026

### v0.3.0

* Added a separate periodic 3D XY Kibble-Zurek proving-ground branch.
    * `KZM_prooving_ground.cu` and `KZM_prooving_ground.cuh` implement a fixed-`dt`, periodic-boundary, Model-A TDGL benchmark with a continuous transition, intended to test whether the code path can recover a textbook KZM power law without confinement or anchoring.
    * The branch writes a `quench_log.dat` schema compatible with the existing sweep tooling, so `tools/quench_rate_sweep.py` can drive the new binary via `--binary ./KZM_prooving_ground_cuda`.
* Added proving-ground configs and dedicated scaling analysis.
    * `configs/validation/kzm_prooving_ground_xy_smoke.cfg` and `configs/validation/kzm_prooving_ground_xy_base.cfg` provide reduced and baseline periodic-XY setups.
    * `tools/analyze_kzm_prooving_ground.py` fits proving-ground observables against both `\tau_Q` and cooling rate and compares them against the expected KZM slopes for the chosen `\nu`, `z`, and defect codimension.
* Tightened the proving-ground topology observable to avoid counting disordered-phase phase noise as vortices.
    * The vortex plaquette count is now gated by `defect_amp_threshold`, so only locally ordered plaquettes contribute to the topology metric.
* Added a dated validation note for the new branch.
    * `validation/kzm_prooving_ground_2026-04-11.md` records that the branch and workflow are operational, while the reduced `32^3` smoke sweep is still only a workflow validation and not yet the final asymptotic exponent test.

### v0.3.1

* Added a reusable benchmark figure for the proving-ground XY branch and a short figure note.
    * `tools/plot_xy_kzm_benchmark_figure.py` now regenerates the log-log figure for the first non-smoke `64^3` result.
    * `pics/kzm_prooving_ground_xy_kzm_2026-04-11.png`, `pics/kzm_prooving_ground_xy_kzm_2026-04-11.pdf`, and `validation/kzm_prooving_ground_xy_figure_notes_2026-04-11.md` capture what the figure represents and why the final-state readout is the meaningful current benchmark signal.
* Added the first matched coarse/fine protocol check for the XY proving ground.
    * `configs/validation/kzm_prooving_ground_xy_protocol_coarse.cfg` and `configs/validation/kzm_prooving_ground_xy_protocol_fine.cfg` refine the same physical protocol at `dt = 0.02` and `dt = 0.01`.
    * `validation/kzm_prooving_ground_xy_protocol_convergence/` now records the first coarse/fine result, with identical `Tc` crossing time and modest differences in total energy, `<S>`, defect density, and `xi_grad_proxy`.
* Added the periodic bulk unconfined Landau-de Gennes intermediate branch.
    * `KZM_bulk_ldg.cu` and `KZM_bulk_ldg.cuh` implement a periodic Q-tensor solver that reuses the active `QSR` bulk convention while removing droplet confinement and anchoring.
    * The branch logs topology-aware plaquette defect density and a full-volume `xi_grad_proxy`, so it can be driven by the same config-driven tooling style used for the XY proving ground.
* Added initial configs and a dated smoke-validation note for the periodic bulk LdG branch.
    * `configs/validation/kzm_bulk_ldg_smoke.cfg` and `configs/validation/kzm_bulk_ldg_base.cfg` provide the first reduced and baseline setups.
    * `validation/kzm_bulk_ldg_2026-04-11.md` records that the first smoke run initially stayed isotropic until the post-ramp hold was lengthened, after which the periodic bulk branch reached a strongly ordered nematic state and became ready for its first rate study.

### v0.3.2

* Added a dedicated analysis path for the periodic bulk-LdG intermediate branch.
    * `tools/analyze_kzm_bulk_ldg.py` now screens final-state, fixed-after-`T_c`, and matched-`avg_S` readouts on the periodic bulk rate sweep without reusing XY-specific argument names.
* Ran the first seven-rate periodic bulk-LdG scan and recorded its outputs.
    * `validation/kzm_bulk_ldg_rate_sweep_initial/` now contains the generated configs, seven successful run outputs, sweep metrics, and the first window-screen analyses.
    * The current best readout is the final-state `defect_line_density`, with slope vs `tau_Q` about `-0.515`, correlation about `-0.990`, and a `~2.9x` spread across the first seven-rate scan.
* Added the first bulk-LdG benchmark figure and folded the first rate-study interpretation into the dated validation note.
    * `tools/plot_bulk_ldg_benchmark_figure.py` regenerates `pics/kzm_bulk_ldg_initial_scan_2026-04-11.png` and `.pdf`.
    * `validation/kzm_bulk_ldg_2026-04-11.md` now documents the first-defect timing band, the window scan, and why the final-state bulk defect readout is currently the most defensible bridge signal before returning to confinement.

### v0.3.3

* Extended the quench protocol-convergence harness so branch-specific logged observables can be compared without cloning the tool.
    * `tools/check_quench_protocol_convergence.py` now accepts `--extra-observables` and writes `final_state_comparison.csv`, which lets the periodic bulk branch compare `defect_line_density` directly at the matched final state.
* Added the first matched coarse/fine protocol pair for the periodic bulk-LdG branch and ran it.
    * `configs/validation/kzm_bulk_ldg_protocol_coarse.cfg` and `configs/validation/kzm_bulk_ldg_protocol_fine.cfg` preserve the same physical cooling rate by halving `dt` and doubling the iteration counts.
    * `validation/kzm_bulk_ldg_protocol_convergence/` now records identical `Tc` crossing times and final offsets after `Tc`, with the final-state `defect_line_density` differing by only about `0.75%` and the final `defect_density_per_plaquette` by about `0.74%`.
* Recorded the interpretation of the first bulk coarse/fine result.
    * The only large relative defect mismatch appears at the first discrete defect turn-on, where the topological signal is still one quantum wide; the late-time bulk bridge readout remains refinement-stable.

### v0.3.4

* Added a second matched coarse/fine protocol pair for a slower bulk-LdG quench taken directly from the first seven-rate scan.
    * `configs/validation/kzm_bulk_ldg_protocol_ramp600_coarse.cfg` and `configs/validation/kzm_bulk_ldg_protocol_ramp600_fine.cfg` preserve the analyzed `ramp600` physical protocol while halving `dt` in the fine run and doubling the corresponding iteration counts.
* Ran the slower bulk protocol-convergence check and recorded the result.
    * `validation/kzm_bulk_ldg_protocol_convergence_ramp600/` now records identical `Tc` crossing times and final offsets after `Tc`, with exact final-state agreement in both `defect_line_density` and `defect_density_per_plaquette`.
* Strengthened the branch interpretation for the bulk bridge readout.
    * The slower matched pair reduces the worst aligned post-`Tc` defect onset mismatch to about `14%`, down from the earlier one-quantum `~50%` onset mismatch, while leaving the late-time final-state defect readout unchanged under refinement.

### v0.3.5

* Added a reusable final-state confined-droplet analysis path.
    * `tools/analyze_confined_final_state.py` now wraps `QSRvis.aggregate_kz_scaling_3d(..., measure='final')` and summarizes final confined defect metrics against both `t_ramp` and rate, together with matched final `avg_S` and final offsets after `Tc`.
* Ran the first confined comparison using the final-state logic selected by the bulk branch.
    * `validation/sanity3_rate_sweep_weak_anchor_screen/analysis_final/` and `validation/sanity3_rate_sweep_weak_anchor_W3em6_screen/analysis_final/` now contain final-state 3D confined metrics for `skeleton_droplet` and `core_droplet` under both default and conservative morphology presets.
* Recorded the main confined result.
    * In the current retained weak-anchor KZ windows, the final droplet-normalized defect content is flat across rate under the default preset (`skeleton_droplet = 0.27904151`, `core_droplet = 0.17621395` for every run in both `W = 1e-5` and `W = 3e-6`) and collapses to zero under the conservative preset. The active confined branch therefore does not retain a measurable bulk-style final-state scaling signal; the observed rate dependence remains transient.

### v0.3.6

* Added a reusable GUI option-7-style quench GIF renderer across the current validation ladder.
    * `tools/render_option7_style_gif.py` reuses `QSRvis.create_nematic_field_animation(...)` directly and converts periodic `xy_field_iter_*.dat` and `q_tensor_iter_*.dat` snapshots into the `nematic_field_iter_*.dat` format expected by the GUI animation path.
* Added representative periodic animation configs that preserve previously analyzed protocols while enabling sparse snapshot capture.
    * `configs/validation/kzm_prooving_ground_xy_anim_ramp300.cfg` mirrors the analyzed XY `ramp300` case with `snapshotFreq = 50`.
    * `configs/validation/kzm_bulk_ldg_anim_ramp600.cfg` mirrors the analyzed bulk-LdG `ramp600` case with `snapshotFreq = 500`.
* Documented the command path used to render branch-matched quench GIFs.
    * `README.md` now records the periodic rerun commands and the three shared renderer commands used for the XY, periodic bulk-LdG, and confined weak-anchor comparison GIFs.

### v0.3.7

* Added a dedicated no-early-stop weaker-anchor final-state baseline for the confined late-hold check.
    * `configs/validation/sanity3_finalstate_kz_weak_anchor_W3em6.cfg` keeps the active `W = 3e-6` branch and fixed-`dt` protocol but disables `kz_stop_early` and drops snapshot capture so the option-1 final-state test stays focused on the true late endpoint.
* Ran the option-1 long-hold weaker-anchor sweep and reduced it with the confined final-state analysis path.
    * `validation/sanity3_rate_sweep_weak_anchor_W3em6_finalhold/` now contains the four-rate no-early-stop sweep outputs, `rate_sweep_metrics.csv`, and `analysis_final/confined_final_summary.csv` for the same `skeleton_droplet` and `core_droplet` readouts used in the earlier final-state bridge step.
* Recorded the outcome of the long-hold final-state check.
    * Extending the weaker-anchor branch to later final offsets after `T_c` (about `7.48e-08` to `8.39e-08 s`) does not recover a bulk-style final-state confined scaling signal: the default droplet-normalized final metrics remain effectively flat across rate (`skeleton_droplet` ratio `1.00000186`, `core_droplet` ratio `1.00000182`), and the conservative preset remains identically zero.

### v0.3.8

* Added a dedicated summary figure for the option-2 confined follow-up on the no-early-stop weaker-anchor branch.
    * `tools/plot_qsr_transient_log_window_followup.py` now turns the existing `analysis_option2_log_windows/transient_log_window_summary.csv` output into a compact figure showing the defect-vs-rate window scan, the scalar-order lag collapse, and the recommended `50-60 ns` follow-up band.
    * The regenerated figure is written to `pics/sanity3_weak_anchor_finalhold_option2_2026-04-11.{png,pdf}`.
* Recorded the first dated note for the option-2 transient-log branch on the finalhold sweep.
    * `validation/point5_option2_transient_log_followup_2026-04-11.md` captures two key points: the apparent strong canonical-sign `40 ns` window is still dominated by ordering lag, while a weaker but more state-matched positive 2D defect signal survives at `50-60 ns` after `T_c`.
* Clarified what does and does not count as an option-2 observable on the finalhold branch.
    * Because `validation/sanity3_rate_sweep_weak_anchor_W3em6_finalhold/` was run with `snapshot_mode = 0`, any 3D `measure='after_Tc'` remeasurement on field files is degenerate and collapses to the same late final state. The meaningful option-2 continuation is therefore the transient quench-log window analysis unless a new snapshot-rich rerun is launched.

### v0.3.9

* Added a dedicated snapshot-rich rerun config for the confined `50-60 ns` transient follow-up.
    * `configs/validation/sanity3_transient_kz_weak_anchor_W3em6_qtensor_window60.cfg` keeps the validated `W = 3e-6` fixed-`dt` weak-anchor branch and raises `kzExtraIters` from `5000` to `5500` so the KZ snapshot horizon extends just beyond `60 ns` after `T_c`.
* Recorded the launch path for the next confined 3D transient dataset.
    * `validation/point6_window60_qtensor_followup_2026-04-11.md` explains why the earlier Q-tensor probe stopped too early, documents the four-rate `quench_rate_sweep.py run` command, and records the ramp300 pilot check that confirmed live `Qtensor_output_iter_*` and `nematic_field_iter_*` writes before the full sweep was launched.
* Brought the README forward to the new active continuation branch.
    * `README.md` now documents the `validation/sanity3_rate_sweep_weak_anchor_W3em6_qtensor_window60/` production path as the direct next step after the option-2 log-window result.

### v0.3.10

* Reduced the completed snapshot-rich `50-60 ns` rerun on the active weaker-anchor branch.
    * `validation/sanity3_rate_sweep_weak_anchor_W3em6_qtensor_window60/` now has the finished `rate_sweep_metrics.csv` plus new `analysis_log_windows/`, `analysis_3d_default/`, `analysis_3d_default_wholesystem/`, `analysis_3d_conservative/`, and `analysis_3d_biaxial_default/` outputs for the actual `50 ns` and `60 ns` windows after `T_c`.
* Recorded the main result of the new confined late-transient 3D follow-up.
    * `validation/point7_window60_3d_transient_followup_2026-04-11.md` shows that the slice-based 2D defect signal survives in the cleaner `50-60 ns` band, but the tested 3D global low-`S` observables do not: default droplet-normalized low-`S` proxies are already flat by `50 ns` and exactly flat by `60 ns`, while conservative and strict biaxial droplet proxies are zero.
* Updated the README to reflect the resolved observable question.
    * The `50-60 ns` snapshot-rich rerun is now documented as a successful operational check that still yields a negative answer for the current confined 3D bridge observables.

# 11-04-2026

### v0.3.11

* Added a localized confined follow-up tool for the surviving late-transient signal.
    * `tools/analyze_midplane_slab.py` measures a contiguous midplane slab defect density by integrating the existing nematic defect plaquette observable across `2h+1` central slices selected from real post-`T_c` snapshots.
* Ran the localized slab analysis on the completed `50-60 ns` snapshot-rich rerun.
    * `validation/sanity3_rate_sweep_weak_anchor_W3em6_qtensor_window60/analysis_midplane_slab/` now contains summary and per-window CSVs for slab half-widths `0,1,2,4` at `50 ns` and `60 ns` after `T_c`.
* Recorded the refined physical interpretation of the late-transient confined signal.
    * `validation/point8_midplane_slab_followup_2026-04-11.md` shows that the positive defect-vs-rate trend survives over a finite-thickness equatorial slab even though the global 3D low-`S` and strict-biaxial observables remain flat or zero.
* Updated the README with the localized-observable continuation.
    * The confined `50-60 ns` result is now documented as a quasi-3D equatorial slab signal rather than a whole-droplet 3D low-`S` bridge observable.

### v0.3.12

* Added a full axial profile analysis for the confined late-transient signal.
    * `tools/analyze_qsr_z_profile.py` computes per-slice defect-density scaling over the selected post-`T_c` snapshots and writes both slice summaries and symmetric slab-width scans.
* Ran the axial profile analysis on the completed `50-60 ns` snapshot-rich rerun.
    * `validation/sanity3_rate_sweep_weak_anchor_W3em6_qtensor_window60/analysis_z_profile/` now contains per-slice CSVs, slab-scan CSVs, and profile plots for the `50 ns` and `60 ns` windows.
* Refined the localization story for the surviving confined signal.
    * `validation/point9_z_profile_followup_2026-04-11.md` shows that the slice-based signal remains positive even when integrated over the full `z` extent, while the strongest individual slice trends are displaced away from the exact midplane toward shell-adjacent layers.
* Updated the README with the axial-profile conclusion.
    * The main unresolved localization question is now radial or shell-versus-interior structure, not axial slab thickness.

### v0.3.13

* Added a shell-depth localization analysis for the confined late-transient signal.
    * `tools/analyze_shell_depth.py` bins XY defect plaquettes by their minimum inward distance from the filled droplet interface and writes both fixed shell-depth summaries and shell-excluded interior scans.
* Ran the shell-depth analysis on the completed `50-60 ns` snapshot-rich rerun.
    * `validation/sanity3_rate_sweep_weak_anchor_W3em6_qtensor_window60/analysis_shell_depth/` now contains per-bin CSVs, shell-exclusion scan CSVs, and profile plots for the `50 ns` and `60 ns` windows.
* Refined the radial interpretation of the surviving confined signal.
    * `validation/point10_shell_depth_followup_2026-04-11.md` shows that the strongest rate dependence sits a few layers inside the interface rather than in the outermost shell skin, while a weaker positive trend still survives in the deep interior. The remaining localization question is therefore sub-surface versus bulk weighting, not a binary shell-versus-core split.
* Pruned the raw `window60` outputs after reduction to durable analysis artifacts.
    * The four `cases/*/output/` directories now keep only `quench_log.dat` plus the exact `50 ns` and `60 ns` snapshot pairs used by the analyses, reducing the retained transient payload from about `80G` to about `1.5G`.

### v0.3.14

* Added a shell-band decomposition analysis on top of the saved shell-depth outputs.
    * `tools/analyze_shell_band_decomposition.py` scans all contiguous shell-depth bands, chooses a single common focus annulus across multiple post-`T_c` windows, compares it against the outer skin and deeper bulk, and writes a band-center depth-moment diagnostic.
* Ran the shell-band decomposition on the active `50-60 ns` branch.
    * `validation/sanity3_rate_sweep_weak_anchor_W3em6_qtensor_window60/analysis_shell_depth/band_decomposition/` now contains the per-window band scans, common-band ranking, focus-region summaries, and depth-moment summaries.
* Refined the confined late-transient observable from a generic shell-depth profile to a specific focus annulus.
    * `validation/point11_shell_band_decomposition_2026-04-11.md` shows that the best common focus band is `[2,6)`, not the outermost shell. The deeper bulk keeps only a weak positive density trend and loses defect share with rate, while the defect-weighted mean shell depth shifts outward as rate increases.
* Updated the README with the focus-annulus interpretation.
    * The current best confined bridge observable is the full-`z` XY defect plaquette density restricted to the `[2,6)` sub-surface annulus, with the bulk response treated as a weaker complement rather than the main signal.

### v0.3.15

* Added direct summary artifacts for the point-11 focus-annulus result.
    * `tools/plot_shell_focus_summary.py` generates `pics/qsr_shell_focus_summary_2026-04-11.png` and `.pdf` together with a compact skin/focus/bulk table under `validation/sanity3_rate_sweep_weak_anchor_W3em6_qtensor_window60/analysis_shell_depth/band_decomposition/`.
* Added a retained-snapshot feasibility audit for neighboring-window checks.
    * `tools/audit_qsr_snapshot_window_feasibility.py` reports when requested post-`T_c` windows alias onto the same retained snapshots, so temporal follow-up analyses cannot silently collapse onto the same files.
* Documented the current blocker on the suggested `45/55/65 ns` stability test.
    * `validation/point12_focus_summary_and_window_audit_2026-04-11.md` records that the trimmed `window60` outputs only preserve two distinct late snapshots per run, so the neighboring-window comparison now requires restored or rerun Q-tensor snapshots rather than another post-processing pass.

### v0.3.16

* Added a dedicated rerun config for real neighboring-window stability checks.
    * `configs/validation/sanity3_transient_kz_weak_anchor_W3em6_qtensor_window65.cfg` keeps the active four-rate weak-anchor branch fixed while extending the post-`T_c` snapshot horizon far enough to preserve distinct `45/50/55/60/65 ns` Q-tensor fields.
* Added a one-command follow-up driver for the shell-focus stability workflow.
    * `tools/run_qsr_neighbor_window_followup.py` runs the retained-snapshot feasibility audit, shell-depth analysis, shell-band decomposition, and focus-summary plotting for a completed sweep root.
* Generalized the focus-summary figure generator for arbitrary late-window sets.
    * `tools/plot_shell_focus_summary.py` now supports the five-window `45/50/55/60/65 ns` analysis layout instead of being limited to the earlier two-window `50/60 ns` case.
* Completed the real neighboring-window stability test on the rerun branch.
    * `validation/sanity3_rate_sweep_weak_anchor_W3em6_qtensor_window65/` now contains distinct late-window shell-depth outputs and band-decomposition outputs across all four rates, with zero sweep failures.
* Upgraded the confined-bridge conclusion from a two-window result to a stable late-window result.
    * `validation/point13_neighbor_window_stability_2026-04-11.md` shows that the best common focus band remains `[2,6)` across genuine `45/50/55/60/65 ns` windows, while skin weakens with time and bulk stays only weakly positive in density and negative in defect-share slope.

### v0.3.17

* Added a dedicated confined focus-band exponent analysis.
    * `tools/analyze_shell_focus_exponent.py` turns the validated `[2,6)` shell-focus annulus into a pooled exponent estimate, writes per-window and common-fit CSV summaries, and produces a publication-style exponent figure.
* Extracted the first explicit exponent from the stable confined bridge observable.
    * `validation/point14_focus_exponent_2026-04-12.md` reports that the focus-band density follows approximately `rho_[2,6) ~ tau_Q^-0.510`, with per-window late-time fits spanning `0.486-0.539` across real `45/50/55/60/65 ns` windows.
* Added the necessary controls to show the exponent is genuinely localized.
    * The same pooled late-window analysis gives common density exponents of about `0.360` in the outer skin and `0.086` in the deeper bulk, confirming that the `[2,6)` annulus is the region carrying the confined bridge scaling signal.

### v0.3.18

* Added a dated dense-ladder refinement note for the active five-window weak-anchor branch.
    * `validation/point15_dense_rate_ladder_refinement_2026-04-12.md` records that the matched ten-rate refinement passes the distinct-window audit with zero sweep failures, broadens the best common annulus from `[2,6)` to `[2,10)`, and lowers the pooled confined density exponent from about `0.510` to about `0.311`.
* Documented the sparse-vs-dense control that explains the exponent shift.
    * Restricting the dense branch back to the original four runs reproduces the old `[2,6)` exponent (`0.5095`) essentially exactly, so the change is caused by rate-ladder densification rather than branch drift or post-processing mismatch.
* Updated the project narrative to use the dense branch as the active confined baseline.
    * The recommended reference point for future anchoring or crossover studies is now the stable `[2,10)` annulus on `validation/sanity3_rate_sweep_weak_anchor_W3em6_qtensor_window65_dense`, with `alpha ~ 0.311` quoted as the current defensible confined-bridge exponent.

## 12-04-2026

### v0.4.0

* Added fixed-band point-2 tooling for the anchoring-strength follow-up.
    * `tools/analyze_shell_focus_exponent.py` now accepts explicit focus-band index overrides, so new branches can be compared on the same `[2,10)` late-window annulus instead of silently drifting to different per-branch bands.
    * `tools/run_qsr_anchor_strength_followup.py` now runs the dense rate sweep, neighboring-window shell analysis, and fixed-band exponent extraction for a supplied weak-anchor config in one reproducible step.
    * `tools/compare_qsr_anchor_strength_point2.py` summarizes completed weak-anchor sweeps at fixed `[2,10)` readout and writes a cross-`W` CSV/markdown table plus figure.
* Added matched five-window configs for the first anchoring-strength comparison around the active baseline.
    * `configs/validation/sanity3_transient_kz_weak_anchor_W1em5_qtensor_window65.cfg` and `configs/validation/sanity3_transient_kz_weak_anchor_W1em6_qtensor_window65.cfg` preserve the active dense `45/50/55/60/65 ns` Q-tensor protocol while varying only `W` away from the completed `W = 3e-6` baseline.
* Added a dated anchoring-strength interpretation note for the fixed-band point-2 follow-up.
    * `validation/point16_anchor_strength_fixed_band_2026-04-12.md` records that the confined `[2,10)` exponent is non-monotonic in `W`, with `W = 3e-6` as the cleanest intermediate-confinement branch between shell domination and shell-loss collapse.
* Added dedicated `200^3` size-sensitivity pilot configs around the active `W = 3e-6` branch.
    * `configs/validation/sanity3_transient_kz_weak_anchor_W3em6_qtensor_window65_size200_logonly.cfg` preserves the full `200^3` physics with logs only for a cheap completion check.
    * `configs/validation/sanity3_transient_kz_weak_anchor_W3em6_qtensor_window65_size200_sparse100.cfg` enables only `nematic_field_iter_*.dat` KZ snapshots every `100` iterations, keeping the late-window timing quantization below about `0.6 ns` while holding retained `200^3` payloads to a manageable size.
* Added the first matched-rate `200^3` size-comparison note against the active `100^3` dense baseline.
    * `validation/point17_size200_matched_rate_shell_depth_2026-04-12.md` records that, at matched `ramp_iters = 50`, the larger droplet raises total late defect density but weakens the relative weight of the shell-adjacent `[2,10)` annulus and shifts defect localization deeper into the interior. The note also makes explicit that this single-run pilot is not yet enough to extract a new `200^3` exponent.
* Added low-storage pruning support for large transient KZ sweeps.
    * `tools/prune_qsr_snapshots_to_offsets.py` trims iterator snapshots down to the nearest retained files for requested after-`T_c` windows, and `tools/quench_rate_sweep.py run` now exposes this through `--retain-offsets` and `--retain-iter-prefixes` so large `200^3` sweeps can be pruned case-by-case instead of keeping full transient payloads.
* Added the first pruned `200^3` four-rate ladder result and comparison note.
    * `validation/point18_size200_four_rate_ladder_2026-04-13.md` records that the larger droplet still selects the shell-adjacent `[2,6)` annulus as the strongest rate-sensitive region, with a steeper sparse-ladder exponent (`alpha ~ 0.649`) and much smaller support fraction than `100^3`. The same note also makes explicit that the fixed-band `[2,10)` value (`alpha ~ 0.509`) cannot yet be interpreted as a decisive size effect because sparse-ladder bias remains entangled with geometry.

## 13-04-2026

### v0.4.1

* Added a consolidated repo-review note tying the benchmark, bulk, and confined findings into one paper-facing narrative.
    * `validation/consolidated_validation_review_2026-04-13.md` collects the dated validation notes and relevant changelog context into one complete summary of what the repo established, how the branches fit together, how the observation timing should be described, and what the active scientific conclusion now is.
    * The note records the current active confined baseline as the dense `100^3`, `W = 3e-6`, `[2,10)` late-window annulus with `alpha ~ 0.311`, while keeping the periodic XY and periodic bulk-LdG branches explicit as the clean unconfined benchmark ladder.

### v0.4.2

* Pruned bulky transient and final-state raw field payloads after reduction to durable analysis products.
    * Retained `quench_log.dat`, generated configs, sweep plans, reduced CSV/Markdown outputs, figures, and dated validation notes while removing archived `*_iter_*.dat` snapshots and raw `*_final.dat` field dumps that can be regenerated with targeted reruns.
* Renamed the main analysis and plotting scripts to describe their purpose without project-specific prefixes.
    * The affected tools now use generalized names such as `tools/analyze_confined_final_state.py`, `tools/analyze_shell_depth.py`, `tools/analyze_shell_focus_exponent.py`, and `tools/check_quench_protocol_convergence.py`.
* Normalized the primary documentation for external reproducibility.
    * `README.md` now presents a concise user manual, and the changelog prose now describes physical baselines and benchmark roles instead of relying on internal branch shorthand.

### v0.4.3
* Added executables to `exe` folder for ease of use and updated `README.md` to reflect the new paths.
    * The `exe` folder now contains `KZM_prooving_ground_cuda`, `KZM_bulk_ldg_cuda`, and `QSR_cuda` executables for the periodic XY, periodic bulk-LdG, and confined QSR branches, respectively.
    * The `README.md` now documents the new executable paths for running the different branches and their associated configs.
* Update `GUI.py` to default to the new `exe/QSR_cuda` path and added a note about the change in the README.

### v0.4.4

* Unified the post-processing tool entry points between `QSRvis.py` and the GUI.
    * New shared tool metadata now live in `tool_catalog.py`, so the registered confined, benchmark, and protocol-analysis scripts are described once and reused consistently.
    * `QSRvis.py` now exposes wrapper functions for the registered `tools/` modules and adds interactive menu option `13` as a shared tool launcher.
* Replaced the old single-purpose GUI evaluation page with a general `Tools` workspace.
    * `GUI.py` now lists the registered post-processing utilities, shows a definition/help panel plus editable argument template, and captures stdout/stderr into the GUI while previewing generated text and figure artifacts.
    * This keeps the project’s accumulated analysis scripts accessible from the main launcher instead of leaving them as disconnected standalone entry points.

### v0.4.5

* Reorganized the main GUI configuration tabs around workflow meaning instead of legacy file-group names.
    * `GUI.py` now surfaces sectioned tabs for `Workflow`, `Geometry`, `Material`, `Elastic & Boundary`, `Solver`, `Protocols`, and `Diagnostics`, with mode-specific descriptions inside each section.
    * Iteration limits, print/log cadence, tolerances, sweep controls, and quench controls are now grouped with the solver or protocol settings they actually affect instead of being left under broad legacy tabs.
* Replaced the numeric plot-mode dropdown labels with descriptive titles.
    * The `Plot` tab selector now shows the actual action names such as final-state slices, quench-log plots, and aggregate KZ scaling instead of only mode numbers.
    * The detailed `Mode options` pane remains the place where the numeric QSRvis mode identity is explained for users who still want to map the GUI back to the script interface.

### v0.4.6

* Completed the GUI 1.0.0 wrap-up pass with dynamic mode-aware organization.
    * `GUI.py` now hides or reveals solver sections and field rows based on the active `sim_mode`, `protocol`, `init_mode`, snapshot mode, and boundary-order choices, so the launcher stops presenting obviously irrelevant controls at the same time.
    * The plot-mode `Mode options` headings were normalized to one consistent naming style while preserving the numeric QSRvis mode identifiers for script-level cross-reference.
* Added a dedicated `About` tab backed by a centralized version catalog.
    * `version_catalog.py` now defines the displayed versions for `QSR GUI` (`1.0.0`), `QSR_cuda` (`0.4.5`), `QSR_cpu` (`prealpha`), `KZM_prooving_ground_cuda` (`1.0.0`), and `KZM_bulk_ldg_cuda` (`1.0.0`).
    * `GUI.py` reads that catalog for the window title and About tab, so future version bumps only need one source-of-truth update.

### v0.4.7

* Added a production review-figure generator driven by the existing reduced validation artifacts.
    * `tools/generate_production_review_figures.py` regenerates a paper-facing figure ladder into `pics/production/` without rerunning the underlying simulations.
    * The current production set covers the periodic `XY` benchmark, the periodic bulk `LdG` bridge, bulk protocol refinement, confined whole-volume control plots, confined shell localization, the fixed `[2,10)` confined exponent, and the fixed-band anchoring comparison.

### v0.4.8

* Integrated the production-review figure ladder into the shared post-processing interfaces.
    * `tool_catalog.py`, `GUI.py`, and `QSRvis.py` now expose `tools/generate_production_review_figures.py` through the main `Tools` workflow instead of leaving it as a standalone script.
    * The tool definition text now states that the figure ladder requires already reduced artifacts and is intended for the final paper-facing review pass rather than unfinished raw sweeps.
* Extended the production figure set with the current `200^3` controls and consolidated the review back to one maintained root document.
    * `tools/generate_production_review_figures.py` now also regenerates the matched-rate `200^3` redistribution figure and the sparse-ladder `200^3` comparison figure, both from existing reduced artifacts.
    * `consolidated_validation_review_2026-04-13.md` is now the single maintained consolidated review, with the embedded production ladder including the size-`200^3` controls; the temporary validation-folder duplicate was removed.
* Polished the launcher around the production-review workflow.
    * `version_catalog.py` bumps the GUI to `1.0.1`.
    * `GUI.py` now moves the `About` tab to the far right, hides `L_stab` and `jacobi_iters` when IMEX is off, and omits those keys from generated configs when semi-implicit stepping is disabled so stale hidden values do not leak into runs.