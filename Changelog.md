# Changelog

This file tracks the main solver, validation, and tooling changes since the initial commit. Entries are ordered by version, with the latest changes at the bottom.

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
    * `tools/quench_protocol_convergence_check.py` runs an ordered coarse-to-fine set of quench configs, computes the $T_c$ crossing time from `quench_log.dat`, and compares post-crossing observables at the coarse run's recorded offsets.
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