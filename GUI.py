from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
	import tkinter as tk
	from tkinter import filedialog, messagebox
	from tkinter import ttk
except ModuleNotFoundError as e:
	# On many Linux distros, tkinter is shipped as a separate system package
	# (it's not installable via pip reliably).
	if e.name != 'tkinter':
		raise
	print(
		"ERROR: tkinter is not available in this Python installation.\n\n"
		"On Ubuntu/Debian, install it with:\n"
		"  sudo apt-get update\n"
		"  sudo apt-get install python3-tk\n\n"
		"Then re-run:\n"
		"  python3 GUI.py\n",
		file=sys.stderr,
	)
	raise SystemExit(1)


try:
	from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
	from matplotlib.backends._backend_tk import NavigationToolbar2Tk  # type: ignore
except Exception:
	FigureCanvasTkAgg = None  # type: ignore[assignment]
	NavigationToolbar2Tk = None  # type: ignore[assignment]


@dataclass(frozen=True)
class FieldSpec:
	key: str
	label: str
	default_hint: str
	kind: str  # 'float'|'int'|'str'|'tri_bool'|'choice'
	choices: list[str] | None = None
	help_text: str | None = None


def _join_lines(*parts: str) -> str:
	return "\n".join([p.rstrip() for p in parts if p is not None and str(p).strip() != ""]).strip()


def _choice_help(spec: FieldSpec) -> str:
	if spec.choices is None:
		return ""
	# Don't include 'default' as a semantic choice (it means: omit key).
	choices = [c for c in spec.choices if c != "default"]
	if not choices:
		return ""
	return "Choices:\n" + "\n".join([f"  - {c}" for c in choices])


def _tri_bool_help() -> str:
	return _join_lines(
		"This is a tri-state option:",
		"  - default: do not write this key to the config; backend uses its built-in default",
		"  - true: force-enable the feature",
		"  - false: force-disable the feature",
	)


def _render_help(spec: FieldSpec) -> str:
	base = spec.help_text or ""
	if spec.kind == "tri_bool":
		return _join_lines(base, "", _tri_bool_help())
	if spec.kind == "choice":
		return _join_lines(base, "", _choice_help(spec))
	return base


def _repo_root() -> Path:
	return Path(__file__).resolve().parent


def _default_backend_path() -> Path:
	# Prefer local ./QSR_cuda built binary.
	return _repo_root() / "QSR_cuda"


def _write_kv_config(path: Path, items: dict[str, str]) -> None:
	lines: list[str] = []
	lines.append("# QSR_cuda config (key=value)")
	lines.append("# Lines starting with # are ignored.")
	lines.append("# Omitted keys use the backend defaults.")
	lines.append("")
	for key in sorted(items.keys()):
		lines.append(f"{key} = {items[key]}")
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_kv_config(text: str) -> dict[str, str]:
	out: dict[str, str] = {}
	for raw in text.splitlines():
		line = raw.split("#", 1)[0].strip()
		if not line:
			continue
		if "=" not in line:
			continue
		k, v = line.split("=", 1)
		k = k.strip()
		v = v.strip()
		if k:
			out[k] = v
	return out


class QSRGui(tk.Tk):
	def __init__(self) -> None:
		super().__init__()
		self.title("QSR GUI")
		self.geometry("1100x750")

		self.backend_path = tk.StringVar(value=str(_default_backend_path()))
		self.last_out_dir: Path | None = None
		self._proc: subprocess.Popen[str] | None = None
		self._reader_thread: threading.Thread | None = None
		self._log_queue: queue.Queue[str] = queue.Queue()
		self._info_visible: dict[str, bool] = {}

		# Plot tab state
		self.plot_source = tk.StringVar(value="")
		self.plot_mode = tk.StringVar(value="6")
		self.plot_out_dir = tk.StringVar(value="pics")
		self._plot_vars: dict[str, tk.Variable] = {}
		self._plot_mode_frames: dict[str, ttk.Frame] = {}
		self._plot_fig = None
		self._plot_canvas = None
		self._plot_toolbar = None
		self._plot_canvas_container = None
		self._plot_status = tk.StringVar(value="")

		self._build_ui()
		self.after(50, self._drain_log_queue)

	# ---------------- UI ----------------

	def _build_ui(self) -> None:
		top = ttk.Frame(self)
		top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

		ttk.Label(top, text="Backend executable:").grid(row=0, column=0, sticky=tk.W)
		backend_entry = ttk.Entry(top, textvariable=self.backend_path, width=80)
		backend_entry.grid(row=0, column=1, sticky=tk.EW, padx=(8, 8))
		ttk.Button(top, text="Browse…", command=self._browse_backend).grid(row=0, column=2, sticky=tk.E)
		top.columnconfigure(1, weight=1)

		controls = ttk.Frame(self)
		controls.pack(side=tk.TOP, fill=tk.X, padx=10)

		ttk.Button(controls, text="Run", command=self._on_run).pack(side=tk.LEFT)
		ttk.Button(controls, text="Stop", command=self._on_stop).pack(side=tk.LEFT, padx=(8, 0))
		ttk.Button(controls, text="Open output folder", command=self._open_output_folder).pack(side=tk.LEFT, padx=(8, 0))
		ttk.Button(controls, text="Plot last run (QSRvis)", command=self._plot_last_run).pack(side=tk.LEFT, padx=(8, 0))
		ttk.Button(controls, text="Save preset…", command=self._save_preset).pack(side=tk.RIGHT)
		ttk.Button(controls, text="Load preset…", command=self._load_preset).pack(side=tk.RIGHT, padx=(0, 8))

		self.status = tk.StringVar(value="Idle")
		ttk.Label(self, textvariable=self.status).pack(side=tk.TOP, fill=tk.X, padx=10, pady=(6, 6))

		split = ttk.Panedwindow(self, orient=tk.VERTICAL)
		split.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

		form_container = ttk.Frame(split)
		log_container = ttk.Frame(split)
		split.add(form_container, weight=3)
		split.add(log_container, weight=2)

		self.notebook = ttk.Notebook(form_container)
		self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

		self._vars: dict[str, tk.Variable] = {}
		self._specs_by_tab: dict[str, list[FieldSpec]] = {
			"General": [
				FieldSpec(
					"sim_mode",
					"Simulation mode",
					"(default: 3=Quench)",
					"choice",
					choices=["default", "1", "2", "3"],
					help_text=_join_lines(
						"Selects which high-level workflow the backend runs.",
						"- 1 (Single T): relaxes at a fixed temperature T until convergence / max iterations; writes into output/ (and energy logs depending on submode).",
						"- 2 (Sweep): runs many Single-T relaxations across a temperature range (T_start→T_end with T_step) and saves per-temperature final fields in output_temp_sweep/.",
						"- 3 (Quench): time-dependent temperature protocol (step or ramp) and writes a CSV-like quench_log.dat plus optional snapshots in out_dir.",
						"Tip: use Quench (3) for Kibble–Zurek style measurements.",
					),
				),
				FieldSpec(
					"init_mode",
					"Init mode",
					"(default: 1)",
					"choice",
					choices=["default", "0", "1", "2"],
					help_text=_join_lines(
						"Controls how the Q-tensor is initialized before relaxation.",
						"- 0 (random): random director/ordering; good for exploring metastability but can introduce extra defects.",
						"- 1 (radial): seeded radial hedgehog-like texture; usually converges fastest to the radial state.",
						"- 2 (isotropic+noise): starts near isotropic with small random perturbations (set noise_amplitude); best for KZ quenches because domains form naturally.",
					),
				),
				FieldSpec(
					"submode",
					"Single-T submode",
					"(default: 1)",
					"choice",
					choices=["default", "1", "2"],
					help_text=_join_lines(
						"Applies only when sim_mode=1 (Single T).",
						"- 1: logs total free energy + radiality.",
						"- 2: logs bulk/elastic/anchoring components separately.",
						"Use submode=2 when you are tuning elastic constants or anchoring: it makes it much easier to see which term is dominating the dynamics.",
						"Numerics note: splitting energy terms is diagnostic only; it does not change the evolution equation.",
					),
				),
				FieldSpec(
					"Nx",
					"Nx",
					"100",
					"int",
					help_text=_join_lines(
						"Grid size in x (number of lattice points).",
						"Memory/time scale roughly with Nx*Ny*Nz.",
						"Droplet radius in lattice units is ~min(Nx,Ny,Nz)/2 in the backend.",
						"Resolution guidance: keep the defect core / interfacial length scale resolved (a common heuristic is ~6–10 grid points across a core diameter).",
						"Practical: if you double Nx (with Ny,Nz), you roughly 8× memory/time for 3D volumes, and snapshots get 8× larger.",
						"If you see grid-aligned dot/checkerboard artifacts at high resolution, it is usually a dt/stiffness issue rather than ‘Nx too large’.",
					),
				),
				FieldSpec(
					"Ny",
					"Ny",
					"100",
					"int",
					help_text=_join_lines(
						"Grid size in y (number of lattice points).",
						"If Nx,Ny,Nz are not equal the droplet radius uses the smallest dimension.",
						"Best practice: keep Nx=Ny=Nz unless you intentionally want a slab/film or you are doing a convergence study.",
						"Numerics: unequal dimensions can bias which Fourier modes are most stiff and can change the ‘first’ direction to show odd-even artifacts.",
					),
				),
				FieldSpec(
					"Nz",
					"Nz",
					"100",
					"int",
					help_text=_join_lines(
						"Grid size in z (number of lattice points).",
						"Also controls the mid-plane slice index used by some 2D logging (defaults to Nz/2 in the backend).",
						"If you rely on 2D defect/xi proxies, choose Nz large enough that the mid-plane is representative (avoid Nz that is so small that the droplet ‘touches’ both z boundaries).",
						"For 3D postprocessing of defect networks, Nz sets the sampling along the symmetry axis and affects whether short disclination loops are captured.",
					),
				),
				FieldSpec(
					"dx",
					"dx (m)",
					"1e-8",
					"float",
					help_text=_join_lines(
						"Physical lattice spacing in x (meters). Also sets the physical droplet radius R≈(min(N)/2)*dx.",
						"Numerical impact: the stiffest elastic modes scale like ~1/dx^2, so explicit stability typically forces dt ∝ dx^2 unless semi-implicit stabilization is enabled.",
						"Physics impact: correlation length xi must be resolved (roughly dx ≲ xi/3) to avoid under-resolved interfaces/defect cores.",
					),
				),
				FieldSpec(
					"dy",
					"dy (m)",
					"1e-8",
					"float",
					help_text=_join_lines(
						"Physical lattice spacing in y (meters).",
						"Use dy=dx for isotropic resolution unless you have a specific reason for anisotropic spacing.",
						"Physics: dy defines the physical distance between neighboring lattice planes in y, so it affects the physical droplet size and the elastic cost of gradients in y.",
						"Numerics: anisotropic spacing effectively weights the discrete Laplacian differently by direction; if dy!=dx you can make one direction ‘stiffer’ and reduce the stable dt.",
					),
				),
				FieldSpec(
					"dz",
					"dz (m)",
					"1e-8",
					"float",
					help_text=_join_lines(
						"Physical lattice spacing in z (meters).",
						"If dz differs from dx/dy you change the effective stiffness of z-derivatives and the stability dt estimate.",
						"If you keep Nx=Ny=Nz but set dz!=dx, the grid is physically stretched; that changes the physical droplet shape only if the backend uses physical coordinates for geometry, but it always changes elastic costs.",
						"Tip: for ‘spherical droplet’ studies, set dx=dy=dz unless you are intentionally probing anisotropic resolution error.",
					),
				),
				FieldSpec(
					"iterations",
					"Iterations (single/sweep)",
					"100000",
					"int",
					help_text=_join_lines(
						"Maximum number of iterations for Single-T mode and for each temperature in Sweep mode.",
						"Quench mode uses total_iters instead.",
						"Physical time interpretation: a run of N iterations corresponds to a simulated time ~N*dt (with dt in seconds).",
						"If you increase gamma (viscosity) without changing dt, relaxation slows down in physical time; you may need more iterations for the same degree of equilibration.",
						"If convergence is slow, consider enabling early-stop (quench) or tightening/loosening tolerance depending on what you measure.",
					),
				),
				FieldSpec(
					"print_freq",
					"Print freq (single/sweep)",
					"200",
					"int",
					help_text=_join_lines(
						"How often (in iterations) the backend prints/logs diagnostics for Single-T and Sweep.",
						"Larger values reduce overhead; smaller values give finer-grained convergence tracking.",
						"Recommendation: for long runs, pick print_freq so you get ~200–2000 log lines total (enough to see trends without drowning in output).",
						"Note: printing can be a bottleneck on some systems; if throughput matters, increase print_freq and rely on file logs/snapshots.",
					),
				),
				FieldSpec(
					"tolerance",
					"Tolerance (energy)",
					"1e-2",
					"float",
					help_text=_join_lines(
						"Relative convergence threshold used in some modes: stop when |ΔF/F| < tolerance (after a minimum physical time).",
						"Smaller tolerance means stricter convergence (more iterations).",
						"This is a numerical stopping criterion; it does not guarantee topological equilibration (e.g., defect annihilation can be slow even if energy changes are small).",
						"For KZ-style statistics, you often care about a consistent ‘observation time’ after crossing Tc rather than full convergence; tune tolerance accordingly.",
					),
				),
				FieldSpec(
					"RbEps",
					"Radiality eps (RbEps)",
					"1e-2",
					"float",
					help_text=_join_lines(
						"Relative convergence threshold for the radiality metric between samples.",
						"Used by the early-stop logic in quench and by convergence checks in some single-temp outputs.",
						"Radiality is a geometry-specific order metric for the droplet texture; it is useful when the expected ground state is the radial hedgehog.",
						"If you are studying non-radial textures (e.g., escaped or bipolar states), avoid using radiality-based stopping as your primary convergence signal.",
					),
				),
			],
			"Material": [
				FieldSpec(
					"a",
					"a (J/m^3/K)",
					"0.044e6",
					"float",
					help_text=_join_lines(
						"Landau-de Gennes bulk coefficient 'a' (sets how A depends on temperature).",
						"The backend constructs A,B,C from (a,b,c,T,T_star) and the chosen bulk convention.",
						"Changing 'a' changes the bulk stiffness scale, which can strongly affect the dt_bulk stability estimate.",
					),
				),
				FieldSpec(
					"b",
					"b (J/m^3)",
					"1.413e6",
					"float",
					help_text=_join_lines(
						"Landau-de Gennes bulk coefficient 'b' (cubic term). Controls first-order character of I–N transition.",
						"Affects equilibrium order parameter S_eq(T) and bulk stiffness.",
						"Physics: larger |b| generally strengthens the discontinuity at the transition and changes the location of the coexistence temperature relative to T* (depending on convention).",
						"Numerics: because b enters the bulk molecular field nonlinearly, large b can increase stiffness near defects where S varies rapidly.",
						"Tip: when comparing KZ scaling across runs, avoid changing (a,b,c) between runs unless that is the study; they set the microscopic scales (xi, relaxation time).",
					),
				),
				FieldSpec(
					"c",
					"c (J/m^3)",
					"1.153e6",
					"float",
					help_text=_join_lines(
						"Landau-de Gennes bulk coefficient 'c' (quartic term). Ensures stability (boundedness) of the bulk energy.",
						"Affects S_eq(T) and bulk stiffness.",
						"Physics: increasing c typically reduces the equilibrium order magnitude (and increases the penalty for large S), which tends to broaden defect cores in physical units.",
						"Numerics: very large c can make the bulk relaxation stiff when S is perturbed far from equilibrium (e.g., strong noise, large dt).",
					),
				),
				FieldSpec(
					"T",
					"T (K)",
					"300",
					"float",
					help_text=_join_lines(
						"Temperature in Kelvin.",
						"In Single-T mode this is fixed; in Quench mode this is used mainly as a default for T_high if you don’t set T_high.",
						"Moving T closer to T* increases the correlation length xi; deep in nematic (lower T) reduces xi and can require smaller dx.",
					),
				),
				FieldSpec(
					"T_star",
					"T* (K)",
					"308",
					"float",
					help_text=_join_lines(
						"Reference temperature used in the bulk model (often described as A=0 / isotropic spinodal in some conventions).",
						"Sets where the bulk transition estimates occur. Changing T* shifts the phase behavior.",
						"Operationally: for ramps, Tc_KZ should be chosen relative to the resulting transition estimates in this convention; if you change T* you usually want to update Tc_KZ as well.",
					),
				),
				FieldSpec(
					"bulk_modechoice",
					"Bulk convention",
					"(default: 1)",
					"choice",
					choices=["default", "1", "2"],
					help_text=_join_lines(
						"Selects how the code converts (a,b,c,T,T*) into the bulk energy coefficients A,B,C.",
						"This affects the predicted S_eq(T), transition temperatures, and bulk stiffness.",
						"If you compare runs, keep this fixed or results won’t be directly comparable.",
						"- 1: standard convention used by this codebase",
						"- 2: alternative (Ravnik-style) convention",
					),
				),
				FieldSpec(
					"S0",
					"Initial S0",
					"(default: S_eq(T) or 0.5)",
					"float",
					help_text=_join_lines(
						"Initial scalar order magnitude used in init_mode=0/1 (random/radial).",
						"If not set, backend uses S_eq(T) when nematic is stable, otherwise a heuristic value.",
						"For init_mode=2, S0 is ignored (use noise_amplitude instead).",
						"Physics: S0 only affects the transient; the equilibrium S is set by the bulk coefficients and temperature.",
						"Numerics: extremely large S0 can make the first few steps stiff (bulk term large), which can amplify instability if dt is aggressive.",
						"Tip: for reproducible KZ runs, prefer init_mode=2; for equilibrium radial runs, keep init_mode=1 and omit S0 so it is consistent with S_eq(T).",
					),
				),
				FieldSpec(
					"T_start",
					"Sweep: T_start (K)",
					"295",
					"float",
					help_text=_join_lines(
						"Sweep mode (sim_mode=2): starting temperature.",
						"The backend runs a full relaxation at each temperature point.",
						"Use sweeps to map equilibrium textures vs temperature (e.g., how radiality, defect core size, or elastic energy changes across the transition range).",
						"Numerics: each temperature point starts from the previous converged state (typical continuation); this reduces hysteresis/noise compared to independent random starts.",
					),
				),
				FieldSpec(
					"T_end",
					"Sweep: T_end (K)",
					"315",
					"float",
					help_text=_join_lines(
						"Sweep mode (sim_mode=2): ending temperature.",
						"If T_step sign does not match (T_end - T_start), the backend flips the step sign automatically.",
						"If your goal is to bracket Tc, make sure the sweep includes both sides of the transition in your chosen bulk convention.",
						"Tip: do one coarse sweep first to locate the interesting window, then rerun a finer sweep around it.",
					),
				),
				FieldSpec(
					"T_step",
					"Sweep: T_step (K)",
					"1",
					"float",
					help_text=_join_lines(
						"Sweep mode (sim_mode=2): temperature increment between sweep points.",
						"Must be non-zero.",
						"Smaller step gives finer sampling but more runs.",
						"A good workflow is: coarse step (1–2 K) to find transitions, then fine step (0.1–0.5 K) near Tc to resolve rapid changes in S and xi.",
						"Be aware that very small steps can ‘track’ metastable branches; consider also reversing sweep direction if you want to quantify hysteresis.",
					),
				),
			],
			"Elastic": [
				FieldSpec(
					"kappa",
					"kappa (J/m)",
					"6.5e-12",
					"float",
					help_text=_join_lines(
						"One-constant elastic coefficient (isotropic elasticity).",
						"Used only when L1/L2/L3 are all zero AND Frank mapping is disabled.",
						"Bigger kappa increases elastic stiffness and typically increases the correlation length xi ~ sqrt(kappa/|A|).",
						"Numerically: larger kappa can force smaller dt unless semi-implicit stabilization is enabled.",
					),
				),
				FieldSpec(
					"use_frank_map",
					"Use Frank→LdG mapping",
					"(default: false)",
					"tri_bool",
					help_text=_join_lines(
						"If true, the backend converts Frank constants (K1,K2,K3) into LdG gradient coefficients (L1,L2,L3) using a reference order parameter S_ref.",
						"This is how you run a full 3-constant elastic model in the implemented LdG form.",
						"Important: this mapping assumes the code’s Q convention Q = S (n⊗n − I/3).",
					),
				),
				FieldSpec(
					"K1",
					"K1 (N)",
					"6.5e-12",
					"float",
					help_text=_join_lines(
						"Frank splay elastic constant (units N). Only used when use_frank_map=true.",
						"Together with K2,K3 determines L1,L2,L3 and therefore stiffness and defect energetics.",
						"Physics: K1 penalizes splay (∇·n) distortions; radial textures are splay-heavy, so increasing K1 tends to increase the cost of the radial hedgehog.",
						"Numerics: larger K1 increases elastic stiffness and can reduce the stable dt for explicit parts; semi-implicit stabilization helps mainly with the isotropic Laplacian-like component.",
					),
				),
				FieldSpec(
					"K2",
					"K2 (N)",
					"4.0e-12",
					"float",
					help_text=_join_lines(
						"Frank twist elastic constant (units N). Only used when use_frank_map=true.",
						"If K2 differs strongly from K1/K3, the mapped L2 can make the high-k dynamics stiffer and more prone to checkerboard instability for explicit stepping.",
						"Physics: K2 penalizes twist; in many droplet problems twist is weak unless boundary conditions or chirality promote it.",
						"Practical: large anisotropy (K2 ≪ K1,K3 or K2 ≫ ...) can change which defect morphologies are favorable, so don’t compare such runs directly to one-constant intuition.",
					),
				),
				FieldSpec(
					"K3",
					"K3 (N)",
					"8.0e-12",
					"float",
					help_text=_join_lines(
						"Frank bend elastic constant (units N). Only used when use_frank_map=true.",
						"K3 − K1 controls the mapped L3 term (cubic-in-Q gradient coupling), which can be numerically stiff.",
						"Physics: K3 penalizes bend; escaped/bipolar textures often trade splay vs bend to reduce the singular core energy.",
						"Numerics: if mapped L3 becomes large, watch dt and the checkerboard guard; L3 is one of the easiest ways to excite odd-even modes in explicit stepping.",
					),
				),
				FieldSpec(
					"S_ref",
					"S_ref for mapping",
					"(default: S_eq(T))",
					"float",
					help_text=_join_lines(
						"Reference scalar order parameter used in the Frank→LdG mapping.",
						"Because L2 scales like ~1/S_ref^2 and L3 like ~1/S_ref^3, changing S_ref changes the mapped L’s a lot.",
						"Recommended: keep S_ref = S_eq(T) (backend default) for consistency with the chosen bulk convention.",
					),
				),
				FieldSpec(
					"L1",
					"L1 (J/m)",
					"0",
					"float",
					help_text=_join_lines(
						"LdG elastic coefficient L1 (units J/m). Used when you want to set L1/L2/L3 directly.",
						"Setting any of L1/L2/L3 non-zero causes the backend to ignore kappa.",
						"Stability note: for L3=0, necessary boundedness conditions include L1>0 and L1+L2>0.",
						"Physics: L1 is the isotropic gradient penalty (roughly the ‘one-constant’ piece in LdG form).",
						"Numerics: this is the term best handled by the semi-implicit stabilizer (it is Laplacian-like). If you set L1 large and disable semi-implicit, dt must typically scale like dx^2/L1.",
					),
				),
				FieldSpec(
					"L2",
					"L2 (J/m)",
					"0",
					"float",
					help_text=_join_lines(
						"LdG elastic coefficient L2 (units J/m). Used when setting L1/L2/L3 directly.",
						"L2 introduces anisotropic stiffness and tends to make explicit stability stricter (more high-k stiffness).",
						"Physics: L2 weights a different invariant of ∂Q and is what lets you emulate distinct splay/twist/bend costs (together with L3).",
						"Numerics: L2 is currently treated explicitly; if L2 is large you may need a smaller dt even with semi-implicit enabled.",
					),
				),
				FieldSpec(
					"L3",
					"L3 (J/m)",
					"0",
					"float",
					help_text=_join_lines(
						"LdG elastic coefficient L3 (units J/m). Used when setting L1/L2/L3 directly.",
						"This is a Q-dependent (effectively nonlinear) elastic contribution and can be the stiffest part of the model.",
						"If you see Nyquist/checkerboard artifacts, L3 (or dt) is often implicated.",
					),
				),
				FieldSpec(
					"W",
					"Weak anchoring W (J/m^2)",
					"0",
					"float",
					help_text=_join_lines(
						"Weak anchoring strength on the shell (penalty toward the imposed boundary order).",
						"W=0 means only the strong/radial boundary condition is applied (as implemented by the boundary kernel).",
						"Larger W forces boundary alignment more strongly but can make the near-shell region stiffer.",
					),
				),
			],
			"Dynamics": [
				FieldSpec(
					"gamma",
					"gamma (Pa·s)",
					"0.1",
					"float",
					help_text=_join_lines(
						"Rotational viscosity / kinetic coefficient in the relaxational dynamics.",
						"Larger gamma slows dynamics (smaller effective D=L/gamma), often allowing larger stable dt but requiring more physical time to relax.",
						"Scaling intuition: for a simple diffusion-like mode, the relaxation time scales like τ ~ gamma * (length)^2 / L (up to convention factors).",
						"For KZ: if you change gamma, you change the microscopic relaxation time; that can affect the inferred scaling unless you account for it.",
					),
				),
				FieldSpec(
					"dt",
					"dt (s)",
					"(default: dt_max)",
					"float",
					help_text=_join_lines(
						"Time step size in seconds.",
						"The backend prints dt stability estimates (dt_diff from elastic stiffness and dt_bulk from bulk stiffness) and clamps dt to dt_max.",
						"If you set dt too large you can trigger runaway S or a Nyquist/checkerboard mode; enable the instability guard to detect that.",
						"If semi-implicit stabilization is enabled, dt_diff can become much less restrictive (sometimes effectively infinite) and dt is then limited by dt_bulk.",
					),
				),
				FieldSpec(
					"use_semi_implicit",
					"Use semi-implicit (IMEX)",
					"(default: true)",
					"tri_bool",
					help_text=_join_lines(
						"Enables a semi-implicit (IMEX) stabilizer for the isotropic Laplacian-like elastic term.",
						"Implementation: each step solves a Helmholtz-type equation via Jacobi iterations (cheap approximate implicit solve).",
						"Effect: greatly improves stability at small dx by removing the strict explicit dt ∝ dx^2 restriction for the stabilized part.",
						"Note: L2/L3 anisotropic contributions remain explicit and can still impose a dt limit.",
					),
				),
				FieldSpec(
					"L_stab",
					"L_stab",
					"(default: |L1| or |kappa|)",
					"float",
					help_text=_join_lines(
						"Stabilization coefficient used by the semi-implicit solver.",
						"Recommended: set L_stab ≈ |kappa| (one-constant mode) or ≈ |L1| (LdG L1/L2/L3 mode).",
						"The backend clamps L_stab so you don’t accidentally make the explicit remainder anti-diffusive.",
						"What it does: the code adds/subtracts a Laplacian-like term with coefficient L_stab, treating the added part implicitly and the remainder explicitly (IMEX split).",
						"If L_stab is too small, you keep most stiffness explicit and dt remains limited by dx^2; if it is too large, the explicit remainder becomes weak but you may overdamp high-k modes (especially with low Jacobi iters).",
					),
				),
				FieldSpec(
					"jacobi_iters",
					"Jacobi iters",
					"25",
					"int",
					help_text=_join_lines(
						"Number of Jacobi iterations used to approximately solve the semi-implicit Helmholtz problem per time step.",
						"More iterations = more accurate implicitness (better stability/less numerical damping) but slower per step.",
						"Typical range: 10–50; for smoke tests 3–10.",
					),
				),
			],
			"Quench": [
				FieldSpec(
					"out_dir",
					"Output directory",
					"output_quench",
					"str",
					help_text=_join_lines(
						"Directory where quench results are written.",
						"GUI behavior: if you enter a relative path, it is resolved relative to the repo root.",
						"Backend behavior: writes quench_log.dat + final fields, and possibly snapshots depending on snapshot_mode.",
						"Run provenance: the GUI also writes run_config.cfg into this directory so you can exactly reproduce the run later.",
						"Tip: prefer unique out_dir names per experiment (e.g., include dx, protocol, and a seed) and keep overwrite_out_dir=false if you want to accumulate multiple runs.",
					),
				),
				FieldSpec(
					"overwrite_out_dir",
					"Overwrite output dir",
					"(default: true)",
					"tri_bool",
					help_text=_join_lines(
						"If the output directory already exists, delete it and start fresh.",
						"Recommended for reproducible runs (avoids mixing logs/snapshots from multiple runs).",
						"If you set this to false and reuse an existing directory, be aware that postprocessing scripts might pick up old snapshots or logs unless you clean up manually.",
					),
				),
				FieldSpec(
					"protocol",
					"Protocol",
					"(default: 2)",
					"choice",
					choices=["default", "1", "2"],
					help_text=_join_lines(
						"Quench temperature protocol.",
						"- 1 (step): instant jump from T_high to T_low after pre_equil_iters.",
						"- 2 (ramp): linear ramp from T_high to T_low over ramp_iters after pre_equil_iters.",
						"KZ mode snapshots behave differently for step vs ramp (see kzItersAfterStep vs Tc_window_K).",
					),
				),
				FieldSpec(
					"T_high",
					"T_high (K)",
					"(default: T)",
					"float",
					help_text=_join_lines(
						"Initial temperature for the quench protocol.",
						"Often chosen above the transition so the system starts weakly ordered/isotropic, then quenches into nematic.",
						"For KZ-style quenches, choose T_high sufficiently above Tc that the initial state is effectively isotropic (short correlation length, no pre-existing nematic domains).",
						"If T_high is already nematic, you are effectively doing an anneal rather than a quench and defect statistics will differ.",
					),
				),
				FieldSpec(
					"T_low",
					"T_low (K)",
					"(default: T_high-5)",
					"float",
					help_text=_join_lines(
						"Final temperature for the quench protocol.",
						"Lower T typically increases S_eq and bulk stiffness, which may reduce dt_bulk and require finer dx for defect cores.",
						"Deep quenches (far below Tc) increase the driving force and can produce sharper cores and stronger gradients; that tends to demand smaller dt and/or semi-implicit stabilization.",
						"If your goal is to measure near-Tc scaling, T_low mainly matters for how long you allow coarsening after crossing Tc (unless you stop early).",
					),
				),
				FieldSpec(
					"pre_equil_iters",
					"Pre-equil iters",
					"0",
					"int",
					help_text=_join_lines(
						"Number of iterations to relax at T_high before starting the step/ramp.",
						"Useful to remove initialization artifacts before the quench begins.",
						"For init_mode=2, keeping pre_equil_iters small helps ensure domains nucleate primarily during the quench window rather than during an initial long equilibration at T_high.",
						"For init_mode=0/1, a modest pre-equil can reduce spurious high-frequency noise from initialization (helpful if you are pushing dt).",
					),
				),
				FieldSpec(
					"ramp_iters",
					"Ramp iters",
					"1000",
					"int",
					help_text=_join_lines(
						"Only used when protocol=2 (ramp). Number of iterations over which T_high→T_low is ramped linearly.",
						"Longer ramp = slower cooling rate; in KZ terms, this tends to produce fewer defects (larger domains).",
						"Cooling rate (discrete): approximately (T_high - T_low) / (ramp_iters * dt).",
						"If you change dt, you also change the physical ramp rate unless you compensate ramp_iters.",
					),
				),
				FieldSpec(
					"total_iters",
					"Total iters",
					"(default: iterations)",
					"int",
					help_text=_join_lines(
						"Total iterations executed in Quench mode.",
						"If KZ stop-early is enabled and snapshot_mode=2, the backend may stop before total_iters once it leaves the snapshot window.",
						"Think of total_iters as your ‘experiment duration’. For KZ postprocessing, you typically care about a consistent observation time relative to Tc (not necessarily full equilibration).",
						"If you disable kz_stop_early, total_iters also controls how much late-time coarsening you include (defect annihilation continues well after Tc).",
					),
				),
				FieldSpec(
					"logFreq",
					"Log freq",
					"(default: print_freq)",
					"int",
					help_text=_join_lines(
						"How often (in iterations) quench diagnostics are appended to out_dir/quench_log.dat.",
						"Smaller values give more time resolution but larger log files.",
						"For KZ: pick logFreq so that the interval in physical time (logFreq*dt) is small compared to the domain-growth timescale near Tc.",
						"Performance note: logging is much cheaper than snapshotting; don’t be afraid of logFreq ~ 50–500 if you need time resolution.",
					),
				),
				FieldSpec(
					"snapshot_mode",
					"Snapshot mode",
					"(default: 2)",
					"choice",
					choices=["default", "0", "1", "2"],
					help_text=_join_lines(
						"Controls when full nematic-field snapshots (nematic_field_iter_*.dat) are saved.",
						"- 0: no iter snapshots (final-only). Fastest and smallest disk usage.",
						"- 1 (GIF): save snapshots every snapshotFreq iterations across the whole run.",
						"- 2 (KZ): save snapshots only near Tc (around Tc_KZ ± Tc_window_K for ramps, or for a fixed window after the step).",
						"Note: snapshots can be very large for 100^3 grids; KZ mode is designed to keep disk usage manageable.",
					),
				),
				FieldSpec(
					"snapshotFreq",
					"GIF snapshot freq",
					"10000",
					"int",
					help_text=_join_lines(
						"Only used when snapshot_mode=1.",
						"Save a snapshot every N iterations.",
						"Disk usage scales like (total_iters/snapshotFreq) * Nx*Ny*Nz.",
						"Choose snapshotFreq based on a *physical* cadence: desired Δt_phys / dt.",
						"If you want smooth movies, you need snapshots frequent enough to resolve relaxation but not so frequent you spend all your time writing files.",
					),
				),
				FieldSpec(
					"Tc_KZ",
					"Tc (KZ)",
					"310.2",
					"float",
					help_text=_join_lines(
						"Transition temperature used only to decide when to record snapshots in KZ snapshot_mode=2.",
						"This does not change the physics directly; it only chooses the window of recorded frames.",
						"Pick Tc_KZ near the bulk transition reported by the backend (T_NI estimate) for your chosen bulk convention.",
					),
				),
				FieldSpec(
					"Tc_window_K",
					"Tc window halfwidth (K)",
					"0.5",
					"float",
					help_text=_join_lines(
						"Only used for ramp quenches (protocol=2) in KZ snapshot_mode=2.",
						"Snapshots are saved while T is within [Tc_KZ − Tc_window_K, Tc_KZ + Tc_window_K].",
						"Wider window = more frames saved.",
						"KZ intent: record the system while the correlation length and relaxation time are rapidly changing near Tc.",
						"If your ramp is very slow, you may want a narrower window to avoid saving many near-equilibrium frames; if your ramp is fast, a wider window can help ensure you still capture the critical region.",
					),
				),
				FieldSpec(
					"kzSnapshotFreq",
					"KZ snapshot freq",
					"1000",
					"int",
					help_text=_join_lines(
						"Snapshot cadence (iterations) during the KZ recording window.",
						"Smaller values give more temporal resolution for postprocessing but increase disk usage.",
						"Postprocessing tip: if you compute defect density and correlation length from snapshots, you typically want enough frames to fit power laws on log–log plots without being dominated by noise.",
						"Start with kzSnapshotFreq ~ 500–2000 and adjust based on runtime/disk constraints.",
					),
				),
				FieldSpec(
					"kzItersAfterStep",
					"Step: iters after step",
					"200000",
					"int",
					help_text=_join_lines(
						"Only used for step quenches (protocol=1) in snapshot_mode=2.",
						"Because step quenches have no Tc ramp, the backend records KZ snapshots for a fixed number of iterations after the step.",
						"Interpretation: kzItersAfterStep * dt is your observation time after the quench. Defect density decays during this time via annihilation/coarsening.",
						"For consistent scaling, keep this fixed across runs (or rescale it in physical units if dt differs).",
					),
				),
				FieldSpec(
					"kz_stop_early",
					"KZ stop early",
					"(default: true)",
					"tri_bool",
					help_text=_join_lines(
						"If enabled, the backend ends the run after the KZ snapshot window finishes (plus optional kzExtraIters).",
						"This saves time and disk when you only care about near-Tc states.",
						"If disabled, the run continues to total_iters.",
						"This does not change the evolution while the KZ window is active; it only changes what happens afterward.",
					),
				),
				FieldSpec(
					"kzExtraIters",
					"KZ extra iters",
					"0",
					"int",
					help_text=_join_lines(
						"Extra iterations to run after leaving the KZ snapshot window when kz_stop_early=true.",
						"Use this if you want to record near-Tc snapshots but then allow additional coarsening before stopping.",
						"If you want a single ‘late-time’ snapshot after coarsening, set kzExtraIters and also set snapshot_mode=1 for one extra snapshot period, or simply rely on the final state files.",
					),
				),
				FieldSpec(
					"kz_nu",
					"KZ: nu",
					"0.5",
					"float",
					help_text=_join_lines(
						"Critical exponent ν used for a Kibble–Zurek (KZ) / Zurek-time estimate.",
						"This is an *analysis-only* parameter: it does not affect the solver; it only affects what the GUI reports.",
						"Default 0.5 is mean-field-like; if you have literature values for your universality class, set them here.",
					),
				),
				FieldSpec(
					"kz_z",
					"KZ: z",
					"2.0",
					"float",
					help_text=_join_lines(
						"Dynamic exponent z used for a KZ / Zurek-time estimate.",
						"This is an *analysis-only* parameter: it does not affect the solver; it only affects what the GUI reports.",
						"Default 2.0 is a common starting point for relaxational (Model-A-like) dynamics; adjust if you are matching a specific theory/paper.",
					),
				),
				FieldSpec(
					"kz_tau0",
					"KZ: tau0 (s)",
					"(default: dt)",
					"float",
					help_text=_join_lines(
						"Microscopic relaxation time τ0 used for a KZ / Zurek-time estimate (seconds).",
						"This is an *analysis-only* parameter: it does not affect the solver; it only affects what the GUI reports.",
						"If omitted, the GUI will fall back to the simulation timestep dt as a ‘time unit’ (which is convenient for planning offsets, but not a true physical τ0).",
						"If you have a better τ0 estimate (from theory, fit, or calibration), set it here.",
					),
				),
				FieldSpec(
					"noise_amplitude",
					"Init noise amplitude",
					"1e-3",
					"float",
					help_text=_join_lines(
						"Only used when init_mode=2 (isotropic+noise).",
						"Sets the amplitude of random perturbations around isotropic Q≈0.",
						"Larger noise seeds more/larger domains early; too large can create strong gradients and require smaller dt.",
						"KZ intent: noise amplitude should be ‘small’ compared to the eventual nematic order so that domains emerge from amplification near the transition rather than being imposed by the initial condition.",
						"Practical: if noise is too small, the system can remain nearly isotropic for a long time and the quench looks artificially ‘clean’; if too large, you can create grid-scale roughness that triggers checkerboard modes.",
					),
				),
			],
			"Safety/Debug": [
				FieldSpec(
					"xi_guard_enabled",
					"xi guard enabled",
					"(default: true)",
					"tri_bool",
					help_text=_join_lines(
						"Correlation-length resolution guard.",
						"The backend estimates xi and aborts if xi is too small compared to dx (under-resolved core/interface).",
						"Disable only for quick smoke tests; for production/KZ runs, keep enabled to avoid meaningless results.",
						"Why it matters: if the nematic correlation length (or defect core size) is smaller than a couple of grid spacings, defects become ‘numerical’ and their density is not physically interpretable.",
						"If the guard triggers, you can (a) increase dx (coarser physical resolution), (b) move closer to Tc (larger xi), or (c) change elastic/bulk parameters that set xi.",
					),
				),
				FieldSpec(
					"enable_instability_guard",
					"Instability guard",
					"(default: true)",
					"tri_bool",
					help_text=_join_lines(
						"Runtime guard that monitors for numerical blow-up / aliasing.",
						"It checks max_S and a checkerboard (Nyquist) metric and can abort or reduce dt.",
						"Use this when exploring small dx or strong anisotropic elasticity (L2/L3) where explicit stiffness can excite odd-even modes.",
					),
				),
				FieldSpec(
					"enable_adaptive_dt",
					"Adaptive dt",
					"(default: true)",
					"tri_bool",
					help_text=_join_lines(
						"Only relevant if enable_instability_guard=true.",
						"If enabled, the guard reduces dt when it detects instability (no rollback).",
						"If disabled, the guard aborts to avoid writing a corrupted final state.",
						"Use adaptive dt when you are exploring parameter space and prefer ‘finish safely’ over strict reproducibility.",
						"For production measurements, fixed dt is usually cleaner: if dt changes mid-run, you should interpret iteration counts in physical time carefully.",
					),
				),
				FieldSpec(
					"S_abort",
					"Guard: S_abort",
					"2.0",
					"float",
					help_text=_join_lines(
						"Guard threshold: abort/reduce dt when the maximum scalar order parameter S exceeds this value.",
						"Typical physical S for 5CB is <~1; values far above 1 usually indicate numerical runaway.",
						"If you routinely hit S_abort early, the root cause is usually dt too large for the current stiffness (dx, elastic constants, bulk stiffness) or an under-resolved core.",
						"If you are using a different material model where S can legitimately approach >1, raise this threshold accordingly (but keep an eye on stability).",
					),
				),
				FieldSpec(
					"checker_rel_abort",
					"Guard: checker_rel_abort",
					"0.10",
					"float",
					help_text=_join_lines(
						"Guard threshold for the odd-even / Nyquist checkerboard metric (relative amplitude).",
						"If this grows, you often see grid-aligned dot artifacts in S or director.",
						"If you get false positives early in weak ordering, increase this threshold or disable adaptive dt.",
					),
				),
				FieldSpec(
					"enable_q_limiter",
					"Clamp |Q| (limiter)",
					"(default: false)",
					"tri_bool",
					help_text=_join_lines(
						"Numerical limiter that clamps |Q| to prevent runaway (caps S approximately).",
						"This is NOT a 'pure' integrator feature; it changes the dynamics and is mainly for debugging.",
						"Prefer semi-implicit + stable dt; use this only to keep a run from blowing up when diagnosing issues.",
					),
				),
				FieldSpec(
					"S_cap",
					"Limiter: S_cap",
					"1.2",
					"float",
					help_text=_join_lines(
						"If enable_q_limiter=true, this is the approximate maximum allowed S.",
						"Set near the physical maximum you expect; too small will artificially melt the droplet core.",
						"Limiter warning: a hard cap can suppress real physics (e.g., core sharpening, biaxiality) by forcing S to saturate artificially.",
						"If you need a limiter to run stably, it’s a strong signal to revisit dt, semi-implicit settings, dx resolution, and L2/L3 anisotropy.",
					),
				),
				FieldSpec(
					"enable_early_stop",
					"Early stop",
					"(default: false)",
					"tri_bool",
					help_text=_join_lines(
						"Stops the simulation early once convergence is detected at the final temperature.",
						"Criteria combine relative energy change (tolerance), relative radiality change (RbEps), and optionally an absolute radiality threshold.",
						"Useful to save time in long quenches once the texture is essentially settled.",
					),
				),
				FieldSpec(
					"radiality_threshold",
					"Early stop: radiality threshold",
					"0.998",
					"float",
					help_text=_join_lines(
						"Only used when enable_early_stop=true.",
						"If >0, requires the radiality metric to exceed this value before early stop is allowed.",
						"Set to 0 to disable the absolute threshold and rely only on relative convergence.",
						"This is best used when you know the true equilibrium is the radial state and you want to stop once you are ‘close enough’.",
						"If you are scanning parameters where the equilibrium might not be radial, keeping a high radiality_threshold can prevent early stop from ever triggering (which is usually safer).",
					),
				),
				FieldSpec(
					"console_output",
					"Console output",
					"(default: true)",
					"tri_bool",
					help_text=_join_lines(
						"Controls whether the backend prints diagnostics to stdout during the run.",
						"Disable for cleaner logs or when running many batch jobs.",
						"Note: quench_log.dat is still written regardless.",
						"If you run the GUI, disabling console_output mainly reduces what appears in the GUI log pane; it does not speed up GPU compute much unless printing was a bottleneck.",
					),
				),
				FieldSpec(
					"debug_cuda_checks",
					"Debug CUDA checks",
					"(default: false)",
					"tri_bool",
					help_text=_join_lines(
						"Enables CUDA error checks at log points (forces GPU sync).",
						"Useful for debugging kernel failures, but slows the run.",
						"Enable this if you suspect illegal memory access, mis-sized launches, or NaNs propagating into CUDA kernels.",
						"Disable for production runs: synchronizing the GPU frequently can severely reduce throughput.",
					),
				),
				FieldSpec(
					"debug_dynamics",
					"Debug dynamics",
					"(default: false)",
					"tri_bool",
					help_text=_join_lines(
						"Prints extra diagnostics like max|mu| and max|ΔQ| at log points (requires copies/reductions).",
						"Useful when diagnosing instability or slow convergence; slows the run.",
						"If you are chasing ‘grid dot’ artifacts, debug_dynamics plus the checkerboard metric can help you see whether high-k modes are growing step-to-step.",
					),
				),
				FieldSpec(
					"log_defects_2d",
					"Log defects 2D",
					"(default: true)",
					"tri_bool",
					help_text=_join_lines(
						"Adds a cheap 2D winding-based defect proxy to quench_log.dat (computed on a single z-slice).",
						"Good for trend tracking (KZ scaling), but can miss defects if the core melts (low-S mask) or if defects are not captured in that slice.",
						"For robust topology, postprocess 3D fields or compute ring-winding charge on a shell.",
					),
				),
				FieldSpec(
					"defects_z_slice",
					"Defects z_slice",
					"Nz/2",
					"int",
					help_text=_join_lines(
						"Which z-plane (0..Nz-1) to use for the 2D defect proxy when log_defects_2d=true.",
						"Default is the mid-plane (Nz/2).",
						"If your defect structures are not symmetric about the mid-plane (e.g., loops near the surface), consider running multiple quenches with different defects_z_slice or rely on 3D snapshots instead.",
						"If you set Nz small, a single slice may be dominated by boundary effects; increasing Nz makes slice proxies more meaningful.",
					),
				),
				FieldSpec(
					"defects_S_threshold",
					"Defects S_threshold",
					"0.1",
					"float",
					help_text=_join_lines(
						"Mask threshold for the 2D defect proxy: points with S below this are ignored.",
						"Raising it makes the proxy focus on well-ordered regions; lowering it includes more of the core but can add noise.",
						"If you are counting ±1/2 disclinations in a 2D slice, masking out low-S regions is essential: the director is ill-defined where S≈0.",
						"Too aggressive a mask can ‘erase’ melted cores and underestimate charge; too permissive a mask can create spurious winding from noisy directors.",
					),
				),
				FieldSpec(
					"defects_charge_cutoff",
					"Defects charge_cutoff",
					"0.25",
					"float",
					help_text=_join_lines(
						"Threshold on local winding charge magnitude |s| to count a plaquette as a defect.",
						"Typical values are around 0.25 for a ±1/2 defect proxy on a discrete grid.",
						"This is a robustness knob: higher cutoff reduces false positives from smooth distortions; lower cutoff increases sensitivity but can overcount noisy regions.",
						"When comparing defect densities across runs, keep this fixed (changing it changes your measurement definition).",
					),
				),
				FieldSpec(
					"log_xi_grad_2d",
					"Log xi grad 2D",
					"(default: true)",
					"tri_bool",
					help_text=_join_lines(
						"Logs a cheap gradient-based correlation-length proxy in quench_log.dat (computed on a z-slice).",
						"Useful as a fast trend indicator without saving snapshots.",
						"For accurate xi, use snapshot-based correlation analysis in QSRvis.",
						"Interpretation: this proxy is built from local gradients (roughly ‘how quickly the order/director changes in space’), so it behaves like an inverse length scale.",
						"Near Tc, xi grows and gradients tend to weaken on average; deeper in the nematic, xi shrinks and gradients steepen near defect cores.",
						"Limitations: it is slice-based (not full 3D), it is sensitive to masking (xi_S_threshold), and it can be biased by boundary layers near the shell.",
						"Use this for lightweight scans and sanity checks; use snapshots when you need publishable xi estimates for log–log fits.",
					),
				),
				FieldSpec(
					"xi_z_slice",
					"xi z_slice",
					"Nz/2",
					"int",
					help_text=_join_lines(
						"Which z-plane to use for xi gradient proxy when log_xi_grad_2d=true.",
						"Often you want the same slice as defects_z_slice.",
						"If you intend to plot defect density vs correlation length from log proxies, using the same slice for both avoids slice-to-slice variability dominating your log–log slope.",
					),
				),
				FieldSpec(
					"xi_S_threshold",
					"xi S_threshold",
					"0.1",
					"float",
					help_text=_join_lines(
						"Mask threshold for xi gradient proxy: only use points with S above this value.",
						"Set consistent with defects_S_threshold when comparing proxies.",
						"If the mask is too strict near Tc (where S is small everywhere), the proxy becomes noisy because too few points contribute.",
						"If the mask is too permissive, you include isotropic regions where gradients are dominated by noise and the inferred xi becomes unreliable.",
					),
				),
			],
		}

		for tab_name, specs in self._specs_by_tab.items():
			frame = ttk.Frame(self.notebook)
			self.notebook.add(frame, text=tab_name)
			self._build_form(frame, specs)

		# Plot tab (in-app QSRvis integration)
		self._build_plot_tab()

		# Log view
		ttk.Label(log_container, text="Backend output:").pack(side=tk.TOP, anchor=tk.W)
		self.log_text = tk.Text(log_container, wrap=tk.NONE, height=12)
		self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		yscroll = ttk.Scrollbar(log_container, orient=tk.VERTICAL, command=self.log_text.yview)
		yscroll.pack(side=tk.RIGHT, fill=tk.Y)
		self.log_text.configure(yscrollcommand=yscroll.set)

	def _build_form(self, parent: ttk.Frame, specs: list[FieldSpec]) -> None:
		canvas = tk.Canvas(parent)
		scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
		inner = ttk.Frame(canvas)

		inner.bind(
			"<Configure>",
			lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
		)
		canvas.create_window((0, 0), window=inner, anchor="nw")
		canvas.configure(yscrollcommand=scroll.set)

		canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		scroll.pack(side=tk.RIGHT, fill=tk.Y)

		for r, spec in enumerate(specs):
			row = ttk.Frame(inner)
			row.grid(row=r, column=0, sticky=tk.EW, padx=(8, 8), pady=2)
			row.columnconfigure(1, weight=1)

			ttk.Label(row, text=spec.label).grid(row=0, column=0, sticky=tk.W, padx=(0, 8), pady=2)

			if spec.kind == "tri_bool":
				var = tk.StringVar(value="default")
				self._vars[spec.key] = var
				w = ttk.OptionMenu(row, var, var.get(), "default", "true", "false")
				w.grid(row=0, column=1, sticky=tk.EW, pady=2)
			elif spec.kind == "choice":
				var = tk.StringVar(value="default")
				self._vars[spec.key] = var
				choices = spec.choices or ["default"]
				w = ttk.OptionMenu(row, var, var.get(), *choices)
				w.grid(row=0, column=1, sticky=tk.EW, pady=2)
			else:
				var = tk.StringVar(value="")
				self._vars[spec.key] = var
				w = ttk.Entry(row, textvariable=var)
				w.grid(row=0, column=1, sticky=tk.EW, pady=2)

			hint = spec.default_hint
			ttk.Label(row, text=hint, foreground="#666").grid(row=0, column=2, sticky=tk.W, padx=(10, 8))

			help_text = _render_help(spec)
			if help_text:
				self._info_visible.setdefault(spec.key, False)
				btn_text = tk.StringVar(value="info")
				info = ttk.Label(row, text=help_text, justify=tk.LEFT, foreground="#444", wraplength=900)

				def _toggle_info(key: str = spec.key, label: ttk.Label = info, btv: tk.StringVar = btn_text) -> None:
					vis = self._info_visible.get(key, False)
					if vis:
						label.grid_remove()
						btv.set("info")
						self._info_visible[key] = False
					else:
						label.grid(row=1, column=1, columnspan=3, sticky=tk.EW, pady=(2, 6))
						btv.set("hide")
						self._info_visible[key] = True

				btn = ttk.Button(row, textvariable=btn_text, command=_toggle_info, width=6)
				btn.grid(row=0, column=3, sticky=tk.E, padx=(6, 0))
				# Hidden by default
				info.grid(row=1, column=1, columnspan=3, sticky=tk.EW, pady=(2, 6))
				info.grid_remove()
			else:
				# Keep column 3 aligned even when no help exists
				ttk.Label(row, text="").grid(row=0, column=3, sticky=tk.E)

		inner.columnconfigure(0, weight=1)

	# ---------------- Actions ----------------

	def _browse_backend(self) -> None:
		p = filedialog.askopenfilename(title="Select QSR_cuda executable")
		if p:
			self.backend_path.set(p)

	# ---------------- Plot tab ----------------

	def _build_plot_tab(self) -> None:
		plot_tab = ttk.Frame(self.notebook)
		self.notebook.add(plot_tab, text="Plot")

		controls = ttk.Frame(plot_tab)
		controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 6))
		controls.columnconfigure(1, weight=1)

		self._add_collapsible_info_section(
			plot_tab,
			key="plot_tab_about",
			title="About plotting",
			text=_join_lines(
				"This tab embeds QSRvis plotting directly in the GUI (no external subprocess).",
				"Source can be either:",
				"  - a run directory (e.g. output_quench_10k/) OR", 
				"  - a specific file (e.g. nematic_field_final.dat).",
				"If Source is blank, the GUI tries to use the last run output directory; otherwise it falls back to the repo root.",
				"Output dir is where plots/GIFs are saved (relative paths are resolved from the repo root).",
				"GIF modes (1/5/7) save a GIF and show a first-frame preview in the GUI.",
				"3D modes (11/12) require SciPy + scikit-image; if they are missing you’ll get an import error.",
			),
			wraplength=980,
		)

		ttk.Label(controls, text="Source (run dir or file):").grid(row=0, column=0, sticky=tk.W)
		ttk.Entry(controls, textvariable=self.plot_source).grid(row=0, column=1, sticky=tk.EW, padx=(8, 8))
		ttk.Button(controls, text="Browse…", command=self._browse_plot_source).grid(row=0, column=2, sticky=tk.E)

		ttk.Label(controls, text="Mode:").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
		self._plot_mode_labels = {
			"0": "0 Final state slice",
			"1": "1 Animation (GIF) from snapshots",
			"2": "2 Free energy vs iteration",
			"3": "3 Energy components vs iteration",
			"4": "4 Sweep summary S(T), F(T)",
			"5": "5 Animate temperature sweep (GIF)",
			"6": "6 Quench log plots",
			"7": "7 Quench animation (GIF)",
			"8": "8 KZ metrics from quench",
			"9": "9 Aggregate KZ scaling (2D proxies)",
			"10": "10 Sweep KZ slope stability",
			"11": "11 3D snapshot metrics preview",
			"12": "12 Aggregate KZ scaling (3D metrics)",
		}
		self._plot_mode_title = tk.StringVar(value=self._plot_mode_labels.get(self.plot_mode.get(), ""))
		ttk.OptionMenu(
			controls,
			self.plot_mode,
			self.plot_mode.get(),
			*sorted(self._plot_mode_labels.keys(), key=lambda x: int(x)),
			command=lambda _v: self._on_plot_mode_change(),
		).grid(row=1, column=1, sticky=tk.W, padx=(8, 8), pady=(6, 0))
		ttk.Label(controls, textvariable=self._plot_mode_title, foreground="#555").grid(row=1, column=2, sticky=tk.W, pady=(6, 0))

		ttk.Label(controls, text="Output dir:").grid(row=2, column=0, sticky=tk.W, pady=(6, 0))
		ttk.Entry(controls, textvariable=self.plot_out_dir, width=20).grid(row=2, column=1, sticky=tk.W, padx=(8, 8), pady=(6, 0))

		btns = ttk.Frame(controls)
		btns.grid(row=2, column=2, sticky=tk.E, pady=(6, 0))
		ttk.Button(btns, text="Render", command=self._on_render_plot).pack(side=tk.LEFT)
		ttk.Button(btns, text="Clear", command=self._clear_plot).pack(side=tk.LEFT, padx=(8, 0))

		status_row = ttk.Frame(plot_tab)
		status_row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 6))
		ttk.Label(status_row, textvariable=self._plot_status, foreground="#444").pack(side=tk.LEFT)

		opts = ttk.LabelFrame(plot_tab, text="Mode options")
		opts.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 8))
		container = ttk.Frame(opts)
		container.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)
		self._build_plot_mode_frames(container)
		self._on_plot_mode_change()

		canvas_box = ttk.LabelFrame(plot_tab, text="Preview")
		canvas_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
		self._plot_canvas_container = ttk.Frame(canvas_box)
		self._plot_canvas_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

	def _add_collapsible_info_section(
		self,
		parent: ttk.Frame,
		*,
		key: str,
		title: str,
		text: str,
		wraplength: int = 900,
	) -> None:
		"""Create a pack-based collapsible info section (info/hide button + wrapped text)."""
		if not text.strip():
			return
		self._info_visible.setdefault(key, False)
		header = ttk.Frame(parent)
		header.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 6))
		ttk.Label(header, text=title, foreground="#555").pack(side=tk.LEFT)
		btn_text = tk.StringVar(value="info")
		info = ttk.Label(parent, text=text, justify=tk.LEFT, foreground="#444", wraplength=wraplength)

		def _toggle() -> None:
			vis = self._info_visible.get(key, False)
			if vis:
				info.pack_forget()
				btn_text.set("info")
				self._info_visible[key] = False
			else:
				info.pack(side=tk.TOP, fill=tk.X, padx=18, pady=(0, 8))
				btn_text.set("hide")
				self._info_visible[key] = True

		ttk.Button(header, textvariable=btn_text, command=_toggle, width=6).pack(side=tk.RIGHT)
		# Hidden by default
		info.pack(side=tk.TOP, fill=tk.X, padx=18, pady=(0, 8))
		info.pack_forget()

	def _build_plot_mode_frames(self, parent: ttk.Frame) -> None:
		def add_row(frm: ttk.Frame, r: int, label: str, var: tk.Variable, *, width: int = 14) -> None:
			ttk.Label(frm, text=label).grid(row=r, column=0, sticky=tk.W, pady=2)
			ttk.Entry(frm, textvariable=var, width=width).grid(row=r, column=1, sticky=tk.W, padx=(8, 16), pady=2)

		def add_choice(frm: ttk.Frame, r: int, label: str, var: tk.StringVar, choices: list[str]) -> None:
			ttk.Label(frm, text=label).grid(row=r, column=0, sticky=tk.W, pady=2)
			ttk.OptionMenu(frm, var, var.get(), *choices).grid(row=r, column=1, sticky=tk.W, padx=(8, 16), pady=2)

		# Mode 0
		m0 = ttk.Frame(parent)
		self._plot_mode_frames["0"] = m0
		self._add_collapsible_info_section(
			m0,
			key="plot_m0_about",
			title="Mode 0: final state slice",
			text=_join_lines(
				"Renders a 2D slice from nematic_field_final.dat (or the last nematic_field_iter_*.dat if final is missing).",
				"The grid (Nx,Ny,Nz) is inferred from the file header; you don’t have to type it.",
				"Slice index: leave blank for the mid-plane.",
				"Color field:",
				"  - S: scalar order parameter (good for cores / droplet boundary)",
				"  - nz: director z-component (shows out-of-plane tilt)",
				"  - n_perp: |n_xy| magnitude.",
				"Arrows per axis: number of director arrows along each axis (set 0 to disable arrows).",
				"Zoom radius: crops around the droplet center in lattice units (useful for large boxes).",
			),
		)
		m0_body = ttk.Frame(m0)
		m0_body.pack(side=tk.TOP, fill=tk.X)
		self._plot_vars["m0_axis"] = tk.StringVar(value="z")
		self._plot_vars["m0_slice"] = tk.StringVar(value="")
		self._plot_vars["m0_zoom"] = tk.StringVar(value="")
		self._plot_vars["m0_color"] = tk.StringVar(value="S")
		self._plot_vars["m0_interp"] = tk.StringVar(value="nearest")
		self._plot_vars["m0_arrows"] = tk.StringVar(value="20")
		self._plot_vars["m0_vmin"] = tk.StringVar(value="")
		self._plot_vars["m0_vmax"] = tk.StringVar(value="")
		add_choice(m0_body, 0, "Slice axis", self._plot_vars["m0_axis"], ["x", "y", "z"])
		add_row(m0_body, 1, "Slice index (blank=mid)", self._plot_vars["m0_slice"], width=10)
		add_row(m0_body, 2, "Zoom radius (blank=off)", self._plot_vars["m0_zoom"], width=10)
		add_choice(m0_body, 3, "Color field", self._plot_vars["m0_color"], ["S", "nz", "n_perp"])
		add_choice(m0_body, 4, "Interpolation", self._plot_vars["m0_interp"], ["nearest", "bilinear", "bicubic", "spline16", "spline36", "sinc"])
		add_row(m0_body, 5, "Arrows per axis (0=off)", self._plot_vars["m0_arrows"], width=10)
		add_row(m0_body, 6, "vmin (blank=auto)", self._plot_vars["m0_vmin"], width=10)
		add_row(m0_body, 7, "vmax (blank=auto)", self._plot_vars["m0_vmax"], width=10)

		# Mode 1
		m1 = ttk.Frame(parent)
		self._plot_mode_frames["1"] = m1
		self._add_collapsible_info_section(
			m1,
			key="plot_m1_about",
			title="Mode 1: animation from snapshots",
			text=_join_lines(
				"Creates a GIF from nematic_field_iter_*.dat snapshots in a directory.",
				"This requires snapshot files (snapshot_mode=1 or 2 in the backend).",
				"duration is the per-frame time in seconds (smaller = faster playback).",
				"frame_stride skips frames: stride=2 uses every 2nd snapshot, etc.",
				"Color field and arrows match mode 0 meanings. The GUI previews only the first frame; the GIF is saved to Output dir.",
			),
		)
		m1_body = ttk.Frame(m1)
		m1_body.pack(side=tk.TOP, fill=tk.X)
		self._plot_vars["m1_axis"] = tk.StringVar(value="z")
		self._plot_vars["m1_slice"] = tk.StringVar(value="")
		self._plot_vars["m1_duration"] = tk.StringVar(value="0.1")
		self._plot_vars["m1_stride"] = tk.StringVar(value="1")
		self._plot_vars["m1_color"] = tk.StringVar(value="S")
		self._plot_vars["m1_interp"] = tk.StringVar(value="nearest")
		self._plot_vars["m1_arrows"] = tk.StringVar(value="20")
		self._plot_vars["m1_zoom"] = tk.StringVar(value="")
		add_choice(m1_body, 0, "Slice axis", self._plot_vars["m1_axis"], ["x", "y", "z"])
		add_row(m1_body, 1, "Slice index (blank=mid)", self._plot_vars["m1_slice"], width=10)
		add_row(m1_body, 2, "duration (s / frame)", self._plot_vars["m1_duration"], width=10)
		add_row(m1_body, 3, "frame_stride", self._plot_vars["m1_stride"], width=10)
		add_choice(m1_body, 4, "Color field", self._plot_vars["m1_color"], ["S", "nz", "n_perp"])
		add_choice(m1_body, 5, "Interpolation", self._plot_vars["m1_interp"], ["nearest", "bilinear", "bicubic", "spline16", "spline36", "sinc"])
		add_row(m1_body, 6, "Arrows per axis (0=off)", self._plot_vars["m1_arrows"], width=10)
		add_row(m1_body, 7, "Zoom radius (blank=off)", self._plot_vars["m1_zoom"], width=10)

		# Mode 2
		m2 = ttk.Frame(parent)
		self._plot_mode_frames["2"] = m2
		self._add_collapsible_info_section(
			m2,
			key="plot_m2_about",
			title="Mode 2: free energy vs iteration",
			text=_join_lines(
				"Plots the total/free energy trace versus iteration for a single run.",
				"Expected input: a run directory containing the backend’s energy log file(s) (e.g. output_quench*/).",
				"Output: a PNG saved into Output dir, and the figure is embedded in the Preview pane.",
				"If you see a flat line or missing data, verify the run actually wrote the energy-vs-iteration file and that you selected the correct Source folder.",
			),
		)

		# Mode 3
		m3 = ttk.Frame(parent)
		self._plot_mode_frames["3"] = m3
		self._add_collapsible_info_section(
			m3,
			key="plot_m3_about",
			title="Mode 3: energy components vs iteration",
			text=_join_lines(
				"Plots energy components (bulk / elastic / surface terms depending on what the backend logged) versus iteration.",
				"Expected input: a run directory containing the energy components log (produced by the backend).",
				"This is useful for diagnosing numerical issues:",
				"  - runaway bulk term often indicates dt too large or instability in S",
				"  - growing elastic term often indicates under-resolved gradients or too aggressive ramp/step.",
				"Output: a PNG saved into Output dir, and the figure is embedded in the Preview pane.",
			),
		)

		# Mode 4
		m4 = ttk.Frame(parent)
		self._plot_mode_frames["4"] = m4
		self._add_collapsible_info_section(
			m4,
			key="plot_m4_about",
			title="Mode 4: sweep summary",
			text=_join_lines(
				"Plots summary outputs from a temperature sweep (average S(T) and/or free energy F(T)).",
				"Use this with output_temp_sweep/ runs that contain the sweep summary files produced by the backend.",
				"If you only want one curve, set Which to S or F.",
			),
		)
		m4_body = ttk.Frame(m4)
		m4_body.pack(side=tk.TOP, fill=tk.X)
		self._plot_vars["m4_which"] = tk.StringVar(value="both")
		add_choice(m4_body, 0, "Which", self._plot_vars["m4_which"], ["both", "S", "F"])

		# Mode 5
		m5 = ttk.Frame(parent)
		self._plot_mode_frames["5"] = m5
		self._add_collapsible_info_section(
			m5,
			key="plot_m5_about",
			title="Mode 5: animate temperature sweep",
			text=_join_lines(
				"Creates a GIF (or saves selected frames) across output_temp_sweep/T_*/ directories.",
				"Source can be either the repo root (it will look for output_temp_sweep/) or the output_temp_sweep folder itself.",
				"Color field controls what is visualized in each frame (S / nz / n_perp).",
				"When output=GIF, duration is the per-frame time in seconds.",
				"When output=frames, use indices like: 0,2,5-7 or 'all'.",
			),
		)
		m5_body = ttk.Frame(m5)
		m5_body.pack(side=tk.TOP, fill=tk.X)
		self._plot_vars["m5_output"] = tk.StringVar(value="gif")
		self._plot_vars["m5_color"] = tk.StringVar(value="S")
		self._plot_vars["m5_duration"] = tk.StringVar(value="0.5")
		self._plot_vars["m5_selected"] = tk.StringVar(value="all")
		add_choice(m5_body, 0, "Output", self._plot_vars["m5_output"], ["gif", "frames"])
		add_choice(m5_body, 1, "Color field", self._plot_vars["m5_color"], ["S", "nz", "n_perp"])
		add_row(m5_body, 2, "duration (s / frame)", self._plot_vars["m5_duration"], width=10)
		add_row(m5_body, 3, "selected indices", self._plot_vars["m5_selected"], width=18)

		# Mode 6
		m6 = ttk.Frame(parent)
		self._plot_mode_frames["6"] = m6
		self._add_collapsible_info_section(
			m6,
			key="plot_m6_about",
			title="Mode 6: quench log",
			text=_join_lines(
				"Plots diagnostics from out_dir/quench_log.dat (T(t), energies, radiality, <S>, etc.).",
				"it_min/it_max restrict the plotted iteration range (useful if you have long runs).",
				"Note: only the summary plot is embedded; other subplots are saved by QSRvis but not yet embedded.",
			),
		)
		m6_body = ttk.Frame(m6)
		m6_body.pack(side=tk.TOP, fill=tk.X)
		self._plot_vars["m6_kind"] = tk.StringVar(value="s")
		self._plot_vars["m6_it_min"] = tk.StringVar(value="")
		self._plot_vars["m6_it_max"] = tk.StringVar(value="")
		add_choice(m6_body, 0, "Quench plot", self._plot_vars["m6_kind"], ["s", "2", "4", "d", "a"])
		add_row(m6_body, 1, "it_min (blank=all)", self._plot_vars["m6_it_min"], width=12)
		add_row(m6_body, 2, "it_max (blank=all)", self._plot_vars["m6_it_max"], width=12)

		# Mode 7
		m7 = ttk.Frame(parent)
		self._plot_mode_frames["7"] = m7
		self._add_collapsible_info_section(
			m7,
			key="plot_m7_about",
			title="Mode 7: quench animation (GIF)",
			text=_join_lines(
				"Same as mode 1, but intended for quench runs (output_quench*/ with nematic_field_iter_*.dat).",
				"If you ran snapshot_mode=2 (KZ), you’ll only have frames near Tc; the GIF will cover that window.",
			),
		)
		m7_body = ttk.Frame(m7)
		m7_body.pack(side=tk.TOP, fill=tk.X)
		self._plot_vars["m7_axis"] = tk.StringVar(value="z")
		self._plot_vars["m7_slice"] = tk.StringVar(value="")
		self._plot_vars["m7_duration"] = tk.StringVar(value="0.1")
		self._plot_vars["m7_stride"] = tk.StringVar(value="1")
		self._plot_vars["m7_color"] = tk.StringVar(value="S")
		self._plot_vars["m7_interp"] = tk.StringVar(value="nearest")
		self._plot_vars["m7_arrows"] = tk.StringVar(value="20")
		self._plot_vars["m7_zoom"] = tk.StringVar(value="")
		add_choice(m7_body, 0, "Slice axis", self._plot_vars["m7_axis"], ["x", "y", "z"])
		add_row(m7_body, 1, "Slice index (blank=mid)", self._plot_vars["m7_slice"], width=10)
		add_row(m7_body, 2, "duration (s / frame)", self._plot_vars["m7_duration"], width=10)
		add_row(m7_body, 3, "frame_stride", self._plot_vars["m7_stride"], width=10)
		add_choice(m7_body, 4, "Color field", self._plot_vars["m7_color"], ["S", "nz", "n_perp"])
		add_choice(m7_body, 5, "Interpolation", self._plot_vars["m7_interp"], ["nearest", "bilinear", "bicubic", "spline16", "spline36", "sinc"])
		add_row(m7_body, 6, "Arrows per axis (0=off)", self._plot_vars["m7_arrows"], width=10)
		add_row(m7_body, 7, "Zoom radius (blank=off)", self._plot_vars["m7_zoom"], width=10)

		# Mode 8
		m8 = ttk.Frame(parent)
		self._plot_mode_frames["8"] = m8
		self._add_collapsible_info_section(
			m8,
			key="plot_m8_about",
			title="Mode 8: KZ metrics (single run)",
			text=_join_lines(
				"Computes proxy correlation length xi and proxy defect density from snapshots in a single quench run.",
				"Requires saved frames (nematic_field_iter_*.dat) around Tc (snapshot_mode=1 or 2).",
				"frame_stride skips frames to trade accuracy vs speed.",
				"S_threshold masks out isotropic regions; too high near Tc can produce noisy metrics.",
			),
		)
		m8_body = ttk.Frame(m8)
		m8_body.pack(side=tk.TOP, fill=tk.X)
		self._plot_vars["m8_z"] = tk.StringVar(value="")
		self._plot_vars["m8_stride"] = tk.StringVar(value="10")
		self._plot_vars["m8_max"] = tk.StringVar(value="50")
		self._plot_vars["m8_sthr"] = tk.StringVar(value="0.1")
		add_row(m8_body, 0, "z_slice (blank=mid)", self._plot_vars["m8_z"], width=10)
		add_row(m8_body, 1, "frame_stride", self._plot_vars["m8_stride"], width=10)
		add_row(m8_body, 2, "max_frames (blank=none)", self._plot_vars["m8_max"], width=10)
		add_row(m8_body, 3, "S_threshold", self._plot_vars["m8_sthr"], width=10)

		# Mode 9
		m9 = ttk.Frame(parent)
		self._plot_mode_frames["9"] = m9
		self._add_collapsible_info_section(
			m9,
			key="plot_m9_about",
			title="Mode 9: aggregate KZ scaling (2D proxies)",
			text=_join_lines(
				"Scans many quench run directories and fits log–log KZ scaling (defects and xi proxies vs ramp time or cooling rate).",
				"parent_dir + pattern define which runs are included (e.g. parent_dir='.' and pattern='output_quench*').",
				"measure selects which snapshot/time to evaluate (final / after_Tlow / after_Tc).",
				"z_avg averages over multiple slices for robustness; z_margin_frac avoids boundary layers.",
				"If your logs include defect/xi proxies, allow_log_only=true lets you proceed even without snapshots.",
			),
		)
		m9_body = ttk.Frame(m9)
		m9_body.pack(side=tk.TOP, fill=tk.X)
		self._plot_vars["m9_parent"] = tk.StringVar(value=".")
		self._plot_vars["m9_pattern"] = tk.StringVar(value="output_quench*")
		self._plot_vars["m9_xaxis"] = tk.StringVar(value="t_ramp")
		self._plot_vars["m9_measure"] = tk.StringVar(value="final")
		self._plot_vars["m9_allow_log"] = tk.StringVar(value="true")
		self._plot_vars["m9_sthr"] = tk.StringVar(value="0.1")
		self._plot_vars["m9_z"] = tk.StringVar(value="")
		self._plot_vars["m9_zavg"] = tk.StringVar(value="1")
		self._plot_vars["m9_zmarg"] = tk.StringVar(value="0.0")
		self._plot_vars["m9_tc"] = tk.StringVar(value="310.2")
		self._plot_vars["m9_after_tc"] = tk.StringVar(value="0.0")
		self._plot_vars["m9_after_tlow"] = tk.StringVar(value="0.0")
		self._plot_vars["m9_fxmin"] = tk.StringVar(value="")
		self._plot_vars["m9_fxmax"] = tk.StringVar(value="")
		add_row(m9_body, 0, "parent_dir", self._plot_vars["m9_parent"], width=26)
		add_row(m9_body, 1, "pattern", self._plot_vars["m9_pattern"], width=26)
		add_choice(m9_body, 2, "x_axis", self._plot_vars["m9_xaxis"], ["t_ramp", "rate"])
		add_choice(m9_body, 3, "measure", self._plot_vars["m9_measure"], ["final", "after_Tlow", "after_Tc"])
		add_choice(m9_body, 4, "allow_log_only", self._plot_vars["m9_allow_log"], ["true", "false"])
		add_row(m9_body, 5, "S_threshold", self._plot_vars["m9_sthr"], width=10)
		add_row(m9_body, 6, "z_slice (blank=mid)", self._plot_vars["m9_z"], width=10)
		add_row(m9_body, 7, "z_avg", self._plot_vars["m9_zavg"], width=10)
		add_row(m9_body, 8, "z_margin_frac", self._plot_vars["m9_zmarg"], width=10)
		add_row(m9_body, 9, "Tc [K]", self._plot_vars["m9_tc"], width=10)
		add_row(m9_body, 10, "after_Tc_s", self._plot_vars["m9_after_tc"], width=10)
		add_row(m9_body, 11, "after_Tlow_s", self._plot_vars["m9_after_tlow"], width=10)
		add_row(m9_body, 12, "fit_x_min", self._plot_vars["m9_fxmin"], width=10)
		add_row(m9_body, 13, "fit_x_max", self._plot_vars["m9_fxmax"], width=10)

		# Mode 10
		m10 = ttk.Frame(parent)
		self._plot_mode_frames["10"] = m10
		self._add_collapsible_info_section(
			m10,
			key="plot_m10_about",
			title="Mode 10: slope stability",
			text=_join_lines(
				"Sweeps the ‘measurement time’ after Tc and checks how fitted log–log slopes change.",
				"This is a robustness diagnostic: in real data, slopes can drift if you measure too early/late relative to the freeze-out window.",
				"snapshotFreq_iters is used to convert offsets_in_snaps into physical seconds using dt inferred from the log.",
			),
		)
		m10_body = ttk.Frame(m10)
		m10_body.pack(side=tk.TOP, fill=tk.X)
		self._plot_vars["m10_parent"] = tk.StringVar(value=".")
		self._plot_vars["m10_pattern"] = tk.StringVar(value="output_quench*")
		self._plot_vars["m10_xaxis"] = tk.StringVar(value="t_ramp")
		self._plot_vars["m10_tc"] = tk.StringVar(value="310.2")
		self._plot_vars["m10_snapfreq"] = tk.StringVar(value="10000")
		self._plot_vars["m10_offsets"] = tk.StringVar(value="0,1,2,5,10")
		self._plot_vars["m10_sthr"] = tk.StringVar(value="0.02")
		self._plot_vars["m10_z"] = tk.StringVar(value="")
		self._plot_vars["m10_zavg"] = tk.StringVar(value="11")
		self._plot_vars["m10_zmarg"] = tk.StringVar(value="0.2")
		add_row(m10_body, 0, "parent_dir", self._plot_vars["m10_parent"], width=26)
		add_row(m10_body, 1, "pattern", self._plot_vars["m10_pattern"], width=26)
		add_choice(m10_body, 2, "x_axis", self._plot_vars["m10_xaxis"], ["t_ramp", "rate"])
		add_row(m10_body, 3, "Tc [K]", self._plot_vars["m10_tc"], width=10)
		add_row(m10_body, 4, "snapshotFreq_iters", self._plot_vars["m10_snapfreq"], width=10)
		add_row(m10_body, 5, "offsets_in_snaps", self._plot_vars["m10_offsets"], width=18)
		add_row(m10_body, 6, "S_threshold", self._plot_vars["m10_sthr"], width=10)
		add_row(m10_body, 7, "z_slice (blank=mid)", self._plot_vars["m10_z"], width=10)
		add_row(m10_body, 8, "z_avg", self._plot_vars["m10_zavg"], width=10)
		add_row(m10_body, 9, "z_margin_frac", self._plot_vars["m10_zmarg"], width=10)

		# Mode 11
		m11 = ttk.Frame(parent)
		self._plot_mode_frames["11"] = m11
		self._add_collapsible_info_section(
			m11,
			key="plot_m11_about",
			title="Mode 11: 3D snapshot metrics",
			text=_join_lines(
				"Computes 3D correlation length xi_3D and a 3D defect-line proxy from a single snapshot.",
				"This is heavier than 2D proxies and requires SciPy + scikit-image.",
				"S_droplet / S_core define a low-S ‘core’ region; defect proxy uses either a skeleton line-length density or a core-volume density.",
				"The GUI preview shows mid-plane slices of S, the core mask, and the skeleton mask.",
			),
		)
		m11_body = ttk.Frame(m11)
		m11_body.pack(side=tk.TOP, fill=tk.X)
		self._plot_vars["m11_sxi"] = tk.StringVar(value="0.1")
		self._plot_vars["m11_sdrop"] = tk.StringVar(value="0.1")
		self._plot_vars["m11_score"] = tk.StringVar(value="0.05")
		self._plot_vars["m11_dilate"] = tk.StringVar(value="2")
		self._plot_vars["m11_mincore"] = tk.StringVar(value="30")
		self._plot_vars["m11_skeleton"] = tk.StringVar(value="true")
		add_row(m11_body, 0, "S_threshold_xi", self._plot_vars["m11_sxi"], width=10)
		add_row(m11_body, 1, "S_droplet", self._plot_vars["m11_sdrop"], width=10)
		add_row(m11_body, 2, "S_core", self._plot_vars["m11_score"], width=10)
		add_row(m11_body, 3, "dilate_iters", self._plot_vars["m11_dilate"], width=10)
		add_row(m11_body, 4, "min_core_voxels", self._plot_vars["m11_mincore"], width=10)
		add_choice(m11_body, 5, "use_skeleton", self._plot_vars["m11_skeleton"], ["true", "false"])

		# Mode 12
		m12 = ttk.Frame(parent)
		self._plot_mode_frames["12"] = m12
		self._add_collapsible_info_section(
			m12,
			key="plot_m12_about",
			title="Mode 12: aggregate KZ scaling (3D metrics)",
			text=_join_lines(
				"Aggregates xi_3D and defect-line proxy across multiple runs and fits log–log scaling.",
				"This is the most expensive plot mode: it loads 3D volumes per run.",
				"defect_proxy:",
				"  - skeleton: line-length density (good proxy for line defects)",
				"  - core: core-volume density (faster, cruder).",
				"Tip: start with max_runs set to a small number when debugging parameters.",
			),
		)
		m12_body = ttk.Frame(m12)
		m12_body.pack(side=tk.TOP, fill=tk.X)
		self._plot_vars["m12_parent"] = tk.StringVar(value=".")
		self._plot_vars["m12_pattern"] = tk.StringVar(value="output_quench*")
		self._plot_vars["m12_xaxis"] = tk.StringVar(value="t_ramp")
		self._plot_vars["m12_measure"] = tk.StringVar(value="after_Tc")
		self._plot_vars["m12_tc"] = tk.StringVar(value="310.2")
		self._plot_vars["m12_after_tc"] = tk.StringVar(value="0.0")
		self._plot_vars["m12_sxi"] = tk.StringVar(value="0.1")
		self._plot_vars["m12_sdrop"] = tk.StringVar(value="0.1")
		self._plot_vars["m12_score"] = tk.StringVar(value="0.05")
		self._plot_vars["m12_dilate"] = tk.StringVar(value="2")
		self._plot_vars["m12_mincore"] = tk.StringVar(value="30")
		self._plot_vars["m12_proxy"] = tk.StringVar(value="skeleton")
		self._plot_vars["m12_maxruns"] = tk.StringVar(value="")
		add_row(m12_body, 0, "parent_dir", self._plot_vars["m12_parent"], width=26)
		add_row(m12_body, 1, "pattern", self._plot_vars["m12_pattern"], width=26)
		add_choice(m12_body, 2, "x_axis", self._plot_vars["m12_xaxis"], ["t_ramp", "rate"])
		add_choice(m12_body, 3, "measure", self._plot_vars["m12_measure"], ["final", "after_Tlow", "after_Tc"])
		add_row(m12_body, 4, "Tc [K]", self._plot_vars["m12_tc"], width=10)
		add_row(m12_body, 5, "after_Tc_s", self._plot_vars["m12_after_tc"], width=10)
		add_row(m12_body, 6, "S_threshold_xi", self._plot_vars["m12_sxi"], width=10)
		add_row(m12_body, 7, "S_droplet", self._plot_vars["m12_sdrop"], width=10)
		add_row(m12_body, 8, "S_core", self._plot_vars["m12_score"], width=10)
		add_row(m12_body, 9, "dilate_iters", self._plot_vars["m12_dilate"], width=10)
		add_row(m12_body, 10, "min_core_voxels", self._plot_vars["m12_mincore"], width=10)
		add_choice(m12_body, 11, "defect_proxy", self._plot_vars["m12_proxy"], ["skeleton", "core"])
		add_row(m12_body, 12, "max_runs (blank=all)", self._plot_vars["m12_maxruns"], width=10)

	def _on_plot_mode_change(self) -> None:
		m = self.plot_mode.get().strip()
		self._plot_mode_title.set(self._plot_mode_labels.get(m, ""))
		for frm in self._plot_mode_frames.values():
			frm.pack_forget()
		frm = self._plot_mode_frames.get(m)
		if frm is not None:
			frm.pack(side=tk.TOP, fill=tk.X)

	def _browse_plot_source(self) -> None:
		p = filedialog.askopenfilename(title="Select file (or cancel to pick folder)")
		if p:
			self.plot_source.set(p)
			return
		p2 = filedialog.askdirectory(title="Select folder")
		if p2:
			self.plot_source.set(p2)

	def _clear_plot(self) -> None:
		try:
			import matplotlib.pyplot as plt
		except Exception:
			plt = None
		if self._plot_fig is not None and plt is not None:
			try:
				plt.close(self._plot_fig)
			except Exception:
				pass
		self._plot_fig = None
		if self._plot_canvas is not None:
			try:
				self._plot_canvas.get_tk_widget().destroy()
			except Exception:
				pass
			self._plot_canvas = None
		if self._plot_toolbar is not None:
			try:
				self._plot_toolbar.destroy()
			except Exception:
				pass
			self._plot_toolbar = None
		self._plot_status.set("")

	def _display_figure(self, fig: Any) -> None:
		if FigureCanvasTkAgg is None:
			raise RuntimeError("Matplotlib Tk backend not available (FigureCanvasTkAgg import failed)")
		if self._plot_canvas_container is None:
			raise RuntimeError("Plot canvas container not initialized")
		self._clear_plot()
		self._plot_fig = fig
		self._plot_canvas = FigureCanvasTkAgg(fig, master=self._plot_canvas_container)
		self._plot_canvas.draw()
		self._plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
		if NavigationToolbar2Tk is not None:
			self._plot_toolbar = NavigationToolbar2Tk(self._plot_canvas, self._plot_canvas_container)
			self._plot_toolbar.update()
			self._plot_toolbar.pack(side=tk.TOP, fill=tk.X)

	def _abs_out_dir(self) -> str:
		out_dir = self.plot_out_dir.get().strip() or "pics"
		p = Path(out_dir).expanduser()
		if not p.is_absolute():
			p = _repo_root() / p
		return str(p)

	def _pick_source(self) -> str:
		s = self.plot_source.get().strip()
		if s:
			return s
		if self.last_out_dir is not None:
			return str(self.last_out_dir)
		return str(_repo_root())

	def _pick_field_file(self, src: str) -> str:
		p = Path(src).expanduser()
		if p.is_file():
			return str(p)
		if p.is_dir():
			cand = p / "nematic_field_final.dat"
			if cand.exists():
				return str(cand)
			snaps = sorted(p.glob("nematic_field_iter_*.dat"))
			if snaps:
				return str(snaps[-1])
		raise FileNotFoundError(f"Could not resolve a field file from: {src}")

	def _on_render_plot(self) -> None:
		self._plot_status.set("Rendering…")
		m = self.plot_mode.get().strip()
		src = self._pick_source()
		out_dir = self._abs_out_dir()
		try:
			import QSRvis as v
			import matplotlib.pyplot as plt
			import numpy as np
		except Exception as e:
			self._plot_status.set(f"Import failed: {e}")
			return

		try:
			if m == "0":
				field_fp = self._pick_field_file(src)
				Nx, Ny, Nz = v.infer_grid_dims_from_nematic_field_file(field_fp)
				axis = str(self._plot_vars["m0_axis"].get())
				sl = str(self._plot_vars["m0_slice"].get()).strip()
				slice_idx = int(float(sl)) if sl else None
				zoom_s = str(self._plot_vars["m0_zoom"].get()).strip()
				zoom = int(float(zoom_s)) if zoom_s else None
				arrows = int(float(str(self._plot_vars["m0_arrows"].get()).strip() or "20"))
				vmin_s = str(self._plot_vars["m0_vmin"].get()).strip()
				vmax_s = str(self._plot_vars["m0_vmax"].get()).strip()
				vmin = float(vmin_s) if vmin_s else None
				vmax = float(vmax_s) if vmax_s else None
				fig = v.plot_nematic_field_slice(
					filename=field_fp,
					Nx=Nx,
					Ny=Ny,
					Nz=Nz,
					z_slice=slice_idx,
					slice_axis=axis,
					output_path=None,
					zoom_radius=zoom,
					interpol=str(self._plot_vars["m0_interp"].get()),
					color_field=str(self._plot_vars["m0_color"].get()),
					arrows_per_axis=arrows,
					vmin=vmin,
					vmax=vmax,
					print_stats=False,
					show=False,
					close=False,
					return_fig=True,
				)
				self._display_figure(fig)
				self._plot_status.set(f"Mode 0 rendered from {Path(field_fp).name}")

			elif m in ("1", "7"):
				data_dir = src
				if not Path(data_dir).is_dir():
					raise ValueError("Mode 1/7 expects a directory containing nematic_field_iter_*.dat")
				import glob
				snaps = glob.glob(str(Path(data_dir) / "nematic_field_iter_*.dat"))
				if not snaps:
					raise FileNotFoundError("No nematic_field_iter_*.dat files found")
				Nx, Ny, Nz = v.infer_grid_dims_from_nematic_field_file(snaps[0])
				out_name = "nematic_field_evolution.gif" if m == "1" else "quench_evolution.gif"
				out_gif = str(Path(out_dir) / out_name)
				prefix = "m1" if m == "1" else "m7"
				axis = str(self._plot_vars[f"{prefix}_axis"].get()).strip() or "z"
				sl_s = str(self._plot_vars[f"{prefix}_slice"].get()).strip()
				slice_idx = int(float(sl_s)) if sl_s else None
				duration = float(str(self._plot_vars[f"{prefix}_duration"].get()).strip() or "0.1")
				stride = int(float(str(self._plot_vars[f"{prefix}_stride"].get()).strip() or "1"))
				color_field = str(self._plot_vars[f"{prefix}_color"].get()).strip() or "S"
				interpol = str(self._plot_vars[f"{prefix}_interp"].get()).strip() or "nearest"
				arrows = int(float(str(self._plot_vars[f"{prefix}_arrows"].get()).strip() or "20"))
				zoom_s = str(self._plot_vars[f"{prefix}_zoom"].get()).strip()
				zoom = int(float(zoom_s)) if zoom_s else None
				frames_dir = str(Path(out_dir) / ("frames_m1" if m == "1" else "frames_m7"))
				v.create_nematic_field_animation(
					data_dir=data_dir,
					output_gif=out_gif,
					Nx=Nx,
					Ny=Ny,
					Nz=Nz,
					slice_axis=axis,
					slice_index=slice_idx,
					frames_dir=frames_dir,
					duration=duration,
					frame_stride=stride,
					color_field=color_field,
					interpol=interpol,
					zoom_radius=zoom,
					arrows_per_axis=arrows,
				)
				try:
					import imageio.v2 as imageio
				except Exception:
					import imageio
					imageio = imageio  # type: ignore
				frame0 = imageio.imread(out_gif)
				fig, ax = plt.subplots(1, 1, figsize=(8, 6))
				ax.imshow(frame0)
				ax.set_axis_off()
				ax.set_title(Path(out_gif).name)
				fig.tight_layout()
				self._display_figure(fig)
				self._plot_status.set(f"Saved GIF -> {out_gif} (stride={stride}, duration={duration:g}s)")

			elif m == "2":
				fig = v.plot_energy_VS_iter(src, out_dir=out_dir, show=False, close=False, return_fig=True)
				self._display_figure(fig)
				self._plot_status.set("Saved and rendered free energy vs iteration")

			elif m == "3":
				fig = v.energy_components(src, out_dir=out_dir, show=False, close=False, return_fig=True)
				self._display_figure(fig)
				self._plot_status.set("Saved and rendered energy components vs iteration")

			elif m == "4":
				which = str(self._plot_vars["m4_which"].get())
				figs = v.plotS_F(src, out_dir=out_dir, show=False, close=False, which=which, return_figs=True)
				if isinstance(figs, dict) and figs:
					fig = figs.get("S") or figs.get("F")
					if fig is None:
						raise RuntimeError("plotS_F returned no figures")
					self._display_figure(fig)
				self._plot_status.set("Saved and rendered sweep summary")

			elif m == "6":
				kind = str(self._plot_vars["m6_kind"].get()).strip().lower() or "s"
				it_min_s = str(self._plot_vars["m6_it_min"].get()).strip()
				it_max_s = str(self._plot_vars["m6_it_max"].get()).strip()
				it_min = int(float(it_min_s)) if it_min_s else None
				it_max = int(float(it_max_s)) if it_max_s else None
				if kind in ("s", "a"):
					fig, _axes = v.plot_quench_log(src, out_dir=out_dir, show=False, close=False, it_min=it_min, it_max=it_max)
					self._display_figure(fig)
					self._plot_status.set("Saved and rendered quench summary")
				else:
					self._plot_status.set("Mode 6 subplots (2/4/d) are saved by QSRvis; summary is embeddable.")

			elif m == "8":
				z_s = str(self._plot_vars["m8_z"].get()).strip()
				z = int(float(z_s)) if z_s else None
				stride = int(float(str(self._plot_vars["m8_stride"].get()).strip() or "10"))
				max_s = str(self._plot_vars["m8_max"].get()).strip()
				max_frames = int(float(max_s)) if max_s else None
				sthr = float(str(self._plot_vars["m8_sthr"].get()).strip() or "0.1")
				res = v.plot_quench_kz_metrics(
					src,
					out_dir=out_dir,
					z_slice=z,
					frame_stride=stride,
					max_frames=max_frames,
					S_threshold=sthr,
					show=False,
					close=False,
					return_figs=True,
				)
				rows = res[0]
				out_png = res[1]
				figs = res[3] if len(res) >= 4 else {}
				fig = figs.get("metrics") if isinstance(figs, dict) else None
				if fig is not None:
					self._display_figure(fig)
				self._plot_status.set(f"Saved KZ metrics -> {out_png} (rows={len(rows)})")

			elif m == "9":
				parent_dir = str(self._plot_vars["m9_parent"].get()).strip() or "."
				pattern = str(self._plot_vars["m9_pattern"].get()).strip() or "output_quench*"
				x_axis = str(self._plot_vars["m9_xaxis"].get()).strip() or "t_ramp"
				measure = str(self._plot_vars["m9_measure"].get()).strip() or "final"
				allow_log_only = str(self._plot_vars["m9_allow_log"].get()).strip().lower() == "true"
				sthr = float(str(self._plot_vars["m9_sthr"].get()).strip() or "0.1")
				z_s = str(self._plot_vars["m9_z"].get()).strip()
				z_slice = int(float(z_s)) if z_s else None
				z_avg = int(float(str(self._plot_vars["m9_zavg"].get()).strip() or "1"))
				z_marg = float(str(self._plot_vars["m9_zmarg"].get()).strip() or "0.0")
				Tc = float(str(self._plot_vars["m9_tc"].get()).strip() or "310.2")
				after_Tc_s = float(str(self._plot_vars["m9_after_tc"].get()).strip() or "0.0")
				after_Tlow_s = float(str(self._plot_vars["m9_after_tlow"].get()).strip() or "0.0")
				fx1 = str(self._plot_vars["m9_fxmin"].get()).strip()
				fx2 = str(self._plot_vars["m9_fxmax"].get()).strip()
				fit_x_min = float(fx1) if fx1 else None
				fit_x_max = float(fx2) if fx2 else None
				res = v.aggregate_kz_scaling(
					parent_dir,
					pattern=pattern,
					out_dir=out_dir,
					z_slice=z_slice,
					z_avg=z_avg,
					z_margin_frac=z_marg,
					S_threshold=sthr,
					prefer_log_defects=allow_log_only,
					prefer_log_xi_proxy=allow_log_only,
					allow_log_only=allow_log_only,
					x_axis=x_axis,
					fit_x_min=fit_x_min,
					fit_x_max=fit_x_max,
					measure=measure,
					after_Tlow_s=after_Tlow_s,
					Tc=Tc,
					after_Tc_s=after_Tc_s,
					show=False,
					close=False,
					return_figs=True,
				)
				rows = res[0]
				out_png = res[1]
				figs = res[3] if len(res) >= 4 else {}
				fig = figs.get("kz_scaling") if isinstance(figs, dict) else None
				if fig is not None:
					self._display_figure(fig)
				self._plot_status.set(f"Saved KZ scaling -> {out_png} (runs={len(rows)})")

			elif m == "10":
				parent_dir = str(self._plot_vars["m10_parent"].get()).strip() or "."
				pattern = str(self._plot_vars["m10_pattern"].get()).strip() or "output_quench*"
				x_axis = str(self._plot_vars["m10_xaxis"].get()).strip() or "t_ramp"
				Tc = float(str(self._plot_vars["m10_tc"].get()).strip() or "310.2")
				snapfreq = int(float(str(self._plot_vars["m10_snapfreq"].get()).strip() or "10000"))
				off_s = str(self._plot_vars["m10_offsets"].get()).strip()
				offsets = None
				if off_s:
					offsets = [int(s.strip()) for s in off_s.split(",") if s.strip()]
				sthr = float(str(self._plot_vars["m10_sthr"].get()).strip() or "0.02")
				z_s = str(self._plot_vars["m10_z"].get()).strip()
				z_slice = int(float(z_s)) if z_s else None
				z_avg = int(float(str(self._plot_vars["m10_zavg"].get()).strip() or "11"))
				z_marg = float(str(self._plot_vars["m10_zmarg"].get()).strip() or "0.2")
				res = v.sweep_kz_slope_stability(
					parent_dir,
					pattern=pattern,
					out_dir=out_dir,
					x_axis=x_axis,
					S_threshold=sthr,
					z_slice=z_slice,
					z_avg=z_avg,
					z_margin_frac=z_marg,
					Tc=Tc,
					snapshotFreq_iters=snapfreq,
					offsets_in_snaps=offsets,
					show=False,
					close=False,
					return_fig=True,
				)
				slopes = res[0]
				out_png = res[1]
				fig = res[3] if len(res) >= 4 else None
				if fig is None:
					raise RuntimeError("sweep_kz_slope_stability did not return a Figure (return_fig=True)")
				self._display_figure(fig)
				self._plot_status.set(f"Saved slope stability -> {out_png} (n={len(slopes)})")

			elif m == "11":
				field_fp = self._pick_field_file(src)
				sxi = float(str(self._plot_vars["m11_sxi"].get()).strip() or "0.1")
				sdrop = float(str(self._plot_vars["m11_sdrop"].get()).strip() or "0.1")
				score = float(str(self._plot_vars["m11_score"].get()).strip() or "0.05")
				dilate = int(float(str(self._plot_vars["m11_dilate"].get()).strip() or "2"))
				mincore = int(float(str(self._plot_vars["m11_mincore"].get()).strip() or "30"))
				use_skel = str(self._plot_vars["m11_skeleton"].get()).strip().lower() == "true"
				xi3, _corr, _rs, dims = v.correlation_length_3d_from_field_file(field_fp, S_threshold=sxi)
				defect = v.defect_line_metrics_3d_from_field_file(
					field_fp,
					S_droplet=sdrop,
					S_core=score,
					dilate_iters=dilate,
					min_core_voxels=mincore,
					use_skeleton=use_skel,
					return_masks=True,
				)
				Nx, Ny, Nz = int(defect.get("Nx", dims[0])), int(defect.get("Ny", dims[1])), int(defect.get("Nz", dims[2]))
				S, _nx, _ny, _nz = v.load_nematic_field_volume(field_fp, Nx, Ny, Nz)
				z = Nz // 2
				core = defect.get("core_mask")
				skel = defect.get("skeleton_mask")
				fig, axes = plt.subplots(1, 3, figsize=(12, 4))
				axes[0].imshow(np.asarray(S)[:, :, z].T, origin="lower", cmap="viridis")
				axes[0].set_title(f"S (z={z})")
				axes[0].set_axis_off()
				axes[1].imshow(np.asarray(core)[:, :, z].T if core is not None else np.zeros((Ny, Nx)), origin="lower", cmap="gray")
				axes[1].set_title("core mask")
				axes[1].set_axis_off()
				axes[2].imshow(np.asarray(skel)[:, :, z].T if skel is not None else np.zeros((Ny, Nx)), origin="lower", cmap="gray")
				axes[2].set_title("skeleton")
				axes[2].set_axis_off()
				fig.suptitle(
					f"xi_3D={xi3:.3g} | line_density={defect.get('line_density_per_system_voxel', float('nan')):.3g} | "
					f"core_density={defect.get('core_density_per_system_voxel', float('nan')):.3g}"
				)
				fig.tight_layout(rect=(0, 0, 1, 0.92))
				self._display_figure(fig)
				self._plot_status.set("Computed 3D snapshot metrics (SciPy + scikit-image required)")

			elif m == "12":
				parent_dir = str(self._plot_vars["m12_parent"].get()).strip() or "."
				pattern = str(self._plot_vars["m12_pattern"].get()).strip() or "output_quench*"
				x_axis = str(self._plot_vars["m12_xaxis"].get()).strip() or "t_ramp"
				measure = str(self._plot_vars["m12_measure"].get()).strip() or "after_Tc"
				Tc = float(str(self._plot_vars["m12_tc"].get()).strip() or "310.2")
				after_Tc_s = float(str(self._plot_vars["m12_after_tc"].get()).strip() or "0.0")
				sxi = float(str(self._plot_vars["m12_sxi"].get()).strip() or "0.1")
				sdrop = float(str(self._plot_vars["m12_sdrop"].get()).strip() or "0.1")
				score = float(str(self._plot_vars["m12_score"].get()).strip() or "0.05")
				dilate = int(float(str(self._plot_vars["m12_dilate"].get()).strip() or "2"))
				mincore = int(float(str(self._plot_vars["m12_mincore"].get()).strip() or "30"))
				proxy = str(self._plot_vars["m12_proxy"].get()).strip() or "skeleton"
				maxruns_s = str(self._plot_vars["m12_maxruns"].get()).strip()
				max_runs = int(float(maxruns_s)) if maxruns_s else None
				res = v.aggregate_kz_scaling_3d(
					parent_dir,
					pattern=pattern,
					out_dir=out_dir,
					x_axis=x_axis,
					measure=measure,
					Tc=Tc,
					after_Tc_s=after_Tc_s,
					S_threshold_xi=sxi,
					S_droplet=sdrop,
					S_core=score,
					dilate_iters=dilate,
					min_core_voxels=mincore,
					defect_proxy=proxy,
					max_runs=max_runs,
					show=False,
					close=False,
					return_fig=True,
				)
				rows = res[0]
				out_png = res[1]
				fig = res[3] if len(res) >= 4 else None
				if fig is None:
					raise RuntimeError("aggregate_kz_scaling_3d did not return a Figure (return_fig=True)")
				self._display_figure(fig)
				self._plot_status.set(f"Saved 3D KZ scaling -> {out_png} (runs={len(rows)})")

			elif m == "5":
				p = Path(src)
				if p.is_file():
					raise ValueError("Mode 5 expects a directory (repo root or output_temp_sweep folder)")
				data_root = p
				if not any(data_root.glob("T_*/")):
					cand = data_root / "output_temp_sweep"
					if cand.is_dir() and any(cand.glob("T_*/")):
						data_root = cand
				out_kind = str(self._plot_vars["m5_output"].get()).strip().lower() or "gif"
				choice = "g" if out_kind == "gif" else "f"
				color_field = str(self._plot_vars["m5_color"].get()).strip() or "S"
				duration = float(str(self._plot_vars["m5_duration"].get()).strip() or "0.5")
				selected = str(self._plot_vars["m5_selected"].get()).strip() or "all"
				gif_path = v.animate_tempSweep(
					data_root=str(data_root),
					color_field=color_field,
					choice=choice,
					selected=selected,
					out_dir=out_dir,
					duration=duration,
					return_gif_path=True,
				)
				if choice == "g":
					if not gif_path:
						raise RuntimeError("animate_tempSweep did not produce a GIF")
					try:
						import imageio.v2 as imageio
					except Exception:
						import imageio
						imageio = imageio  # type: ignore
					frame0 = imageio.imread(gif_path)
					fig, ax = plt.subplots(1, 1, figsize=(8, 6))
					ax.imshow(frame0)
					ax.set_axis_off()
					ax.set_title(Path(gif_path).name)
					fig.tight_layout()
					self._display_figure(fig)
					self._plot_status.set(f"Saved temp sweep GIF -> {gif_path} (duration={duration:g}s)")
				else:
					self._plot_status.set(f"Saved selected temp sweep frames to {out_dir}")
			else:
				self._plot_status.set("Mode not implemented in GUI yet.")
		except Exception as e:
			self._plot_status.set(f"Plot failed: {e}")
			self._append_log(f"[GUI][Plot] Error: {e}\n")

	def _collect_config_items(self) -> dict[str, str]:
		items: dict[str, str] = {}
		for key, var in self._vars.items():
			val = str(var.get()).strip()
			if not val:
				continue
			if val == "default":
				continue
			items[key] = val

		# If sim_mode is not set at all, default GUI to quench (3).
		if "sim_mode" not in items:
			items["sim_mode"] = "3"
		return items

	@staticmethod
	def _safe_float(items: dict[str, str], key: str) -> float | None:
		try:
			v = str(items.get(key, "")).strip()
			if not v:
				return None
			return float(v)
		except Exception:
			return None

	@staticmethod
	def _safe_int(items: dict[str, str], key: str) -> int | None:
		try:
			v = str(items.get(key, "")).strip()
			if not v:
				return None
			return int(float(v))
		except Exception:
			return None

	@staticmethod
	def _kz_zurek_from_tauQ(*, tau_Q_s: float, tau0_s: float, nu: float, z: float) -> dict[str, float] | None:
		"""Return KZ freeze-out (Zurek) estimates given tau_Q and microscopic tau0.

		Uses the common KZ scaling (linear quench in reduced temperature):
		  t_hat = tau0^(1/(1+z nu)) * tau_Q^(z nu/(1+z nu))
		  eps_hat = (tau0/tau_Q)^(1/(1+z nu))
		"""
		try:
			if not (tau_Q_s > 0.0 and tau0_s > 0.0):
				return None
			if not (nu > 0.0 and z > 0.0):
				return None
			a = 1.0 + z * nu
			if not (a > 0.0):
				return None
			inv = 1.0 / a
			t_hat = (tau0_s ** inv) * (tau_Q_s ** ((z * nu) * inv))
			eps_hat = (tau0_s / tau_Q_s) ** inv
			return {"t_hat_s": float(t_hat), "eps_hat": float(eps_hat), "a": float(a)}
		except Exception:
			return None

	def _log_zurek_estimate_from_items(self, items: dict[str, str], *, stage: str) -> None:
		"""Best-effort Zurek time estimate from GUI config (pre-run)."""
		try:
			protocol = self._safe_int(items, "protocol")
			snapshot_mode = self._safe_int(items, "snapshot_mode")
			Tc = self._safe_float(items, "Tc_KZ")
			T_high = self._safe_float(items, "T_high")
			T_low = self._safe_float(items, "T_low")
			ramp_iters = self._safe_int(items, "ramp_iters")
			dt = self._safe_float(items, "dt")
			nu = self._safe_float(items, "kz_nu") or 0.5
			z = self._safe_float(items, "kz_z") or 2.0
			tau0 = self._safe_float(items, "kz_tau0")
			if tau0 is None:
				tau0 = dt

			if protocol != 2:
				return
			if Tc is None or T_high is None or T_low is None or ramp_iters is None:
				return
			if not (ramp_iters > 0):
				return
			if dt is None or not (dt > 0.0):
				# Can't convert ramp_iters -> seconds (yet). We'll do a post-run estimate from quench_log.
				self._append_log(
					f"[GUI][KZ] ({stage}) Zurek-time estimate skipped (dt not set; will estimate after run from quench_log).\n"
				)
				return
			dT = (T_low - T_high)
			if dT == 0.0:
				return
			t_ramp = float(ramp_iters) * dt
			rate = dT / t_ramp  # K/s (signed)
			rate_abs = abs(rate)
			tau_Q = abs(Tc) / rate_abs if rate_abs > 0.0 else None
			if tau_Q is None or not (tau_Q > 0.0):
				return
			if tau0 is None or not (tau0 > 0.0):
				return

			kz = self._kz_zurek_from_tauQ(tau_Q_s=tau_Q, tau0_s=tau0, nu=nu, z=z)
			if kz is None:
				return
			t_hat = float(kz["t_hat_s"])
			eps_hat = float(kz["eps_hat"])
			dT_hat = abs(Tc) * eps_hat
			iters_hat = t_hat / dt
			win = self._safe_float(items, "Tc_window_K")
			win_note = ""
			if win is not None and win > 0.0:
				win_note = f" | Tc_window_K={win:g} (suggest ≥ {dT_hat:g})"
			elif snapshot_mode == 2:
				win_note = f" | (suggest Tc_window_K ≥ {dT_hat:g} for z freeze-out window)"

			self._append_log(
				"[GUI][KZ] "
				f"({stage}) ramp rate ≈ {rate_abs:.6g} K/s, tau_Q≈{tau_Q:.6g} s\n"
				f"          Using nu={nu:g}, z={z:g}, tau0={tau0:.6g} s => t_Zurek≈{t_hat:.6g} s (~{iters_hat:.3g} iters), |T-Tc|_Zurek≈{dT_hat:.6g} K{win_note}\n"
			)
		except Exception as e:
			self._append_log(f"[GUI][KZ] ({stage}) Zurek estimate failed: {e}\n")

	@staticmethod
	def _read_kv_file(path: Path) -> dict[str, str]:
		try:
			text = path.read_text(encoding="utf-8", errors="ignore")
			return _parse_kv_config(text)
		except Exception:
			return {}

	@staticmethod
	def _read_quench_log_time_temp(path: Path) -> tuple[list[float], list[float], list[float], list[int]]:
		"""Return (t_s, T_K, dt_s, iter) from quench_log.dat (best-effort)."""
		t_s: list[float] = []
		T_K: list[float] = []
		dt_s: list[float] = []
		iters: list[int] = []
		try:
			with path.open("r", encoding="utf-8", errors="ignore") as f:
				for line in f:
					line = line.strip()
					if not line or line.startswith("#"):
						continue
					parts = [p.strip() for p in line.split(",")]
					# Expected columns (from QSR.cu): iter,time_s,dt,T_current,...
					if len(parts) < 5:
						continue
					try:
						it = int(float(parts[0]))
						t = float(parts[1])
						dt = float(parts[2])
						T = float(parts[3])
					except Exception:
						continue
					iters.append(it)
					t_s.append(t)
					dt_s.append(dt)
					T_K.append(T)
		except Exception:
			pass
		return t_s, T_K, dt_s, iters

	def _log_zurek_estimate_post_run(self, out_dir: Path) -> None:
		"""Estimate Zurek time from the produced quench_log.dat (post-run)."""
		try:
			cfg = self._read_kv_file(out_dir / "run_config.cfg")
			protocol = self._safe_int(cfg, "protocol")
			if protocol != 2:
				return
			Tc = self._safe_float(cfg, "Tc_KZ")
			if Tc is None or not (Tc > 0.0):
				return
			nu = self._safe_float(cfg, "kz_nu") or 0.5
			z = self._safe_float(cfg, "kz_z") or 2.0
			tau0 = self._safe_float(cfg, "kz_tau0")

			log_path = out_dir / "quench_log.dat"
			if not log_path.exists():
				return
			t_s, T_K, dt_s, iters = self._read_quench_log_time_temp(log_path)
			if len(t_s) < 5:
				return
			# Fall back tau0 to median dt if user didn't supply one.
			if tau0 is None or not (tau0 > 0.0):
				# pick a robust dt (last finite)
				for d in reversed(dt_s):
					if d > 0.0:
						tau0 = d
						break
			if tau0 is None or not (tau0 > 0.0):
				return

			# Find a neighborhood around Tc and fit a local slope dT/dt.
			Tc_window = self._safe_float(cfg, "Tc_window_K") or 0.5
			win = max(float(Tc_window), 0.05)
			idx = [i for i, T in enumerate(T_K) if abs(T - Tc) <= win]
			if len(idx) < 3:
				# widen if needed
				win2 = max(win * 2.0, 0.5)
				idx = [i for i, T in enumerate(T_K) if abs(T - Tc) <= win2]
			if len(idx) < 3:
				return
			# Simple least-squares slope for T(t) in the selected window.
			# slope = cov(t,T)/var(t)
			tm = sum(t_s[i] for i in idx) / len(idx)
			Tm = sum(T_K[i] for i in idx) / len(idx)
			cov = sum((t_s[i] - tm) * (T_K[i] - Tm) for i in idx)
			var = sum((t_s[i] - tm) ** 2 for i in idx)
			if var <= 0.0:
				return
			slope = cov / var  # K/s
			rate_abs = abs(slope)
			if not (rate_abs > 0.0):
				return
			tau_Q = abs(Tc) / rate_abs

			kz = self._kz_zurek_from_tauQ(tau_Q_s=tau_Q, tau0_s=float(tau0), nu=nu, z=z)
			if kz is None:
				return
			t_hat = float(kz["t_hat_s"])
			eps_hat = float(kz["eps_hat"])
			dT_hat = abs(Tc) * eps_hat
			# Estimate crossing time by interpolation
			cross_t = None
			cross_it = None
			for i in range(1, len(T_K)):
				T0, T1 = T_K[i - 1], T_K[i]
				if (T0 - Tc) == 0.0:
					cross_t = t_s[i - 1]
					cross_it = iters[i - 1]
					break
				if (T0 - Tc) * (T1 - Tc) <= 0.0 and T1 != T0:
					frac = (Tc - T0) / (T1 - T0)
					cross_t = t_s[i - 1] + frac * (t_s[i] - t_s[i - 1])
					cross_it = int(round(iters[i - 1] + frac * (iters[i] - iters[i - 1])))
					break

			msg = (
				"[GUI][KZ] (post-run) Estimated from quench_log near Tc:\n"
				f"          |dT/dt|≈{rate_abs:.6g} K/s => tau_Q≈{tau_Q:.6g} s\n"
				f"          Using nu={nu:g}, z={z:g}, tau0={float(tau0):.6g} s => t_Zurek≈{t_hat:.6g} s, |T-Tc|_Zurek≈{dT_hat:.6g} K\n"
			)
			if cross_t is not None and cross_it is not None:
				msg += f"          Tc crossing at t≈{cross_t:.6g} s (iter≈{cross_it}); a common choice is to measure at t≈t_cross + c·t_Zurek (e.g. c=1..5).\n"
			self._append_log(msg)
		except Exception as e:
			self._append_log(f"[GUI][KZ] (post-run) Zurek estimate failed: {e}\n")

	def _on_run(self) -> None:
		if self._proc is not None and self._proc.poll() is None:
			messagebox.showwarning("Already running", "A simulation is already running. Stop it first.")
			return

		backend = Path(self.backend_path.get()).expanduser()
		if not backend.exists():
			messagebox.showerror("Backend not found", f"Backend executable not found:\n{backend}")
			return

		items = self._collect_config_items()
		out_dir = Path(items.get("out_dir", "output_quench")).expanduser()
		if not out_dir.is_absolute():
			out_dir = _repo_root() / out_dir
		items["out_dir"] = str(out_dir)

		# write config inside output directory
		out_dir.mkdir(parents=True, exist_ok=True)
		cfg_path = out_dir / "run_config.cfg"
		_write_kv_config(cfg_path, items)

		self.last_out_dir = out_dir
		self.log_text.delete("1.0", tk.END)
		self._append_log(f"[GUI] Wrote config: {cfg_path}\n")
		# Pre-run: report a best-effort KZ/Zurek time estimate if we can.
		self._log_zurek_estimate_from_items(items, stage="pre-run")

		cmd = [str(backend), "--config", str(cfg_path)]
		self._append_log(f"[GUI] Running: {' '.join(cmd)}\n")
		self.status.set("Running…")

		try:
			self._proc = subprocess.Popen(
				cmd,
				cwd=str(_repo_root()),
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				text=True,
				bufsize=1,
			)
		except Exception as e:
			self.status.set("Idle")
			messagebox.showerror("Failed to start", str(e))
			return

		self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
		self._reader_thread.start()

	def _on_stop(self) -> None:
		if self._proc is None or self._proc.poll() is not None:
			return
		self._append_log("[GUI] Terminating process…\n")
		try:
			self._proc.terminate()
		except Exception:
			pass

	def _open_output_folder(self) -> None:
		if not self.last_out_dir:
			messagebox.showinfo("No output yet", "Run a simulation first.")
			return
		p = self.last_out_dir
		# Linux friendly
		try:
			subprocess.Popen(["xdg-open", str(p)])
		except Exception:
			messagebox.showinfo("Output folder", str(p))

	def _plot_last_run(self) -> None:
		if not self.last_out_dir:
			messagebox.showinfo("No output yet", "Run a simulation first.")
			return
		# Run QSRvis plotting for the chosen out_dir
		py = sys.executable
		out_dir = str(self.last_out_dir)
		cmd = [py, "-c", f"import QSRvis as v; v.plot_quench_log(r'{out_dir}', show=False)"]
		self._append_log(f"[GUI] Plotting: {' '.join(cmd)}\n")
		try:
			subprocess.Popen(cmd, cwd=str(_repo_root()))
		except Exception as e:
			messagebox.showerror("Plot failed", str(e))

	def _save_preset(self) -> None:
		items = self._collect_config_items()
		p = filedialog.asksaveasfilename(
			title="Save preset",
			defaultextension=".cfg",
			filetypes=[("QSR config", "*.cfg"), ("All files", "*.*")],
		)
		if not p:
			return
		_write_kv_config(Path(p), items)
		self._append_log(f"[GUI] Saved preset: {p}\n")

	def _load_preset(self) -> None:
		p = filedialog.askopenfilename(
			title="Load preset",
			filetypes=[("QSR config", "*.cfg"), ("All files", "*.*")],
		)
		if not p:
			return
		text = Path(p).read_text(encoding="utf-8", errors="ignore")
		items = _parse_kv_config(text)
		for key, var in self._vars.items():
			if key in items:
				var.set(items[key])
			else:
				# reset to unset
				if isinstance(var, tk.StringVar) and var.get() in ("default", "true", "false"):
					var.set("default")
				else:
					var.set("")
		self._append_log(f"[GUI] Loaded preset: {p}\n")

	# ---------------- Process I/O ----------------

	def _reader_loop(self) -> None:
		assert self._proc is not None
		assert self._proc.stdout is not None
		for line in self._proc.stdout:
			self._log_queue.put(line)

		rc = self._proc.wait()
		self._log_queue.put(f"\n[GUI] Process exited with code {rc}\n")
		self._log_queue.put("__DONE__")

	def _drain_log_queue(self) -> None:
		try:
			while True:
				msg = self._log_queue.get_nowait()
				if msg == "__DONE__":
					self.status.set("Idle")
					# Post-run: estimate KZ/Zurek time from the actual temperature trace.
					if self.last_out_dir:
						self._log_zurek_estimate_post_run(self.last_out_dir)
					self._proc = None
					break
				self._append_log(msg)
		except queue.Empty:
			pass
		self.after(50, self._drain_log_queue)

	def _append_log(self, msg: str) -> None:
		self.log_text.insert(tk.END, msg)
		self.log_text.see(tk.END)


def main() -> None:
	app = QSRGui()
	app.mainloop()


if __name__ == "__main__":
	main()

