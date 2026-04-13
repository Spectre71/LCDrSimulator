from __future__ import annotations

from dataclasses import dataclass


def _lines(*parts: str) -> str:
    return "\n".join(part.rstrip() for part in parts if part and part.strip())


@dataclass(frozen=True)
class QSRToolSpec:
    key: str
    label: str
    module_name: str
    summary: str
    help_text: str
    arg_template: str = ""
    uses_source: bool = True
    uses_out_dir: bool = True
    source_label: str = "Sweep root / primary path"


QSR_TOOL_SPECS: tuple[QSRToolSpec, ...] = (
    QSRToolSpec(
        key="analyze_kzm_prooving_ground",
        label="Analyze Periodic XY Benchmark",
        module_name="tools.analyze_kzm_prooving_ground",
        summary="Reduce a periodic XY rate sweep into per-run tables and fitted Kibble-Zurek benchmark observables.",
        help_text=_lines(
            "Use this on the proving-ground periodic XY sweep root containing cases/*/output/quench_log.dat.",
            "The tool supports final, fixed-after-Tc, and matched-amplitude readouts and writes detail plus summary CSV files.",
            "This is the analysis step that feeds the periodic XY benchmark figure.",
        ),
        arg_template='"{source}" --analysis-dir "{out_dir}"',
    ),
    QSRToolSpec(
        key="analyze_kzm_bulk_ldg",
        label="Analyze Periodic Bulk-LdG Benchmark",
        module_name="tools.analyze_kzm_bulk_ldg",
        summary="Reduce a periodic bulk Landau-de Gennes rate sweep into fitted benchmark observables.",
        help_text=_lines(
            "Use this on the periodic bulk-LdG sweep root containing generated configs and quench logs under cases/*/output/.",
            "The tool compares final, fixed-after-Tc, and matched-order readouts and writes detail plus summary CSV files.",
            "This is the analysis step that feeds the periodic bulk-LdG benchmark figure.",
        ),
        arg_template='"{source}" --analysis-dir "{out_dir}"',
    ),
    QSRToolSpec(
        key="check_quench_protocol_convergence",
        label="Check Quench Protocol Convergence",
        module_name="tools.check_quench_protocol_convergence",
        summary="Run matched fixed-dt protocol checks across one or more configs and compare post-crossing observables.",
        help_text=_lines(
            "Use this when you want coarse/fine or otherwise matched protocol comparisons with the same physics and geometry.",
            "The primary path placeholder inserts one config path; replace it with multiple quoted configs when running an ordered ladder.",
            "Outputs are written under the chosen output root and include protocol metrics plus matched-offset comparison CSV files.",
        ),
        arg_template='"{source}" --binary "{binary}" --output-root "{out_dir}"',
        source_label="Config path (replace with multiple quoted configs when needed)",
    ),
    QSRToolSpec(
        key="analyze_confined_final_state",
        label="Analyze Confined Final State",
        module_name="tools.analyze_confined_final_state",
        summary="Summarize final-state confined 3D defect metrics across a quench-rate sweep.",
        help_text=_lines(
            "Use this on a confined sweep root containing cases/*/output and rate_sweep_metrics.csv.",
            "The tool wraps the 3D aggregate scaling path in QSRvis and compares final-state observables against both ramp time and rate.",
            "It writes per-proxy CSV tables together with a compact final-state summary CSV.",
        ),
        arg_template='"{source}" --analysis-dir "{out_dir}"',
    ),
    QSRToolSpec(
        key="analyze_midplane_slab",
        label="Analyze Midplane Slab",
        module_name="tools.analyze_midplane_slab",
        summary="Measure the 2D defect observable across centered midplane slabs of different half-widths.",
        help_text=_lines(
            "Use this on a confined sweep root when you want an equatorial slab observable instead of a shell-depth decomposition.",
            "The tool evaluates one or more late offsets after Tc and writes per-window detail plus summary CSV files.",
            "It is useful for checking whether a centered slab can capture the confined signal without shell localization.",
        ),
        arg_template='"{source}" --analysis-dir "{out_dir}" --offsets 5.0e-8,6.0e-8',
    ),
    QSRToolSpec(
        key="analyze_qsr_z_profile",
        label="Analyze Full Z Profile",
        module_name="tools.analyze_qsr_z_profile",
        summary="Measure defect-density profiles across all z slices and identify the strongest equatorial band.",
        help_text=_lines(
            "Use this on a confined sweep root when you want slice-by-slice localization instead of shell-depth bins.",
            "The tool writes detail, per-slice summary, and slab-band selection outputs for each requested offset after Tc.",
            "This is mainly a localization diagnostic rather than the current preferred confined observable.",
        ),
        arg_template='"{source}" --analysis-dir "{out_dir}" --offsets 5.0e-8,6.0e-8',
    ),
    QSRToolSpec(
        key="analyze_shell_depth",
        label="Analyze Shell Depth",
        module_name="tools.analyze_shell_depth",
        summary="Bin defects by inward shell depth and build shell-excluded interior scans.",
        help_text=_lines(
            "Use this on a confined sweep root as the first step in the shell-focus workflow.",
            "The tool writes fixed shell-depth tables, summary tables, profile plots, and shell-exclusion scans for each requested late offset.",
            "These outputs are the input for shell-band decomposition and focus-band selection.",
        ),
        arg_template='"{source}" --analysis-dir "{out_dir}" --offsets 4.5e-8,5.0e-8,5.5e-8,6.0e-8,6.5e-8',
    ),
    QSRToolSpec(
        key="analyze_shell_band_decomposition",
        label="Analyze Shell-Band Decomposition",
        module_name="tools.analyze_shell_band_decomposition",
        summary="Choose a common focus annulus and compare it against the outer skin and deeper bulk interior.",
        help_text=_lines(
            "Run this after shell-depth analysis on the same confined sweep root.",
            "The tool searches for a defendable focus band shared across the selected late offsets and writes common-band, region-summary, and depth-moment tables.",
            "Its outputs feed both the shell-focus summary figure and the pooled exponent extraction.",
        ),
        arg_template='"{source}" --analysis-dir "{out_dir}" --offsets 4.5e-8,5.0e-8,5.5e-8,6.0e-8,6.5e-8',
    ),
    QSRToolSpec(
        key="analyze_shell_focus_exponent",
        label="Analyze Shell-Focus Exponent",
        module_name="tools.analyze_shell_focus_exponent",
        summary="Estimate pooled late-window exponents for the validated shell-focus annulus.",
        help_text=_lines(
            "Run this after shell-band decomposition on the same confined sweep root.",
            "The tool combines per-window log-log fits with a common-slope model that allows different late-window amplitudes.",
            "It writes detail tables, per-window fits, common-fit tables, a markdown summary, and a figure when requested.",
        ),
        arg_template='"{source}" --analysis-dir "{out_dir}" --offsets 4.5e-8,5.0e-8,5.5e-8,6.0e-8,6.5e-8',
    ),
    QSRToolSpec(
        key="plot_shell_focus_summary",
        label="Plot Shell-Focus Summary",
        module_name="tools.plot_shell_focus_summary",
        summary="Render the compact skin/focus/bulk summary figure and supporting tables from shell-band outputs.",
        help_text=_lines(
            "Use this after shell-depth analysis and shell-band decomposition on the same sweep root.",
            "The tool reads the decomposition outputs, writes PNG/PDF figure files, and emits compact CSV and Markdown tables.",
            "This is the main presentation layer for the current confined shell-focus result.",
        ),
        arg_template='"{source}" --offsets 4.5e-8,5.0e-8,5.5e-8,6.0e-8,6.5e-8 --png "{out_dir}/shell_focus_summary.png" --pdf "{out_dir}/shell_focus_summary.pdf"',
    ),
    QSRToolSpec(
        key="plot_confined_transient_bridge_summary",
        label="Plot Confined Transient Bridge",
        module_name="tools.plot_confined_transient_bridge_summary",
        summary="Render the confined transient bridge summary from existing merged cohort analyses.",
        help_text=_lines(
            "This figure is a presentation tool built from already reduced core and skeleton cohort summaries.",
            "It does not need a sweep-root placeholder unless you want to override the default summary CSV paths manually in the argument box.",
            "The GUI will preview the generated PNG when one is written.",
        ),
        arg_template='--png "{out_dir}/confined_transient_bridge_summary.png" --pdf "{out_dir}/confined_transient_bridge_summary.pdf"',
        uses_source=False,
        source_label="Primary path (unused by default)",
    ),
    QSRToolSpec(
        key="plot_xy_kzm_benchmark_figure",
        label="Plot Periodic XY Benchmark Figure",
        module_name="tools.plot_xy_kzm_benchmark_figure",
        summary="Regenerate the periodic XY benchmark figure from existing analysis artifacts.",
        help_text=_lines(
            "Use this when the proving-ground summary CSV files already exist and you only want the publication-style figure regenerated.",
            "The tool uses its own default validation paths unless you override them manually in the argument box.",
            "The GUI will preview the generated PNG when one is written.",
        ),
        arg_template='--output "{out_dir}/xy_kzm_benchmark.png" --output-pdf "{out_dir}/xy_kzm_benchmark.pdf"',
        uses_source=False,
        source_label="Primary path (unused by default)",
    ),
    QSRToolSpec(
        key="plot_bulk_ldg_benchmark_figure",
        label="Plot Periodic Bulk-LdG Benchmark Figure",
        module_name="tools.plot_bulk_ldg_benchmark_figure",
        summary="Regenerate the periodic bulk-LdG benchmark figure from existing analysis artifacts.",
        help_text=_lines(
            "Use this when the bulk benchmark summary CSV files already exist and you only want the publication-style figure regenerated.",
            "The tool uses its own default validation paths unless you override them manually in the argument box.",
            "The GUI will preview the generated PNG when one is written.",
        ),
        arg_template='--output "{out_dir}/bulk_ldg_benchmark.png" --output-pdf "{out_dir}/bulk_ldg_benchmark.pdf"',
        uses_source=False,
        source_label="Primary path (unused by default)",
    ),
)


QSR_TOOL_SPECS_BY_KEY: dict[str, QSRToolSpec] = {spec.key: spec for spec in QSR_TOOL_SPECS}