from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentVersion:
	key: str
	label: str
	artifact: str
	version: str
	notes: str = ""


COMPONENT_VERSIONS: tuple[ComponentVersion, ...] = (
	ComponentVersion(
		key="gui",
		label="QSR GUI",
		artifact="GUI.py",
		version="1.0.1",
		notes="Launcher update with production-review tooling and IMEX-aware form cleanup.",
	),
	ComponentVersion(
		key="qsr_cuda",
		label="Confined QSR solver",
		artifact="exe/QSR_cuda",
		version="0.4.10",
		notes="Active CUDA solver branch.",
	),
	ComponentVersion(
		key="qsr_cpu",
		label="QSR CPU reference",
		artifact="QSR_cpu",
		version="prealpha",
		notes="Reference-only path until a stable CPU release exists.",
	),
	ComponentVersion(
		key="kzm_xy",
		label="Periodic XY benchmark solver",
		artifact="exe/KZM_prooving_ground_cuda",
		version="1.0.0",
		notes="Periodic XY Kibble-Zurek benchmark executable.",
	),
	ComponentVersion(
		key="kzm_bulk_ldg",
		label="Periodic bulk-LdG benchmark solver",
		artifact="exe/KZM_bulk_ldg_cuda",
		version="1.0.0",
		notes="Periodic bulk Landau-de Gennes benchmark executable.",
	),
)


COMPONENT_VERSION_BY_KEY: dict[str, ComponentVersion] = {
	component.key: component for component in COMPONENT_VERSIONS
}


GUI_VERSION = COMPONENT_VERSION_BY_KEY["gui"].version