# SPDX-License-Identifier: GPL-2.0-or-later
"""Bundle emitter scaffolding (ARCHITECTURE.md §5).

Phase 1 Month 1-2: writes data_manifest, seed_registry, compiled_specs
(AST JSON + lowered R code), and backend_versions to the bundle directory.

Full bundle emission (gate decisions, search trajectory, etc.) is Month 5-6.
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 — used at runtime in __init__

from apmode.bundle.models import (  # noqa: TC001 — used at runtime in method signatures
    BackendVersions,
    DataManifest,
    SeedRegistry,
)
from apmode.dsl.ast_models import DSLSpec  # noqa: TC001 — used at runtime
from apmode.dsl.nlmixr2_emitter import emit_nlmixr2
from apmode.ids import generate_run_id


class BundleEmitter:
    """Writes reproducibility bundle artifacts to disk.

    Phase 1 scaffolding — supports the minimal artifact set needed for
    integration testing from Month 2:
    - data_manifest.json
    - seed_registry.json
    - backend_versions.json
    - compiled_specs/{candidate_id}.json  (DSLSpec AST)
    - compiled_specs/{candidate_id}.R     (lowered R code)
    """

    def __init__(self, base_dir: Path, run_id: str | None = None) -> None:
        self.run_id = run_id or generate_run_id()
        self.run_dir = base_dir / self.run_id
        self._compiled_specs_dir = self.run_dir / "compiled_specs"

    def initialize(self) -> Path:
        """Create the bundle directory structure. Returns the run directory."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._compiled_specs_dir.mkdir(exist_ok=True)
        (self.run_dir / "gate_decisions").mkdir(exist_ok=True)
        (self.run_dir / "results").mkdir(exist_ok=True)
        return self.run_dir

    def write_data_manifest(self, manifest: DataManifest) -> Path:
        """Write data_manifest.json."""
        path = self.run_dir / "data_manifest.json"
        path.write_text(manifest.model_dump_json(indent=2))
        return path

    def write_seed_registry(self, registry: SeedRegistry) -> Path:
        """Write seed_registry.json."""
        path = self.run_dir / "seed_registry.json"
        path.write_text(registry.model_dump_json(indent=2))
        return path

    def write_backend_versions(self, versions: BackendVersions) -> Path:
        """Write backend_versions.json."""
        path = self.run_dir / "backend_versions.json"
        path.write_text(versions.model_dump_json(indent=2))
        return path

    def write_compiled_spec(
        self,
        spec: DSLSpec,
        initial_estimates: dict[str, float] | None = None,
    ) -> tuple[Path, Path | None]:
        """Write compiled_specs/{candidate_id}.json and .R.

        NODE specs are written as JSON only (no R lowering in Phase 1).

        Returns (json_path, r_path). r_path is None for NODE specs.
        """
        json_path = self._compiled_specs_dir / f"{spec.model_id}.json"
        json_path.write_text(spec.model_dump_json(indent=2))

        if spec.has_node_modules():
            return json_path, None

        r_path = self._compiled_specs_dir / f"{spec.model_id}.R"
        r_code = emit_nlmixr2(spec, initial_estimates=initial_estimates)
        r_path.write_text(r_code)

        return json_path, r_path

    def write_policy_file(self, policy_data: dict[str, object]) -> Path:
        """Write policy_file.json (copy of the gate thresholds used for this run)."""
        path = self.run_dir / "policy_file.json"
        path.write_text(json.dumps(policy_data, indent=2, default=str))
        return path
