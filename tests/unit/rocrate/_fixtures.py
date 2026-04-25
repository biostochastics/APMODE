# SPDX-License-Identifier: GPL-2.0-or-later
"""Reusable bundle fixtures for the RO-Crate projection tests.

These helpers construct small, realistic sealed bundles on disk using
the canonical :class:`apmode.bundle.emitter.BundleEmitter`. Tests that
need a richer bundle can compose :func:`build_submission_bundle` with
additional emitter calls.

Goals:

- Stay decoupled from the specific Pydantic field set used by
  higher-level pipelines — fixtures write only the fields our
  projector actually reads.
- Seal the bundle so that :class:`RoCrateEmitter` accepts it.
- Keep file counts small so golden snapshots stay reviewable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _digest_bundle(run_dir: Path) -> str:
    digest = hashlib.sha256()
    # Mirror ``apmode.bundle.emitter._DIGEST_EXCLUDED_RELATIVE_PATHS``: sentinel
    # itself + post-seal sidecars (CycloneDX SBOM, SBC manifest) that
    # are explicitly excluded so regenerating them never invalidates
    # ``_COMPLETE``.
    excluded = {"_COMPLETE", "bom.cdx.json", "sbc_manifest.json"}
    for p in sorted(run_dir.rglob("*"), key=lambda q: q.relative_to(run_dir).as_posix()):
        if not p.is_file() or p.name in excluded:
            continue
        digest.update(p.relative_to(run_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(p.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def build_submission_bundle(
    tmp_path: Path,
    *,
    run_id: str = "canonical-submission-run",
    candidate_ids: tuple[str, ...] = ("cand001",),
    add_credibility: bool = False,
    add_bayesian: bool = False,
    add_agentic: bool = False,
    add_regulatory: bool = False,
) -> Path:
    """Build a sealed Submission-lane bundle suitable for RO-Crate export.

    Returns the path to the bundle directory.
    """
    bundle = tmp_path / run_id
    bundle.mkdir()

    _write_json(
        bundle / "data_manifest.json",
        {
            "data_sha256": "0" * 64,
            "ingestion_format": "nonmem_csv",
            "column_mapping": {
                "subject_id": "ID",
                "time": "TIME",
                "dv": "DV",
                "evid": "EVID",
                "amt": "AMT",
            },
            "n_subjects": 12,
            "n_observations": 120,
            "n_doses": 12,
        },
    )
    _write_json(
        bundle / "seed_registry.json",
        {
            "root_seed": 42,
            "r_seed": 42,
            "r_rng_kind": "L'Ecuyer-CMRG",
            "np_seed": 42,
        },
    )
    _write_json(
        bundle / "backend_versions.json",
        {
            "apmode_version": "0.6.0",
            "python_version": "3.12",
            "r_version": "4.4.0",
            "nlmixr2_version": "3.0.0",
        },
    )
    _write_json(
        bundle / "evidence_manifest.json",
        {
            "manifest_schema_version": 3,
            "route_certainty": "confirmed",
            "absorption_complexity": "simple",
            "nonlinear_clearance_evidence_strength": "none",
            "multi_dose_detected": False,
            "richness_category": "moderate",
            "identifiability_ceiling": "medium",
            "covariate_burden": 0,
            "covariate_correlated": False,
            "blq_burden": 0.0,
            "protocol_heterogeneity": "single-study",
            "absorption_phase_coverage": "adequate",
            "elimination_phase_coverage": "adequate",
        },
    )
    _write_json(
        bundle / "policy_file.json",
        {
            "policy_version": "0.5.0",
            "lane": "submission",
            "gate1_thresholds": {"min_subjects": 5},
            "gate2_thresholds": {"min_obs": 20},
        },
    )
    _write_json(
        bundle / "split_manifest.json",
        {"split_seed": 42, "split_strategy": "subject_level", "assignments": []},
    )
    _write_json(
        bundle / "initial_estimates.json",
        {"entries": {}},
    )
    _write_json(
        bundle / "ranking.json",
        {
            "ranked_candidates": [
                {
                    "candidate_id": cid,
                    "rank": i + 1,
                    "bic": 100.0 + i,
                    "n_params": 4,
                    "backend": "nlmixr2",
                }
                for i, cid in enumerate(candidate_ids)
            ],
            "best_candidate_id": candidate_ids[0] if candidate_ids else None,
            "ranking_metric": "bic",
            "n_survivors": len(candidate_ids),
        },
    )

    # Candidate lineage
    _write_json(
        bundle / "candidate_lineage.json",
        {
            "entries": [
                {"candidate_id": candidate_ids[0], "parent_id": None, "transform": None},
                *(
                    {
                        "candidate_id": cid,
                        "parent_id": candidate_ids[0],
                        "transform": "add_second_cmt",
                    }
                    for cid in candidate_ids[1:]
                ),
            ]
        },
    )

    # Search trajectory & graph (minimal)
    (bundle / "search_trajectory.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "candidate_id": cid,
                    "parent_id": None,
                    "backend": "nlmixr2",
                    "converged": True,
                    "ofv": 100.0,
                    "timestamp": "2026-04-17T10:00:00Z",
                }
            )
            for cid in candidate_ids
        )
        + "\n"
    )
    (bundle / "failed_candidates.jsonl").write_text("")
    _write_json(
        bundle / "search_graph.json",
        {
            "nodes": [
                {"candidate_id": cid, "parent_id": None, "backend": "nlmixr2"}
                for cid in candidate_ids
            ],
            "edges": [],
        },
    )

    # Compiled specs + results for each candidate
    (bundle / "compiled_specs").mkdir()
    (bundle / "results").mkdir()
    (bundle / "gate_decisions").mkdir()
    for cid in candidate_ids:
        _write_json(
            bundle / "compiled_specs" / f"{cid}.json",
            {
                "model_id": cid,
                "absorption": {"type": "FirstOrder", "ka": 1.0},
                "distribution": {"type": "OneCmt", "V": 70.0},
                "elimination": {"type": "LinearElim", "CL": 5.0},
            },
        )
        # A tiny .R companion file so the entity picks up the R source
        (bundle / "compiled_specs" / f"{cid}.R").write_text(
            f"# nlmixr2 model {cid}\nfunction() {{}}\n"
        )
        _write_json(
            bundle / "results" / f"{cid}_result.json",
            {
                "model_id": cid,
                "backend": "nlmixr2",
                "converged": True,
                "ofv": 100.0,
                "aic": 110.0,
                "bic": 120.0,
                "parameter_estimates": {},
                "wall_time_seconds": 10.0,
            },
        )
        _write_json(
            bundle / "gate_decisions" / f"gate1_{cid}.json",
            {
                "gate_id": "gate1",
                "gate_name": "Technical Validity",
                "candidate_id": cid,
                "passed": True,
                "checks": [],
                "summary_reason": "All checks passed",
                "policy_version": "0.5.0",
                "timestamp": "2026-04-17T10:05:00Z",
            },
        )
        _write_json(
            bundle / "gate_decisions" / f"gate3_{cid}.json",
            {
                "gate_id": "gate3",
                "gate_name": "Submission Ranking",
                "candidate_id": cid,
                "passed": True,
                "checks": [],
                "summary_reason": "Ranked",
                "policy_version": "0.5.0",
                "timestamp": "2026-04-17T10:07:00Z",
            },
        )

    if add_credibility:
        (bundle / "credibility").mkdir()
        for cid in candidate_ids:
            _write_json(
                bundle / "credibility" / f"{cid}.json",
                {
                    "candidate_id": cid,
                    "context_of_use": "population PK characterisation",
                    "model_credibility": {"sensitivity_ok": True},
                    "data_adequacy": "adequate",
                    "limitations": [],
                },
            )

    if add_bayesian:
        (bundle / "bayesian").mkdir()
        for cid in candidate_ids:
            _write_json(
                bundle / "bayesian" / f"{cid}_prior_manifest.json",
                {
                    "policy_version": "0.5.0",
                    "entries": [],
                    "default_prior_policy": "weakly_informative",
                },
            )
            _write_json(
                bundle / "bayesian" / f"{cid}_simulation_protocol.json",
                {
                    "policy_version": "0.5.0",
                    "scenarios": [{"name": "base", "n_subjects": 12, "n_replicates": 100}],
                    "metrics": ["vpc_coverage"],
                    "seed": 42,
                },
            )
            _write_json(
                bundle / "bayesian" / f"{cid}_mcmc_diagnostics.json",
                {
                    "rhat_max": 1.01,
                    "ess_bulk_min": 400.0,
                    "ess_tail_min": 400.0,
                    "n_divergent": 0,
                    "n_max_treedepth": 0,
                    "ebfmi_min": 0.3,
                },
            )

    if add_agentic:
        (bundle / "agentic_trace").mkdir()
        for i in (1, 2):
            iteration = f"iter{i:03d}"
            _write_json(
                bundle / "agentic_trace" / f"{iteration}_input.json",
                {
                    "iteration_id": iteration,
                    "run_id": run_id,
                    "candidate_id": candidate_ids[0],
                    "prompt_hash": f"{i:064x}",
                    "prompt_template": "propose_transform",
                    "dsl_spec_json": "{}",
                },
            )
            _write_json(
                bundle / "agentic_trace" / f"{iteration}_output.json",
                {
                    "iteration_id": iteration,
                    "raw_output": f"Transform {i}",
                    "parsed_transforms": [f"add_transit_compartments:{i}"],
                    "validation_passed": True,
                    "validation_errors": [],
                },
            )
            _write_json(
                bundle / "agentic_trace" / f"{iteration}_meta.json",
                {
                    "iteration_id": iteration,
                    "model_id": "claude-sonnet-4",
                    "model_version": "20250514",
                    "prompt_hash": f"{i:064x}",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.001,
                    "temperature": 0.0,
                    "wall_time_seconds": 1.0,
                },
            )

    if add_regulatory:
        (bundle / "regulatory").mkdir()
        _write_json(
            bundle / "regulatory" / "md.json",
            {"modification_description": "Add absorption-lag branch."},
        )
        _write_json(
            bundle / "regulatory" / "mp.json",
            {"modification_protocol": "LORO-CV across regimens."},
        )
        _write_json(
            bundle / "regulatory" / "ia.json",
            {"impact_assessment": "Expected AUC GMR within BE window."},
        )
        (bundle / "regulatory" / "traceability.csv").write_text(
            "modification,md,mp,ia\nabs-lag,md.json,mp.json,ia.json\n"
        )

    # Seal last. ``sealed_at`` is carried inside ``_COMPLETE`` so the
    # RO-Crate projector can stamp a cross-host-stable
    # ``datePublished`` without depending on the sentinel's filesystem
    # mtime (which diverges when the bundle is copied between hosts).
    digest = _digest_bundle(bundle)
    (bundle / "_COMPLETE").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": run_id,
                "sha256": digest,
                "sealed_at": "2026-04-17T10:00:00Z",
            },
            indent=2,
        )
        + "\n"
    )
    return bundle
