# SPDX-License-Identifier: GPL-2.0-or-later
"""End-to-end smoke test of the APMODE Bayesian backend on the Boeckmann 1994
theophylline dataset (12 subjects, single oral dose, ~10 samples/subject).

Published literature values for theophylline in adults (Boeckmann et al. 1994;
Upton 1982):
  ka ≈ 1.5 /h
  V  ≈ 0.5 L/kg (≈ 35 L for 70 kg adult)
  CL ≈ 0.04 L/h/kg (≈ 2.8 L/h for 70 kg adult)
  BSV ~ 30-50% CV

Priors here are weakly-informative centered on those literature values.

Short run: chains=2, warmup=300, sampling=300 — finishes in ~2-4 min on a
modern laptop. Not a production fit; just verifies the full pipeline.

Usage:
    uv run python scripts/bayesian_smoke_theophylline.py
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

from apmode.backends.bayesian_runner import BayesianRunner
from apmode.bundle.models import SamplerConfig
from apmode.dsl.ast_models import (
    DSLSpec,
    FirstOrder,
    IIV,
    LinearElim,
    OneCmt,
    Proportional,
)
from apmode.dsl.priors import (
    HalfCauchyPrior,
    NormalPrior,
    PriorSpec,
)


def build_theophylline_spec() -> DSLSpec:
    """1-cmt first-order absorption + linear elimination + proportional error."""
    return DSLSpec(
        model_id="theophylline_1cpt_po",
        absorption=FirstOrder(ka=1.5),
        distribution=OneCmt(V=35.0),
        elimination=LinearElim(CL=2.8),
        variability=[IIV(params=["CL", "V", "ka"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
        priors=[
            # Structural priors on log scale, centered on literature values.
            PriorSpec(
                target="CL",
                family=NormalPrior(mu=math.log(2.8), sigma=0.5),
            ),
            PriorSpec(
                target="V",
                family=NormalPrior(mu=math.log(35.0), sigma=0.5),
            ),
            PriorSpec(
                target="ka",
                family=NormalPrior(mu=math.log(1.5), sigma=0.5),
            ),
            # IIV SDs — weakly informative half-cauchy
            PriorSpec(target="omega_CL", family=HalfCauchyPrior(scale=0.5)),
            PriorSpec(target="omega_V", family=HalfCauchyPrior(scale=0.5)),
            PriorSpec(target="omega_ka", family=HalfCauchyPrior(scale=0.5)),
            # Residual SD
            PriorSpec(target="sigma_prop", family=HalfCauchyPrior(scale=0.3)),
        ],
    )


async def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    data = repo / "tests/fixtures/suite_b_theophylline/theophylline.csv"
    if not data.is_file():
        print(f"FATAL: dataset not found at {data}")
        return 2

    work = repo / "_smoke_bayes_work"
    work.mkdir(parents=True, exist_ok=True)

    spec = build_theophylline_spec()

    cfg = SamplerConfig(
        chains=2,
        warmup=300,
        sampling=300,
        adapt_delta=0.95,
        max_treedepth=10,
        seed=42,
        parallel_chains=2,
    )

    runner = BayesianRunner(work_dir=work, default_sampler_config=cfg)

    print(f"\n=== APMODE Bayesian smoke test on theophylline ===")
    print(f"  Dataset:  {data}")
    print(f"  Model:    {spec.model_id}")
    print(f"  Priors:   {len(spec.priors)} declared (see script)")
    print(f"  Sampler:  chains={cfg.chains} warmup={cfg.warmup} "
          f"sampling={cfg.sampling} adapt_delta={cfg.adapt_delta}")

    try:
        result = await runner.run(
            spec=spec,
            data_manifest=None,  # type: ignore[arg-type]
            initial_estimates={"CL": 2.8, "V": 35.0, "ka": 1.5},
            seed=42,
            data_path=data,
            timeout_seconds=900,
        )
    except Exception as exc:
        print(f"\nFAILED with exception: {type(exc).__name__}: {exc}")
        # Dump harness stderr if available
        latest = sorted(work.glob("*/response.json"))
        if latest:
            print(f"  latest response.json: {latest[-1]}")
            try:
                print(json.dumps(json.loads(latest[-1].read_text()), indent=2)[:2000])
            except Exception:
                pass
        return 1

    print("\n=== Results ===")
    print(f"  Backend:         {result.backend}")
    print(f"  Converged:       {result.converged}")
    print(f"  Wall time:       {result.wall_time_seconds:.1f}s")

    if result.posterior_diagnostics:
        pd = result.posterior_diagnostics
        print(f"\n=== MCMC diagnostics ===")
        print(f"  R-hat max:           {pd.rhat_max:.3f}  (want <= 1.01)")
        print(f"  ESS bulk min:        {pd.ess_bulk_min:.0f} (want >= 400)")
        print(f"  ESS tail min:        {pd.ess_tail_min:.0f} (want >= 400)")
        print(f"  Divergent:           {pd.n_divergent}     (want 0)")
        print(f"  Max treedepth hits:  {pd.n_max_treedepth}")
        print(f"  E-BFMI min:          {pd.ebfmi_min:.3f}  (want >= 0.3)")
        if pd.pareto_k_max is not None:
            print(f"  Pareto-k max:        {pd.pareto_k_max:.3f} (want <= 0.7)")

    print(f"\n=== Structural parameters (posterior summaries) ===")
    print(f"  {'Param':10} {'Est':>10} {'SD':>10} {'q05':>10} {'q50':>10} {'q95':>10}")
    for name in ("CL", "V", "ka"):
        if name in result.parameter_estimates:
            p = result.parameter_estimates[name]
            print(f"  {p.name:10} {p.estimate:10.3f} {(p.posterior_sd or 0):10.3f} "
                  f"{(p.q05 or 0):10.3f} {(p.q50 or 0):10.3f} {(p.q95 or 0):10.3f}")

    print(f"\n=== IIV omega + sigma ===")
    for name in ("omega_CL", "omega_V", "omega_ka", "sigma_prop"):
        if name in result.parameter_estimates:
            p = result.parameter_estimates[name]
            print(f"  {p.name:12} {p.estimate:10.3f} (sd {p.posterior_sd or 0:.3f})")

    if result.posterior_draws_path:
        draws = Path(result.posterior_draws_path)
        print(f"\nDraws parquet: {draws} (exists={draws.exists()})")

    # Literature-anchored sanity checks — not strict pass/fail, just alerts.
    print(f"\n=== Literature sanity checks (Boeckmann 1994 / Upton 1982) ===")
    checks = [
        ("CL", 2.8, 0.5, 10.0),
        ("V", 35.0, 5.0, 100.0),
        ("ka", 1.5, 0.2, 10.0),
    ]
    for name, ref, lo, hi in checks:
        if name in result.parameter_estimates:
            est = result.parameter_estimates[name].estimate
            ok = lo <= est <= hi
            marker = "OK" if ok else "SUSPECT"
            print(f"  [{marker}] {name} posterior mean {est:.2f} (reference ~{ref}, plausible {lo}-{hi})")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
