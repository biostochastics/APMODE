# SPDX-License-Identifier: GPL-2.0-or-later
"""Stan parameter-name → diagnostic class mapping.

Single source of truth shared by both producers (the Bayesian harness,
which buckets per-parameter R-hat / ESS into class-level worst-case
fields on :class:`PosteriorDiagnostics`) and consumers (the Gate 1
Bayesian evaluator in :mod:`apmode.governance.gates`, which compares
those buckets against per-class policy thresholds).

The classifier matches the DSL's on-disk naming (see
``apmode/dsl/priors.py`` / ``apmode/dsl/stan_emitter.py``).
"""

from __future__ import annotations

from typing import Literal

ParamClass = Literal["fixed_effects", "iiv", "residual", "correlations"]

PARAM_CLASS_NAMES: tuple[ParamClass, ...] = (
    "fixed_effects",
    "iiv",
    "residual",
    "correlations",
)


def classify_param_class(name: str) -> ParamClass:
    """Map a Stan parameter name to its :class:`Gate1BayesianConfig` class.

    Recognised prefixes:

    * ``omega_<p>`` → ``iiv`` (between-subject SDs, centered or
      non-centered — both produce ``omega_*`` names after decomposition).
    * ``sigma_``-prefixed / ``residual_sd`` / ``sigma_prop`` /
      ``sigma_add`` → ``residual``.
    * ``L_corr_``, ``corr_iiv``, ``L_Omega`` → ``correlations``.
    * everything else → ``fixed_effects`` (structural parameters and
      covariate betas).

    "Strict side" here means *highest diagnostic bar* — unknown names
    default to ``fixed_effects`` because that bucket has the tightest
    R-hat and highest ESS floors. The consequence is more rigorous
    gating, not blanket rejection: a fit with healthy mixing on an
    unknown parameter will still pass.
    """
    if name.startswith(("L_corr_", "L_Omega")) or name == "corr_iiv":
        return "correlations"
    if name.startswith("omega_"):
        return "iiv"
    if name.startswith("sigma_") or name in {"residual_sd", "sigma_prop", "sigma_add"}:
        return "residual"
    return "fixed_effects"
