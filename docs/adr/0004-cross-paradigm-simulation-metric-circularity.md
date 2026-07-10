# ADR 0004 — Cross-Paradigm Simulation-Metric Circularity Risk

## Related documentation

- [../PRD_APMODE_v0.3.md](../PRD_APMODE_v0.3.md) — §4.3.1 (Gate 3 cross-paradigm ranking) and §10 Q2 (cross-paradigm NLPD comparability), the two sections this ADR's finding sits between.
- `src/apmode/governance/ranking.py` — module docstring cross-links here; `rank_cross_paradigm` implements the metrics this ADR analyzes.
- `src/apmode/governance/gates.py` — `_gate3_cross_paradigm` docstring cross-links here; consumes `rank_cross_paradigm`'s output and surfaces `paradigm_metric_spread` as an observability check.
- `tests/unit/test_cross_paradigm_ranking.py::TestSimulationMetricParadigmSensitivity` — pins the empirical result this ADR's Decision section reports.

**Date:** 2026-07-10
**Status:** Accepted (finding unresolved, re-evaluation trigger defined)
**Context:** Post-audit review flagged that Gate 3's cross-paradigm ranking
(PRD §4.3.1) disables BIC/NLPD by default and instead ranks candidates on
VPC coverage concordance, an NPE proxy, and (optionally) AUC/Cmax
bioequivalence — all three computed by simulating each candidate's *own*
fitted model forward and comparing the simulated trajectories to observed
data. `ranking.py`'s module docstring already explains *why likelihood is
excluded* (PRD §10 Q2: NLPD/BIC are not comparable across observation
models). It says nothing about whether the *replacement* metrics are
themselves paradigm-neutral.

---

## Finding

**The risk, stated verbatim.** VPC, NPE, and AUC/Cmax BE are all
self-referential: candidate C is scored by asking "how well does C's own
simulator, run forward from C's own fitted parameters, reproduce the
observed data C was fit to?" If the Suite A/B/C ground-truth data-generating
process (DGP) structurally resembles one paradigm's simulator more closely
than another's — e.g. a closed-form/ODE DGP is a much closer structural
match to a classical NLME simulator (which literally is that same
closed-form/ODE solver) than to a NODE simulator (whose learned dynamics
approximate the ODE but are not it) — then the paradigm whose simulator
structurally matches the DGP will show tighter VPC bands and lower NPE for
reasons that have nothing to do with which candidate would generalize
better to *new* data. This is a circularity risk, not a proven bias: it has
not been empirically confirmed or disproven for APMODE's actual metric
implementations prior to this ADR.

**Why this is not the same risk that PRD §10 Q2 already addresses.**
§10 Q2 is about likelihood-scale incomparability — NLPD/BIC computed under
different observation-model parameterizations aren't on the same numeric
scale, so ranking directly on them is invalid regardless of *any* fitting
quality question. The simulation-metric risk here is different in kind: VPC/
NPE/AUC-Cmax scores are already dimensionless-ish and nominally comparable
across paradigms by construction (percentile coverage, normalized residual,
bioequivalence fraction). The open question is not "are these on the same
scale" (they are) but "do they measure the same thing regardless of which
paradigm produced the candidate" (unverified prior to this ADR).

**Status: unresolved, not disproven.** This ADR does not claim the bias is
real or absent in general. Section "Decision" below reports one concrete
empirical result — under a controlled same-parameter DGP the ranker showed
no material paradigm-correlated spread — but that result covers a narrow
case (identical structural parameters fed through two numerically distinct
but structurally identical forward simulators) and does **not** generalize
to the harder case the risk statement describes: a NODE simulator whose
*learned* dynamics only approximate, rather than reproduce, the DGP's
governing ODE. That harder case is exactly the re-evaluation trigger below.

---

## Decision

Keep VPC/NPE/AUC-Cmax as the default cross-paradigm ranking metrics (no
change to `Gate3Config` defaults). Reasons:

1. No viable cross-paradigm alternative exists today. Likelihood-scale
   metrics are excluded per §10 Q2 for an orthogonal, already-settled
   reason; there is no third metric family in the codebase that avoids
   both problems simultaneously.
2. The empirical check added by this ADR
   (`TestSimulationMetricParadigmSensitivity::test_identical_parameters_across_simulator_paths_stay_within_tolerance`,
   `tests/unit/test_cross_paradigm_ranking.py`) exercises the actual Suite
   A ground-truth DGP for scenario A1 (`ka=1.5, V=70, CL=5` from
   `benchmarks/suite_a/reference_params.json`), scored through two
   structurally-identical-but-numerically-distinct forward simulators (a
   closed-form 1-compartment-oral analytic solution, standing in for the
   classical/nlmixr2 simulator path, vs. a fixed-step RK4 ODE integration
   of the same governing equations, standing in for a NODE-style forward
   pass) against the same observed `a1_1cmt_oral_linear.csv` data. With
   numerically identical population parameters, the resulting composite
   score gap between the two paradigm labels was **within the documented
   tolerance** (`weighted_sum` score delta ≤ 0.05; Borda-rank distance ≤ 1
   — see the pinning test for exact figures). This is evidence *against* a
   trivial numerical-integration-method bias, but it does **not** rule out
   the harder DGP-vs-simulator-family mismatch the Finding describes, since
   both simulators here solve the exact same ODE exactly (RK4 at a fine
   step converges to the closed-form solution) — it is not a test of a
   simulator whose *structural form* differs from the DGP.
3. Downgrading this from "we assume it's fine" to "we tested the narrow
   case and it's fine, the broad case is flagged and unresolved" is itself
   the improvement this ADR delivers — it converts an implicit assumption
   into an auditable, re-testable claim with a named gap.

**Observability change.** `_gate3_cross_paradigm`
(`src/apmode/governance/gates.py`) now emits a `paradigm_metric_spread`
`GateCheckResult` whenever survivors span ≥ 2 backends, reporting the
composite-score spread between the best candidate of each backend
paradigm. This does not gate anything (`passed=True` unconditionally — it
is observability, not a pass/fail check) and does not introduce a new
policy-JSON threshold, since no concrete bias has been observed that would
warrant one; if a future re-evaluation *does* find paradigm-correlated
bias, the threshold for flagging it must land in `policies/*.json` per
repo convention (no bare numeric literals in governance code), not inline.

---

## Re-evaluation trigger

Build a **DGP-blind holdout**: a Suite A/B-style benchmark scenario whose
ground-truth simulator is deliberately of a structural family that no
current backend's simulator matches exactly (e.g. a DGP with a mechanistic
nonlinearity — saturable binding, non-Michaelis-Menten elimination — that
neither the classical closed-form/ODE solver nor the Bräm-style hybrid
NODE architecture (PRD §4.2.4, R6) was designed to represent natively).
Run Gate 3 cross-paradigm ranking over candidates fit to that scenario by
each backend and check whether one paradigm systematically wins on
VPC/NPE/AUC-Cmax independent of true out-of-sample predictive accuracy
(measured via a genuinely held-out test split, per the
`test_data_path`/`fixed_parameter` LORO-CV contract documented in
CLAUDE.md). If a systematic, paradigm-correlated advantage appears there,
this ADR's Decision must be revisited — likely by either (a) adding a
paradigm-neutral correction term, or (b) restricting cross-paradigm Gate 3
ranking to lanes that explicitly accept the residual circularity risk.

This specific scenario does not exist yet. Suite B (`benchmarks/suite_b/`)
already has real-data structural-mismatch anchors (e.g.
`b10_bolus_1cptmm_mismatch`) and controlled perturbations, but those are
real-data anchors without a known ground-truth DGP — they cannot by
themselves substitute for a *synthetic*, DGP-known, deliberately
paradigm-mismatched scenario of the kind this trigger requires. Building
that scenario is out of scope for this ADR and is not currently scheduled
against a specific milestone.
