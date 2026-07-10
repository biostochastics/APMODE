# ADR 0001 — Review Deferrals

## Related documentation

- [../ARCHITECTURE.md](../ARCHITECTURE.md) — its own Cross-references section already lists the same six deferred items (future-annotations, Pyright, god-modules, FREM goldens, `type: ignore`, module-level Console) by name; this link makes the pointer bidirectional.
- `../DIAGNOSTICS_REFINEMENT_ROADMAP.md` *(internal-only, gitignored)* — §M9's FREM-emitter-golden-test deferral and re-evaluation trigger ("research branch" → "production path") directly bears on this roadmap's FREM-deprecation item A1; the two should be reconciled, not silently divergent.

**Date:** 2026-04-17
**Status:** Accepted
**Context:** Post-review register of items intentionally NOT fixed in the
April 2026 APMODE audit (report at session start). Each item documents the
reason for deferral and the condition under which it should be
re-evaluated.

This ADR exists so future reviewers don't re-file the same finding and
don't waste cycles "fixing" something that was a conscious engineering
choice.

---

## L1 — `from __future__ import annotations` on 70 files

**Finding (deferred).** Every Python source file in `src/apmode/` begins
with `from __future__ import annotations`. On Python 3.14+ this import
is redundant (PEP 649 makes deferred annotation evaluation the default).
A review recommendation was to remove the import across the tree.

**Decision:** Keep the import until `requires-python` is raised to
`>=3.14`. The pyproject range is currently `>=3.12,<3.15`; the import is
still load-bearing on 3.12 and 3.13 for forward references. Removing
it prematurely would break those interpreters. Revisit when 3.12 is
dropped from support.

**Re-evaluation trigger:** pyproject minimum bumps to 3.14.

---

## L3 — Pyright disabled (`pyrightconfig.json typeCheckingMode="off"`)

**Finding (deferred).** The repo has a `pyrightconfig.json` that
disables Pyright's type-checking mode. A review recommendation was to
enable Pyright for faster / stricter checking alongside (or replacing)
mypy.

**Decision:** Continue with `mypy --strict` + `pydantic.mypy` plugin as
the single source of type-truth. Reasons:

1. Pyright and mypy disagree on Pydantic v2 and discriminated-union
   edge cases; running both creates redundant-but-conflicting errors
   that slow contributors down.
2. `mypy --strict` is already enforced in CI and pre-commit and is
   known-clean.
3. Pyright's editor integration can be used informally without being
   authoritative.

**Re-evaluation trigger:** Astral's `ty` (Rust-based type checker)
reaches production readiness, OR Pyright gains first-class support for
Pydantic v2 discriminated unions to parity with mypy's plugin.

---

## M8 — God modules (`profiler.py` 1857 LOC; `orchestrator/__init__.py` 1491 LOC)

**Finding (deferred).** `src/apmode/data/profiler.py` and
`src/apmode/orchestrator/__init__.py` are large, domain-heavy modules.
A proposed decomposition (into
`profiler/{_terminal,_clearance,_absorption,_coverage,_covariates}.py`
and `orchestrator/{config,checkpointing,stages/*}.py`) was sketched in
the session review.

**Decision:** Defer until Phase 3 is feature-complete.

**Rationale:**
- Both modules are currently clean (`mypy --strict` / `ruff` pass) and
  covered by extensive tests (1700+ total).
- Decomposing a live module during active Phase 3 feature work risks
  introducing subtle behavioral changes that are invisible through
  diff review but manifest as test regressions only under specific
  scenarios.
- The modules have low cyclomatic-complexity-per-function; they are
  large because the domain is large, not because they are poorly
  factored.
- Splitting is a pure refactoring and can be landed as a single
  reviewable PR once Phase 3 ships.

**Re-evaluation trigger:** Phase 3 scope (CLAUDE.md §8) complete, OR a
module exceeds 2500 LOC, OR a concrete maintenance pain point arises.

---

## M9 — FREM-emitter golden tests

**Finding (deferred).** Stan and FREM emitters lack syrupy golden
snapshot tests (nlmixr2 has them). A review recommendation was to add
golden tests for both. Stan goldens are scheduled as a follow-up to C1
(IVBolus fix).

**Decision:** Add Stan goldens in the C1 follow-up PR; defer FREM
goldens indefinitely.

**Rationale:** PRD R6 marks FREM as Phase-2-with-limited-scope; the
FREM emitter is a research branch rather than a stable shipping
target. Adding snapshot tests now would lock in output that will
change as FREM scope is refined. Stan emitter (Phase 2+) is a
different story — its output surface is frozen for v0.3 and warrants
snapshots.

**Re-evaluation trigger:** FREM scope promoted from "research branch"
to "production path" (see PRD §4.2.4).

---

## L2 — `# type: ignore` audit (33 occurrences)

**Finding (deferred).** 33 `# type: ignore` comments across
`src/apmode/`. A review recommendation was to audit each and remove
those that can be replaced by `cast()` or proper narrowing.

**Decision:** Not a high-value sweep. Most occurrences are justified
(Lark `Transformer` is untyped in the generic case; pandera's
`@check` decorator is untyped; a handful of Pydantic-runtime patterns
unavoidably escape mypy's reach). The ones that are genuinely
removable will surface when their file is edited for an unrelated
reason.

**Re-evaluation trigger:** Lark or pandera ships usable type stubs, OR
the count exceeds 50.

---

## L6 — Module-level `Console()` instances in `cli.py`

**Finding (deferred).** `cli.py` creates `Console()` and
`Console(stderr=True)` at module scope. A review recommendation was to
lazy-initialize via a cached helper.

**Decision:** Not worth the indirection. `cli.py` is the CLI entry
point — if it's imported, the user is running the CLI, and a stdout
Console is always needed immediately. Programmatic embedders (tests,
library use) already pay this cost only on first `import apmode.cli`
and it is sub-millisecond. The cache mechanism would add code surface
area for negligible win.

**Re-evaluation trigger:** Measured import-time regression tied to
`Console()` construction, OR evidence of a programmatic use-case that
imports `cli` without needing the terminal.

---

## Status update (2026-07-09)

Re-checked the counts and triggers cited above against current `main`.
None of the original decision text above has been altered; this section
records present-day facts only.

- **L1 (`from __future__ import annotations`):**
  `grep -rl 'from __future__ import annotations' src/apmode/ | wc -l`
  now returns **135** files (up from the ~70 cited at write time — the
  tree has grown substantially since April 2026). The stated
  re-evaluation trigger ("pyproject minimum bumps to 3.14") has **not**
  fired: `pyproject.toml` still declares
  `requires-python = ">=3.12,<3.15"`. Decision stands unchanged.

- **L2 (`# type: ignore` audit):**
  `grep -rn '# type: ignore' src/apmode/ | wc -l` now returns **51**
  (up from 33 at write time). The stated re-evaluation trigger ("the
  count exceeds 50") **has fired** — 51 > 50. Per the ADR's own terms
  this item is now due for re-evaluation: either re-run the audit to
  confirm the new occurrences remain justified (untyped Lark/pandera
  surfaces, Pydantic-runtime escapes), or file a follow-up sweep.

- **M8 (god modules):** `wc -l` reports `src/apmode/data/profiler.py`
  at **1911 LOC** (up from 1857) and
  `src/apmode/orchestrator/__init__.py` at **1902 LOC** (up from
  1491 — a ~28% increase). The stated re-evaluation trigger ("a module
  exceeds 2500 LOC") has **not** fired for either file, but both are
  now materially closer to it — profiler.py at ~76% of the threshold,
  orchestrator/__init__.py also at ~76% after closing a large fraction
  of the remaining gap since April. Decision stands unchanged; worth
  monitoring given the growth rate on `orchestrator/__init__.py` in
  particular.

- **M9 (FREM-emitter golden tests) — citation fix:** the re-evaluation
  trigger above cites "PRD R6" and "PRD §4.2.4" for FREM's
  research-branch-vs-production-path status. Both citations are wrong:
  `docs/PRD_APMODE_v0.3.md` never mentions FREM (`grep -ci frem
  docs/PRD_APMODE_v0.3.md` → 0), and both §4.2.4 ("Hybrid
  Mechanistic-NODE Backend") and R6 ("Hybrid NODE population modeling
  is a constrained approximation") describe the NODE backend, not
  FREM — an unrelated missing-data method
  (`src/apmode/dsl/frem_emitter.py`, Karlsson 2011 PAGE). The
  citations were evidently copy-pasted from NODE-adjacent text. FREM's
  actual production status is documented in `docs/ARCHITECTURE.md`
  ("What is active today" lists "FREM + MI-PMM + MI-missRanger
  missing-data pipelines" as shipped) and `docs/FORMULAR_SEMANTICS.md`
  (the section describing `frem_emitter.py`). Per that evidence the
  re-evaluation trigger
  has plausibly already fired (FREM is a shipped pipeline, not a
  research branch), but the original decision text is left unedited
  per this ADR's own convention — this bullet only fixes the citation
  and flags the trigger for owner re-evaluation.
