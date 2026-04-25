# SPDX-License-Identifier: GPL-2.0-or-later
"""Agentic LLM backend runner (PRD §4.2.6).

Orchestrates the propose → validate → compile → fit → evaluate loop.
Operates exclusively through typed Formular transforms. Capped at 25
iterations per run. All LLM I/O cached in agentic_trace/ for reproducibility.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

from apmode.backends.diagnostic_summarizer import (
    redact_for_llm,
    summarize_diagnostics,
    summarize_for_llm,
    summarize_stability_diagnostics,
    summarize_stability_for_llm,
)
from apmode.backends.prompts.system_v1 import SYSTEM_PROMPT_VERSION, build_system_prompt
from apmode.backends.protocol import Lane
from apmode.backends.transform_parser import parse_llm_response
from apmode.bundle.models import (
    AgenticTraceInput,
    AgenticTraceMeta,
    AgenticTraceOutput,
    BackendResult,
    DataManifest,
    RunLineage,
)
from apmode.dsl.transforms import apply_transform, validate_transform
from apmode.dsl.validator import validate_dsl
from apmode.errors import AgenticExhaustionError
from apmode.ids import generate_candidate_id

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from apmode.backends.llm_client import LLMResponse
    from apmode.backends.protocol import BackendRunner
    from apmode.bundle.models import (
        ImputationStabilityManifest,
        MissingDataDirective,
        NCASubjectDiagnostic,
    )
    from apmode.dsl.ast_models import DSLSpec
    from apmode.governance.policy import Gate3Config

logger = structlog.get_logger(__name__)


_ROLE_MARKER_RE = re.compile(
    r"""
    (?im)
    (?:^|[\r\n])              # start of string OR any newline (CR / LF / CRLF)
    \s*                        # optional indent
    (?:system|user|assistant)  # role marker
    \s*:\s*                    # ``role:`` separator
    """,
    re.VERBOSE,
)


def _sanitize_for_prompt(text: str, max_len: int = 500) -> str:
    """Strip patterns that could manipulate the LLM via injected error text.

    Backend error messages (e.g., from R or nlmixr2) are embedded as user
    content when relaying failures to the LLM. A hostile or unusual error
    message could contain markdown code fences or JSON-like sequences that
    the LLM would interpret as instructions. This helper truncates the
    message and escapes obvious code-fence / system-prompt sequences.

    Compared with the prior version, the role-marker pattern now also
    fires on any in-string ``\\n`` followed by ``role:`` \u2014 not just at
    line-start \u2014 so a single-line error containing
    ``\\n\\nsystem: ignore previous instructions`` is still neutered.
    """
    if not text:
        return ""
    # Remove triple backticks (code fences) that could terminate our own fence
    cleaned = text.replace("```", "\u2063``\u2063`\u2063")
    # Collapse any lines that look like role markers \u2014 including those
    # smuggled in via embedded ``\\n`` inside a single payload string.
    cleaned = _ROLE_MARKER_RE.sub("\n", cleaned)
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + f"\u2026 [truncated, {len(text) - max_len} chars]"
    return cleaned


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol for LLM clients (real or replay)."""

    async def complete(self, iteration_id: str, messages: list[dict[str, str]]) -> LLMResponse: ...


@dataclass(frozen=True)
class AgenticConfig:
    """Configuration for the agentic runner."""

    max_iterations: int = 25
    lane: str = "discovery"
    system_prompt_version: str = SYSTEM_PROMPT_VERSION
    run_id: str | None = None
    parent_run_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_iterations < 1 or self.max_iterations > 25:
            msg = f"max_iterations must be in [1, 25] (PRD §4.2.6), got {self.max_iterations}"
            raise ValueError(msg)
        valid_lanes = {"submission", "discovery", "optimization"}
        if self.lane not in valid_lanes:
            msg = f"lane must be one of {sorted(valid_lanes)}, got '{self.lane}'"
            raise ValueError(msg)


@dataclass
class IterationRecord:
    """Record of a single agentic iteration."""

    iteration: int
    spec_before: str  # model_id
    spec_after: str | None = None  # model_id after transforms
    transforms_proposed: list[str] = field(default_factory=list)
    transforms_rejected: list[str] = field(default_factory=list)
    reasoning: str = ""
    converged: bool = False
    bic: float | None = None
    error: str | None = None
    validation_feedback: list[str] = field(default_factory=list)


class AgenticRunner:
    """Agentic LLM backend implementing the BackendRunner protocol.

    Core loop:
      1. Evaluate current spec via inner_runner
      2. Build diagnostic summary
      3. Send to LLM with system prompt + history
      4. Parse transforms
      5. Validate transforms against spec + lane
      6. Apply transforms to get new spec
      7. Write trace (input, output, meta)
      8. Repeat or stop
    """

    def __init__(
        self,
        inner_runner: BackendRunner,
        llm_client: LLMClientProtocol,
        config: AgenticConfig,
        trace_dir: Path,
    ) -> None:
        self._inner = inner_runner
        self._llm = llm_client
        self._config = config
        self._trace_dir = trace_dir

    @property
    def trace_dir(self) -> Path:
        """Current directory where iteration traces are written."""
        return self._trace_dir

    @staticmethod
    def _bound_raw_output(raw_text: str, max_chars: int = 32_000) -> str:
        """Cap LLM raw output so a runaway response can't bloat the trace.

        ``raw_output`` is the verbatim LLM reply, kept for ``ReplayClient``
        determinism and for human audit of agentic reasoning. We do *not*
        redact the content (input-side ``redact_for_llm`` already gates
        which subject-derived stats reach the LLM); we only bound length.
        Bundles are served through the API behind the static-API-key
        dependency (`apmode.api.routes._build_require_api_key`) so the
        trust boundary for unredacted reasoning is the bundle itself.
        """
        if len(raw_text) <= max_chars:
            return raw_text
        suffix = f"\n… [truncated, {len(raw_text) - max_chars} chars omitted]"
        return raw_text[:max_chars] + suffix

    async def _llm_complete_with_fallback(
        self,
        iter_id: str,
        messages: list[dict[str, str]],
        best_result: BackendResult | None,
        iteration_records: list[IterationRecord],
    ) -> LLMResponse | None:
        """Call ``self._llm.complete`` with terminal-error handling.

        Returns the LLMResponse on success. On a transient or terminal
        provider error returns ``None`` so the iteration loop can break
        cleanly instead of crashing the orchestrator and discarding
        every classical result that came before. Cancellation propagates
        unchanged so the API DELETE path still works.
        """
        try:
            return await self._llm.complete(iter_id, messages)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "agentic_llm_call_failed",
                extra={
                    "iter_id": iter_id,
                    "exception_type": type(exc).__name__,
                    "have_classical_result": best_result is not None,
                    "iterations_done": len(iteration_records),
                },
            )
            return None

    @contextlib.contextmanager
    def with_trace_dir(self, trace_dir: Path) -> Iterator[AgenticRunner]:
        """Temporarily redirect trace output to ``trace_dir``.

        Used by the orchestrator to isolate Mode 1 (refine) and Mode 2
        (independent) trace artifacts into separate subdirectories
        without mutating private attributes from the outside.
        """
        previous = self._trace_dir
        self._trace_dir = trace_dir
        try:
            yield self
        finally:
            self._trace_dir = previous

    async def run(
        self,
        spec: DSLSpec,
        data_manifest: DataManifest,
        initial_estimates: dict[str, float],
        seed: int,
        timeout_seconds: int | None = None,
        *,
        data_path: Path | None = None,
        split_manifest: dict[str, object] | None = None,
        gate3_policy: Gate3Config | None = None,
        nca_diagnostics: list[NCASubjectDiagnostic] | None = None,
        fixed_parameter: bool = False,
        test_data_path: Path | None = None,
        stability_manifest: ImputationStabilityManifest | None = None,
        directive: MissingDataDirective | None = None,
    ) -> BackendResult:
        """Execute the agentic LLM loop.

        Returns the best BackendResult across all iterations.

        When ``directive.llm_pooled_only`` is True and a matching entry
        exists in ``stability_manifest`` for the current candidate, the
        LLM receives pooled/stability diagnostics only — never per-
        imputation results. This is the structural guard against
        imputation cherry-picking (PRD §4.2.1). When either argument is
        absent the runner falls back to the classical per-fit diagnostic
        summary.

        ``gate3_policy`` and ``nca_diagnostics`` are forwarded verbatim to
        the inner runner on every iteration so posterior-predictive
        diagnostics populate on each fit and the LLM sees the same
        cross-paradigm signal Gate 3 will evaluate.

        ``fixed_parameter`` is accepted for BackendRunner-protocol
        conformance but is incompatible with iterative LLM refinement
        (the loop inherently re-fits). Raise rather than silently
        refitting under a flag that promises the opposite.
        """
        if fixed_parameter:
            msg = (
                "AgenticRunner.run cannot honour fixed_parameter=True — the "
                "agentic loop re-fits on every iteration by design. Use a "
                "classical runner for LORO-CV fixed-parameter evaluation."
            )
            raise NotImplementedError(msg)
        _ = test_data_path  # held-out NPE flows via inner runner; agentic loop is full-data
        pooled_only = directive is not None and directive.llm_pooled_only
        stability_by_candidate: dict[str, Any] = (
            {e.candidate_id: e for e in stability_manifest.entries}
            if stability_manifest is not None
            else {}
        )
        self._trace_dir.mkdir(parents=True, exist_ok=True)

        # Single, stable run_id for the entire loop — every iteration trace
        # and the RunLineage artifact share this identifier (PRD §4.2.6).
        run_id = self._config.run_id or generate_candidate_id()

        # Build available transforms based on lane
        available_transforms = [
            "swap_module",
            "add_covariate_link",
            "adjust_variability",
            "set_transit_n",
            "toggle_lag",
        ]
        if self._config.lane in ("discovery", "optimization"):
            available_transforms.append("replace_with_node")

        system_prompt = build_system_prompt(
            lane=self._config.lane,
            available_transforms=available_transforms,
        )

        current_spec = spec
        best_result: BackendResult | None = None
        history: list[dict[str, Any]] = []
        iteration_records: list[IterationRecord] = []
        lineage_entries: list[dict[str, str | None]] = []

        # Conversation history preserves multi-turn context across iterations
        # so the LLM knows what it tried before and what happened. A sliding
        # window is applied at message-construction time to prevent
        # unbounded token growth over 25 iterations (full history still
        # captured in trace files).
        conversation_history: list[dict[str, str]] = []
        # Keep the system prompt + last N*2 messages (N iterations worth of
        # user + assistant pairs). 12 iterations x 2 = 24 messages fits
        # comfortably within 128K-token context windows even with verbose
        # diagnostics and multi-turn validation feedback.
        max_history_messages = 24

        # ``_finalise_trace`` runs in *every* exit path (normal,
        # CancelledError, unhandled exception) so the three summary
        # artifacts (``agentic_iterations.jsonl``,
        # ``agentic_lineage.json``, ``run_lineage.json``) capture the
        # iterations that completed before the loop exited. Without
        # this, a DELETE /runs/{id} cancellation loses the rollup of
        # every iteration that completed before the cancel.
        def _finalise_trace() -> None:
            try:
                if iteration_records:
                    self._write_iteration_records(iteration_records)
                if lineage_entries:
                    self._write_agentic_lineage(lineage_entries)
                lineage = RunLineage(
                    current_run_id=run_id,
                    parent_run_ids=list(self._config.parent_run_ids),
                    lineage_type=(
                        "continuation" if self._config.parent_run_ids else "independent"
                    ),
                )
                (self._trace_dir / "run_lineage.json").write_text(
                    lineage.model_dump_json(indent=2)
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("agentic_trace_finalise_failed")

        try:
            for iteration in range(1, self._config.max_iterations + 1):
                iter_id = f"iter_{iteration:03d}"
                record = IterationRecord(iteration=iteration, spec_before=current_spec.model_id)

                # 1. Evaluate current spec
                runner_error: str | None = None
                result: BackendResult | None = None
                try:
                    result = await self._inner.run(
                        spec=current_spec,
                        data_manifest=data_manifest,
                        initial_estimates=initial_estimates,
                        seed=seed,
                        timeout_seconds=timeout_seconds,
                        data_path=data_path,
                        split_manifest=split_manifest,
                        gate3_policy=gate3_policy,
                        nca_diagnostics=nca_diagnostics,
                    )
                except Exception as e:
                    logger.warning("Iteration %d: inner runner failed: %s", iteration, e)
                    runner_error = str(e)
                    record.error = runner_error

                # If runner failed, relay error to LLM so it can propose a fix
                if result is None:
                    safe_err = _sanitize_for_prompt(runner_error or "unknown")
                    error_msg = (
                        f"## Iteration {iteration}/{self._config.max_iterations}\n\n"
                        f"**Backend execution failed:** {safe_err}\n\n"
                        f"The current model spec could not be evaluated. "
                        f"Please propose transforms to address this failure, "
                        f"or signal stop if no recovery is possible."
                    )
                    conversation_history.append({"role": "user", "content": error_msg})
                    trimmed = conversation_history[-max_history_messages:]
                    messages = [
                        {"role": "system", "content": system_prompt},
                        *trimmed,
                    ]
                    prompt_hash = hashlib.sha256(
                        json.dumps(messages, sort_keys=True).encode()
                    ).hexdigest()

                    trace_input = AgenticTraceInput(
                        iteration_id=iter_id,
                        run_id=run_id,
                        candidate_id=current_spec.model_id,
                        prompt_hash=prompt_hash,
                        prompt_template=self._config.system_prompt_version,
                        dsl_spec_json=current_spec.model_dump_json(),
                        diagnostics_summary={"error": runner_error or "unknown"},
                    )
                    self._write_trace_input(trace_input)

                    llm_response = await self._llm_complete_with_fallback(
                        iter_id, messages, best_result, iteration_records
                    )
                    if llm_response is None:
                        record.error = (record.error or "") + "; llm_call_failed_terminal"
                        iteration_records.append(record)
                        break
                    self._write_cached_response(iter_id, llm_response)
                    # Sanitize before retaining in history. The current-iteration
                    # parser still consumes the verbatim raw_text; history is used
                    # only as context for subsequent LLM calls, where injected
                    # role markers or code fences could manipulate the model.
                    conversation_history.append(
                        {
                            "role": "assistant",
                            "content": _sanitize_for_prompt(llm_response.raw_text, max_len=4000),
                        }
                    )

                    parse_result = parse_llm_response(llm_response.raw_text)
                    trace_output = AgenticTraceOutput(
                        iteration_id=iter_id,
                        raw_output=self._bound_raw_output(llm_response.raw_text),
                        parsed_transforms=[str(t) for t in parse_result.transforms],
                        validation_passed=parse_result.success,
                        validation_errors=parse_result.errors,
                    )
                    self._write_trace_output(trace_output)

                    has_det_ver = (
                        llm_response.model_version != ""
                        and llm_response.model_version != llm_response.model_id
                    )
                    trace_meta = AgenticTraceMeta(
                        iteration_id=iter_id,
                        model_id=llm_response.model_id,
                        model_version=llm_response.model_version,
                        prompt_hash=prompt_hash,
                        input_tokens=llm_response.input_tokens,
                        output_tokens=llm_response.output_tokens,
                        cost_usd=llm_response.cost_usd,
                        temperature=0.0,
                        wall_time_seconds=llm_response.wall_time_seconds,
                        request_payload_hash=llm_response.request_payload_hash,
                        agentic_reproducibility="full" if has_det_ver else "best-effort",
                    )
                    self._write_trace_meta(trace_meta)

                    if parse_result.stop:
                        iteration_records.append(record)
                        break

                    # Apply corrective transforms if any, with feedback.
                    # Lineage entries are STAGED in a local list and only
                    # committed to ``lineage_entries`` after ``validate_dsl``
                    # passes — otherwise an invalid post-transform spec
                    # would leave orphan candidate_ids in
                    # ``agentic_lineage.json`` for a candidate that was
                    # never officially adopted.
                    err_feedback: list[str] = []
                    if parse_result.success and parse_result.transforms:
                        new_spec = current_spec
                        staged_lineage: list[dict[str, str | None]] = []
                        for transform in parse_result.transforms:
                            t_errors = validate_transform(new_spec, transform)
                            if t_errors:
                                err_feedback.append(
                                    f"Transform `{transform}` rejected: " + "; ".join(t_errors)
                                )
                            else:
                                try:
                                    prev_id = new_spec.model_id
                                    new_spec = apply_transform(new_spec, transform)
                                    staged_lineage.append(
                                        {
                                            "candidate_id": new_spec.model_id,
                                            "parent_id": prev_id,
                                            "transform": str(transform),
                                        }
                                    )
                                except ValueError as e:
                                    err_feedback.append(
                                        f"Transform `{transform}` apply failed: {e}"
                                    )
                        lane_enum = Lane(self._config.lane)
                        dsl_errors = validate_dsl(new_spec, lane=lane_enum)
                        if dsl_errors:
                            err_feedback.append(
                                "Post-transform DSL validation failed: "
                                + "; ".join(e.message for e in dsl_errors)
                            )
                        else:
                            current_spec = new_spec
                            lineage_entries.extend(staged_lineage)
                    elif not parse_result.success:
                        err_feedback.append(
                            "Response parse failed: " + "; ".join(parse_result.errors)
                        )

                    # Feed validation failures back so the LLM can correct
                    if err_feedback:
                        feedback_msg = (
                            "## Validation Feedback\n\n"
                            + "\n".join(f"- {f}" for f in err_feedback)
                            + "\n\nPlease propose corrected transforms."
                        )
                        conversation_history.append({"role": "user", "content": feedback_msg})

                    history.append(
                        {
                            "model_id": current_spec.model_id,
                            "bic": None,
                            "converged": False,
                            "iteration": iteration,
                            "error": runner_error,
                        }
                    )
                    iteration_records.append(record)
                    continue

                # Track best result
                if result.converged:
                    record.converged = True
                    record.bic = result.bic
                    if best_result is None or (
                        result.bic is not None
                        and (best_result.bic is None or result.bic < best_result.bic)
                    ):
                        best_result = result

                # Record for search history
                history.append(
                    {
                        "model_id": current_spec.model_id,
                        "bic": result.bic,
                        "converged": result.converged,
                        "iteration": iteration,
                    }
                )

                # 2. Build diagnostic summary for LLM.
                # When the missing-data directive requires pooled-only inputs and
                # a stability entry exists for the current candidate, substitute
                # the pooled/stability summary. Otherwise fall back to the
                # classical per-fit summary.
                stability_entry = stability_by_candidate.get(current_spec.model_id)
                if pooled_only and stability_entry is not None and stability_manifest is not None:
                    diag_text = summarize_stability_for_llm(
                        stability_entry,
                        stability_manifest,
                        iteration=iteration,
                        max_iterations=self._config.max_iterations,
                        search_history=history,
                    )
                    diag_summary = redact_for_llm(
                        summarize_stability_diagnostics(stability_entry, stability_manifest)
                    )
                else:
                    diag_text = summarize_for_llm(
                        result,
                        iteration=iteration,
                        max_iterations=self._config.max_iterations,
                        search_history=history,
                    )
                    # Redaction gate: enforce allow-list before any data leaves the
                    # process to the LLM provider (PRD §10, ARCHITECTURE.md §11).
                    diag_summary = redact_for_llm(summarize_diagnostics(result))

                # 3. Build messages with conversation history (sliding window)
                conversation_history.append({"role": "user", "content": diag_text})
                trimmed = conversation_history[-max_history_messages:]
                messages = [
                    {"role": "system", "content": system_prompt},
                    *trimmed,
                ]

                # 4. Write trace input
                prompt_hash = hashlib.sha256(
                    json.dumps(messages, sort_keys=True).encode()
                ).hexdigest()

                trace_input = AgenticTraceInput(
                    iteration_id=iter_id,
                    run_id=run_id,
                    candidate_id=current_spec.model_id,
                    prompt_hash=prompt_hash,
                    prompt_template=self._config.system_prompt_version,
                    dsl_spec_json=current_spec.model_dump_json(),
                    diagnostics_summary={
                        # str(True) → "True" instead of "1"; isinstance(True, int)
                        # is True so bool must be handled before int in a tuple check.
                        k: str(v)
                        for k, v in diag_summary.items()
                        if isinstance(v, bool | str | int | float)
                    },
                )
                self._write_trace_input(trace_input)

                # 5. Call LLM. ``_llm_complete_with_fallback`` returns
                # ``None`` on a terminal provider error so the loop can
                # break and the orchestrator still gets the best
                # classical result so far instead of crashing.
                llm_response = await self._llm_complete_with_fallback(
                    iter_id, messages, best_result, iteration_records
                )
                if llm_response is None:
                    record.error = (record.error or "") + "; llm_call_failed_terminal"
                    iteration_records.append(record)
                    break

                # 5a. Write cached response for ReplayClient deterministic replay
                self._write_cached_response(iter_id, llm_response)
                # Sanitize before retaining in history (see duplicate above) —
                # protects subsequent iterations from injected role markers /
                # code fences in the current LLM output.
                conversation_history.append(
                    {
                        "role": "assistant",
                        "content": _sanitize_for_prompt(llm_response.raw_text, max_len=4000),
                    }
                )

                # 6. Write trace output + meta
                parse_result = parse_llm_response(llm_response.raw_text)

                trace_output = AgenticTraceOutput(
                    iteration_id=iter_id,
                    raw_output=self._bound_raw_output(llm_response.raw_text),
                    parsed_transforms=[str(t) for t in parse_result.transforms],
                    validation_passed=parse_result.success,
                    validation_errors=parse_result.errors,
                )
                self._write_trace_output(trace_output)

                # Model-version escrow (PRD §4.2.6): if model_version equals
                # model_id (no deterministic fingerprint), flag as best-effort
                has_deterministic_version = (
                    llm_response.model_version != ""
                    and llm_response.model_version != llm_response.model_id
                )
                reproducibility = "full" if has_deterministic_version else "best-effort"

                trace_meta = AgenticTraceMeta(
                    iteration_id=iter_id,
                    model_id=llm_response.model_id,
                    model_version=llm_response.model_version,
                    prompt_hash=prompt_hash,
                    input_tokens=llm_response.input_tokens,
                    output_tokens=llm_response.output_tokens,
                    cost_usd=llm_response.cost_usd,
                    temperature=0.0,
                    wall_time_seconds=llm_response.wall_time_seconds,
                    request_payload_hash=llm_response.request_payload_hash,
                    agentic_reproducibility=reproducibility,
                )
                self._write_trace_meta(trace_meta)

                # 7. Check stop signal
                if parse_result.stop:
                    record.reasoning = parse_result.reasoning
                    iteration_records.append(record)
                    logger.info(
                        "Iteration %d: LLM signaled stop — %s",
                        iteration,
                        parse_result.reasoning,
                    )
                    break

                # 8. Parse failure → feed back to LLM
                if not parse_result.success:
                    record.error = f"Parse failure: {'; '.join(parse_result.errors)}"
                    record.reasoning = parse_result.reasoning
                    iteration_records.append(record)
                    logger.warning(
                        "Iteration %d: parse failed — %s", iteration, parse_result.errors
                    )
                    conversation_history.append(
                        {
                            "role": "user",
                            "content": (
                                "## Validation Feedback\n\n"
                                f"Your response could not be parsed: "
                                f"{'; '.join(parse_result.errors)}\n\n"
                                "Please respond with valid JSON matching the schema."
                            ),
                        }
                    )
                    continue

                # 9. Apply transforms sequentially, collecting validation
                # feedback. Lineage entries are STAGED locally and only
                # committed to ``lineage_entries`` after ``validate_dsl``
                # accepts the post-transform spec — otherwise an invalid
                # spec would seed orphan candidate_ids in the lineage.
                new_spec = current_spec
                applied_transforms: list[str] = []
                validation_feedback: list[str] = []
                staged_lineage = []  # reset per-iteration; type re-uses outer annotation

                for transform in parse_result.transforms:
                    # Validate transform against current spec
                    t_errors = validate_transform(new_spec, transform)
                    if t_errors:
                        logger.warning(
                            "Iteration %d: transform validation failed: %s",
                            iteration,
                            t_errors,
                        )
                        validation_feedback.append(
                            f"Transform `{transform}` rejected: " + "; ".join(t_errors)
                        )
                        continue

                    # Apply transform
                    try:
                        prev_id = new_spec.model_id
                        new_spec = apply_transform(new_spec, transform)
                        applied_transforms.append(str(transform))
                        staged_lineage.append(
                            {
                                "candidate_id": new_spec.model_id,
                                "parent_id": prev_id,
                                "transform": str(transform),
                            }
                        )
                    except ValueError as e:
                        logger.warning("Iteration %d: transform apply failed: %s", iteration, e)
                        validation_feedback.append(f"Transform `{transform}` apply failed: {e}")
                        continue

                # 10. Validate new spec against lane
                lane_enum = Lane(self._config.lane)
                dsl_errors = validate_dsl(new_spec, lane=lane_enum)
                if dsl_errors:
                    logger.warning(
                        "Iteration %d: new spec failed DSL validation: %s",
                        iteration,
                        [e.message for e in dsl_errors],
                    )
                    record.error = f"DSL validation: {[e.message for e in dsl_errors]}"
                    record.validation_feedback = [
                        *validation_feedback,
                        "Post-transform DSL validation failed: "
                        + "; ".join(e.message for e in dsl_errors),
                    ]
                    record.transforms_rejected = [
                        str(t) for t in parse_result.transforms if str(t) not in applied_transforms
                    ]
                    iteration_records.append(record)
                    # Feed validation failures back to LLM for next iteration
                    conversation_history.append(
                        {
                            "role": "user",
                            "content": (
                                "## Validation Feedback\n\n"
                                + "\n".join(f"- {f}" for f in validation_feedback)
                                + "\n\nPlease propose corrected transforms."
                            ),
                        }
                    )
                    continue

                # validate_dsl accepted the post-transform spec — promote
                # the staged lineage entries (gemini #4: avoids orphan
                # candidate_ids in agentic_lineage.json on transform-side
                # rejection paths above).
                lineage_entries.extend(staged_lineage)

                # Feed partial validation feedback if some transforms were rejected
                if validation_feedback:
                    conversation_history.append(
                        {
                            "role": "user",
                            "content": (
                                "## Partial Validation Feedback\n\n"
                                "Some transforms were applied but others were rejected:\n"
                                + "\n".join(f"- {f}" for f in validation_feedback)
                            ),
                        }
                    )

                record.spec_after = new_spec.model_id
                all_proposed = [str(t) for t in parse_result.transforms]
                record.transforms_proposed = all_proposed
                applied_set = set(applied_transforms)
                record.transforms_rejected = [t for t in all_proposed if t not in applied_set]
                record.reasoning = parse_result.reasoning
                record.validation_feedback = validation_feedback
                iteration_records.append(record)

                # Use fitted params as warm-start for next iteration
                # #35: non-finite estimates (NaN / Inf) from a brittle fit
                # would poison the next iteration's starting values and
                # cascade into a run of useless candidates. Drop them and
                # fall back to the incoming defaults for those parameters;
                # downstream validation will reject the iteration if the
                # fallback is also inadequate.
                if result.converged:
                    from math import isfinite

                    warm: dict[str, float] = {}
                    for name, pe in result.parameter_estimates.items():
                        if pe.category != "structural":
                            continue
                        value = float(pe.estimate)
                        if isfinite(value):
                            warm[name] = value
                        else:
                            logger.warning(
                                "agentic_warm_start_skipped_non_finite",
                                extra={
                                    "iteration": iteration,
                                    "param": name,
                                    "value": repr(pe.estimate),
                                },
                            )
                    initial_estimates = warm or initial_estimates

                current_spec = new_spec
        finally:
            # Always flush the audit-trail rollup, including on
            # CancelledError or unexpected exception. The per-iteration
            # trace files (input/output/meta) are written inside the
            # loop, but the iteration_records / lineage / run_lineage
            # summaries are only constructible from the in-memory state
            # we accumulated above.
            _finalise_trace()
            # Close the LLM provider's underlying httpx pool so a
            # long-running APMODE process does not accumulate leaked
            # sockets across successive agentic runs. Best-effort:
            # ``aclose`` is optional on the protocol (ReplayClient and
            # the litellm fallback don't implement it) and a transient
            # close failure must not mask the loop's own outcome.
            aclose = getattr(self._llm, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # pragma: no cover - defensive
                    logger.warning("agentic_llm_client_close_failed", exc_info=True)

        # Return best result, falling back to last result
        if best_result is None:
            msg = "Agentic runner: no converged results across all iterations"
            raise AgenticExhaustionError(msg, iterations=len(iteration_records))

        # Stamp the result as agentic_llm backend.
        # ``model_copy(update=...)`` preserves every field on the source
        # ``BackendResult`` — including backend-specific extensions like
        # ``posterior_diagnostics`` / ``sampler_config`` from the
        # Bayesian runner — that would be silently dropped if we
        # rebuilt the model field-by-field.
        return best_result.model_copy(update={"backend": "agentic_llm"})

    def _write_trace_input(self, inp: AgenticTraceInput) -> None:
        path = self._trace_dir / f"{inp.iteration_id}_input.json"
        path.write_text(inp.model_dump_json(indent=2))

    def _write_trace_output(self, out: AgenticTraceOutput) -> None:
        path = self._trace_dir / f"{out.iteration_id}_output.json"
        path.write_text(out.model_dump_json(indent=2))

    def _write_trace_meta(self, meta: AgenticTraceMeta) -> None:
        path = self._trace_dir / f"{meta.iteration_id}_meta.json"
        path.write_text(meta.model_dump_json(indent=2))

    def _write_cached_response(self, iteration_id: str, llm_response: LLMResponse) -> None:
        """Write cached_response.json for deterministic replay via ReplayClient."""
        path = self._trace_dir / f"{iteration_id}_cached_response.json"
        path.write_text(llm_response.model_dump_json(indent=2))

    def _write_agentic_lineage(self, entries: list[dict[str, str | None]]) -> None:
        """Write agentic_lineage.json — candidate derivation DAG from transforms."""
        path = self._trace_dir / "agentic_lineage.json"
        path.write_text(json.dumps({"entries": entries}, indent=2))

    def _write_iteration_records(self, records: list[IterationRecord]) -> None:
        """Write agentic_iterations.jsonl — complete audit trail of reasoning.

        Flush and fsync each line so the audit trail survives abrupt
        termination; otherwise a mid-run crash loses in-flight records
        under the default buffered-write behaviour.
        """
        path = self._trace_dir / "agentic_iterations.jsonl"
        with path.open("w") as f:
            for rec in records:
                entry = {
                    "iteration": rec.iteration,
                    "spec_before": rec.spec_before,
                    "spec_after": rec.spec_after,
                    "transforms_proposed": rec.transforms_proposed,
                    "transforms_rejected": rec.transforms_rejected,
                    "reasoning": rec.reasoning,
                    "converged": rec.converged,
                    "bic": rec.bic,
                    "error": rec.error,
                    "validation_feedback": rec.validation_feedback,
                }
                f.write(json.dumps(entry) + "\n")
                f.flush()
                import os as _os_local

                _os_local.fsync(f.fileno())
