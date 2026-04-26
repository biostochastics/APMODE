# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for R subprocess request/response schemas (ARCHITECTURE.md §4.2)."""

import json

import pytest
from pydantic import ValidationError

from apmode.backends.r_schemas import (
    RSessionInfo,
    RSubprocessRequest,
    RSubprocessResponse,
)
from apmode.dsl.ast_models import (
    IIV,
    DSLSpec,
    FirstOrder,
    LinearElim,
    OneCmt,
    Proportional,
)
from apmode.ids import generate_candidate_id, generate_run_id


def _session_info() -> RSessionInfo:
    return RSessionInfo(
        r_version="4.4.1", nlmixr2_version="3.0.0", platform="aarch64-apple-darwin"
    )


def _test_spec() -> DSLSpec:
    return DSLSpec(
        model_id="test_model_id_0000000",
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=70.0),
        elimination=LinearElim(CL=5.0),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
    )


class TestRSubprocessRequest:
    def test_valid(self) -> None:
        req = RSubprocessRequest(
            schema_version="1.0",
            request_id=generate_run_id(),
            run_id=generate_run_id(),
            candidate_id=generate_candidate_id(),
            spec=_test_spec(),
            data_path="/mnt/data/pk.csv",
            seed=42,
            rng_kind="L'Ecuyer-CMRG",
            initial_estimates={"CL": 5.0, "V": 70.0, "ka": 1.0},
            estimation=["saem", "focei"],
        )
        assert req.schema_version == "1.0"
        assert req.seed == 42

    def test_json_roundtrip(self) -> None:
        req = RSubprocessRequest(
            schema_version="1.0",
            request_id="test_req_id_12345678",
            run_id="test_run_id_12345678",
            candidate_id="test_cand_id_1234567",
            spec=_test_spec(),
            data_path="/data/pk.csv",
            seed=123,
            rng_kind="L'Ecuyer-CMRG",
            initial_estimates={"CL": 3.0},
            estimation=["saem"],
        )
        json_str = req.model_dump_json()
        parsed = json.loads(json_str)
        roundtripped = RSubprocessRequest.model_validate(parsed)
        assert roundtripped.request_id == req.request_id

    def test_empty_estimation_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RSubprocessRequest(
                schema_version="1.0",
                request_id="id1",
                run_id="id2",
                candidate_id="id3",
                spec=_test_spec(),
                data_path="/data.csv",
                seed=1,
                rng_kind="L'Ecuyer-CMRG",
                initial_estimates={},
                estimation=[],
            )

    def test_path_traversal_rejected(self) -> None:
        with pytest.raises(ValidationError, match="traversal"):
            RSubprocessRequest(
                schema_version="1.0",
                request_id="id1",
                run_id="id2",
                candidate_id="id3",
                spec=_test_spec(),
                data_path="/tmp/../../etc/passwd",
                seed=1,
                rng_kind="L'Ecuyer-CMRG",
                initial_estimates={},
                estimation=["saem"],
            )

    def test_relative_path_rejected(self) -> None:
        with pytest.raises(ValidationError, match="absolute"):
            RSubprocessRequest(
                schema_version="1.0",
                request_id="id1",
                run_id="id2",
                candidate_id="id3",
                spec=_test_spec(),
                data_path="data/pk.csv",
                seed=1,
                rng_kind="L'Ecuyer-CMRG",
                initial_estimates={},
                estimation=["saem"],
            )

    def test_test_data_path_traversal_rejected(self) -> None:
        """Held-out CSV path is subject to the same no-traversal rule as data_path."""
        with pytest.raises(ValidationError, match="traversal"):
            RSubprocessRequest(
                schema_version="1.0",
                request_id="id1",
                run_id="id2",
                candidate_id="id3",
                spec=_test_spec(),
                data_path="/data/train.csv",
                seed=1,
                rng_kind="L'Ecuyer-CMRG",
                initial_estimates={},
                estimation=["saem"],
                test_data_path="/tmp/../../etc/shadow",
            )

    def test_test_data_path_relative_rejected(self) -> None:
        with pytest.raises(ValidationError, match="absolute"):
            RSubprocessRequest(
                schema_version="1.0",
                request_id="id1",
                run_id="id2",
                candidate_id="id3",
                spec=_test_spec(),
                data_path="/data/train.csv",
                seed=1,
                rng_kind="L'Ecuyer-CMRG",
                initial_estimates={},
                estimation=["saem"],
                test_data_path="relative/test.csv",
            )

    def test_test_data_path_none_is_default(self) -> None:
        """``test_data_path`` defaults to None and ``fixed_parameter`` to False
        so the wire shape is bit-identical to the v0.6 baseline when callers
        do not opt into honest mode.
        """
        req = RSubprocessRequest(
            schema_version="1.0",
            request_id="id1",
            run_id="id2",
            candidate_id="id3",
            spec=_test_spec(),
            data_path="/data/pk.csv",
            seed=1,
            rng_kind="L'Ecuyer-CMRG",
            initial_estimates={},
            estimation=["saem"],
        )
        assert req.test_data_path is None
        assert req.fixed_parameter is False


class TestRSubprocessResponse:
    def test_success(self) -> None:
        resp = RSubprocessResponse(
            schema_version="1.0",
            status="success",
            error_type=None,
            result={"converged": True, "ofv": -1234.5},
            r_session_info=_session_info(),
            random_seed_state=[1, 2, 3, 4, 5],
        )
        assert resp.status == "success"

    def test_error(self) -> None:
        resp = RSubprocessResponse(
            schema_version="1.0",
            status="error",
            error_type="convergence",
            result=None,
            r_session_info=_session_info(),
            random_seed_state=None,
        )
        assert resp.error_type == "convergence"

    def test_invalid_status(self) -> None:
        with pytest.raises(ValidationError):
            RSubprocessResponse(
                schema_version="1.0",
                status="invalid",
                error_type=None,
                result=None,
                r_session_info=_session_info(),
                random_seed_state=None,
            )

    def test_invalid_error_type(self) -> None:
        with pytest.raises(ValidationError):
            RSubprocessResponse(
                schema_version="1.0",
                status="error",
                error_type="invalid_type",
                result=None,
                r_session_info=_session_info(),
                random_seed_state=None,
            )

    def test_success_with_error_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be None"):
            RSubprocessResponse(
                schema_version="1.0",
                status="success",
                error_type="convergence",
                result=None,
                r_session_info=_session_info(),
                random_seed_state=None,
            )

    def test_error_without_error_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="required"):
            RSubprocessResponse(
                schema_version="1.0",
                status="error",
                error_type=None,
                result=None,
                r_session_info=_session_info(),
                random_seed_state=None,
            )


class TestHarnessRRenamesNMID:
    """Pin the NMID->ID rename in `r/harness.R`.

    Removing the rename re-introduces a silent FOCEI failure: nlmixr2/rxode2
    only recognise the column name ``ID`` for subject identity, and on
    canonical APMODE CSVs (which use ``NMID``) FOCEI's outer optimiser
    enters a "Theta reset (ETA drift)" loop that surfaces as a per-fit
    BackendTimeoutError. The rename happens R-side so persisted train/test
    CSVs (and their bundle digest contributions) keep ``NMID`` byte-for-byte.
    """

    def _harness_text(self) -> str:
        from pathlib import Path

        import apmode

        harness = Path(apmode.__file__).parent / "r" / "harness.R"
        assert harness.is_file(), f"harness.R missing at {harness}"
        return harness.read_text()

    def test_normalize_id_helper_present(self) -> None:
        text = self._harness_text()
        assert ".normalize_id_column" in text, (
            "harness.R must define .normalize_id_column to map NMID -> ID "
            "(see eta-drift loop discussion in test docstring)."
        )

    def test_normalize_applied_to_train_data(self) -> None:
        text = self._harness_text()
        assert "data <- .normalize_id_column(data)" in text, (
            "harness.R must apply .normalize_id_column to the training data; "
            "without it nlmixr2 sees no ID column and FOCEI enters the "
            "Theta-reset (ETA drift) loop."
        )

    def test_normalize_applied_to_test_data(self) -> None:
        text = self._harness_text()
        assert "req$.test_data_frame <- .normalize_id_column(" in text, (
            "harness.R must apply .normalize_id_column to the held-out test "
            "data path so rxode2 partitions the held-out events by subject "
            "id rather than treating the file as a single subject."
        )


class TestPredictedSimulations1DCoercion:
    """Defence-in-depth: PredictedSimulationsSubject must accept the 1D
    shape that ``jsonlite::toJSON(..., auto_unbox = TRUE)`` emits for
    single-observation subjects when an upstream caller forgets the
    ``I()`` wrap inside ``r/harness.R::.simulate_posterior_predictive``.

    The bug surfaces on sparse PK fixtures (``pheno_sd``: 155 obs across
    59 subjects -> some subjects have a single observation): the
    runner used to crash with 200 ``Input should be a valid list``
    Pydantic errors per fit, aborting the entire fixture. The R-side
    ``I()`` wrap is the primary fix; this coercion is defence in
    depth so a future R-side regression cannot silently re-break
    sparse-data Phase-1 runs.
    """

    def test_flat_list_of_floats_is_coerced_to_list_of_singletons(self) -> None:
        from apmode.backends.r_schemas import PredictedSimulationsSubject

        # 200 sims at a single observation, emitted by an
        # auto_unbox-ing jsonlite call without the I() wrap.
        flat_sims = [float(i) / 10.0 for i in range(200)]
        subj = PredictedSimulationsSubject(
            subject_id="s1",
            t_observed=[1.5],
            observed_dv=[8.0],
            sims_at_observed=flat_sims,  # type: ignore[arg-type]
        )
        assert len(subj.sims_at_observed) == 200
        assert all(len(row) == 1 for row in subj.sims_at_observed)
        assert subj.sims_at_observed[0] == [0.0]
        assert subj.sims_at_observed[10] == pytest.approx([1.0])

    def test_well_formed_2d_input_is_unchanged(self) -> None:
        from apmode.backends.r_schemas import PredictedSimulationsSubject

        # Multi-obs subject — list[list[float]] verbatim.
        sims = [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]
        subj = PredictedSimulationsSubject(
            subject_id="s1",
            t_observed=[0.5, 1.0, 2.0],
            observed_dv=[5.0, 8.0, 6.0],
            sims_at_observed=sims,
        )
        assert subj.sims_at_observed == sims

    def test_harness_uses_I_wrap_for_sparse_subjects(self) -> None:
        from pathlib import Path

        import apmode

        harness = (Path(apmode.__file__).parent / "r" / "harness.R").read_text()
        # The R-side primary fix: each per-sim row gets ``I()``-wrapped
        # so the n_obs == 1 case keeps its array shape under
        # ``auto_unbox = TRUE``. Pin the literal so a refactor that
        # drops the wrap re-introduces the sparse-data crash.
        assert "I(as.numeric(mat[r, ]))" in harness, (
            "r/harness.R must wrap each per-sim row in I() so jsonlite "
            "preserves length-1 inner arrays (n_obs == 1 subjects). "
            "Without the wrap the runner crashes on sparse PK fixtures "
            "with ``Input should be a valid list`` Pydantic errors."
        )
        # And both observation vectors get the same I() wrap for the
        # symmetric ``t_observed: 1.5`` (vs ``[1.5]``) and
        # ``observed_dv: 24.3`` (vs ``[24.3]``) bug class.
        assert "t_observed = I(t_obs)" in harness, (
            "r/harness.R must wrap t_observed in I() so a single-"
            "observation subject emits a length-1 JSON array, not a "
            "bare scalar that fails Pydantic's list[float] validation."
        )
        assert "observed_dv = I(dv_obs)" in harness, (
            "r/harness.R must wrap observed_dv in I() symmetrically "
            "with t_observed; otherwise sparse-data subjects produce "
            "a JSON scalar instead of a length-1 list."
        )

    def test_scalar_t_observed_is_coerced_to_list(self) -> None:
        from apmode.backends.r_schemas import PredictedSimulationsSubject

        # n_obs == 1 with auto_unbox-without-I() shape: t_observed and
        # observed_dv emit as bare floats; sims_at_observed comes
        # through as an n_sims-long flat list. All three must coerce.
        subj = PredictedSimulationsSubject(
            subject_id="s1",
            t_observed=57.5,  # type: ignore[arg-type]
            observed_dv=24.3,  # type: ignore[arg-type]
            sims_at_observed=[1.0, 2.0, 3.0],  # type: ignore[arg-type]
        )
        assert subj.t_observed == [57.5]
        assert subj.observed_dv == [24.3]
        assert subj.sims_at_observed == [[1.0], [2.0], [3.0]]

    def test_harness_emits_null_for_undefined_gof(self) -> None:
        """Pin the ``.finite_or_null`` policy around CWRES aggregates.

        nlmixr2 fits can be numerically converged with usable
        parameter estimates yet produce all-NaN residuals (Suite-B
        b8_mavoglurant_null_covariates: 5 random null covariates
        contaminate the residual computation). Per ICH M15 §3 and
        Karlsson 2007 (the canonical CWRES diagnostic paper),
        CWRES is a *diagnostic*, not a convergence indicator: a
        silent 0/1 fallback would let degenerate fits pass
        downstream Gate 1 ranking as "perfectly diagnosed".

        The harness emits JSON ``null`` (via R ``NULL``) for any
        non-finite aggregate; ``GOFMetrics`` accepts
        ``Optional[float]`` so the response round-trips cleanly.
        Downstream consumers (Gate 1, ranker, summarizer, report
        renderer) explicitly handle ``None`` as "diagnostic
        unavailable" with fail-closed semantics.
        """
        from pathlib import Path

        import apmode

        harness = (Path(apmode.__file__).parent / "r" / "harness.R").read_text()
        assert ".finite_or_null <- function(x)" in harness, (
            "r/harness.R must define .finite_or_null so non-finite CWRES "
            "aggregates emit JSON null (not a fabricated 0/1)."
        )
        for line in (
            "cwres_mean = .finite_or_null(mean(wres, na.rm = TRUE))",
            "cwres_sd = .finite_or_null(sd(wres, na.rm = TRUE))",
            "outlier_fraction = .finite_or_null(mean(abs(wres) > 4, na.rm = TRUE))",
        ):
            assert line in harness, f"r/harness.R missing .finite_or_null sanitiser at: {line!r}"

    def test_gof_metrics_schema_accepts_none(self) -> None:
        """``GOFMetrics`` must accept ``None`` for cwres_mean / cwres_sd /
        outlier_fraction so the harness's "diagnostic unavailable"
        signal round-trips through Pydantic. Downstream consumers
        (Gate 1, ranker, summarizer) handle ``None`` with explicit
        fail-closed semantics.
        """
        from apmode.bundle.models import GOFMetrics

        gof = GOFMetrics()
        assert gof.cwres_mean is None
        assert gof.cwres_sd is None
        assert gof.outlier_fraction is None
        assert gof.obs_vs_pred_r2 is None

        gof = GOFMetrics(
            cwres_mean=0.05,
            cwres_sd=None,
            outlier_fraction=0.02,
            obs_vs_pred_r2=0.91,
        )
        assert gof.cwres_mean == 0.05
        assert gof.cwres_sd is None

    def test_cwres_npe_proxy_returns_inf_when_unavailable(self) -> None:
        """The cross-paradigm ranker's CWRES-NPE-proxy fallback must
        rank a fit with undefined CWRES *worst* (not best). Without
        this, a fit with cwres_mean=None would silently sort to the
        front under any "lower is better" ordering.
        """
        import math
        from types import SimpleNamespace

        from apmode.bundle.models import (
            BLQHandling,
            DiagnosticBundle,
            GOFMetrics,
            IdentifiabilityFlags,
        )
        from apmode.governance.ranking import compute_cwres_npe_proxy

        diag = DiagnosticBundle(
            gof=GOFMetrics(),  # all None
            identifiability=IdentifiabilityFlags(
                condition_number=10.0,
                profile_likelihood_ci={},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        )
        result = SimpleNamespace(diagnostics=diag)
        assert math.isinf(compute_cwres_npe_proxy(result))  # type: ignore[arg-type]

        # Sanity: a well-defined fit returns a finite value.
        diag2 = DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=0.0, cwres_sd=1.0, outlier_fraction=0.0),
            identifiability=IdentifiabilityFlags(
                condition_number=10.0,
                profile_likelihood_ci={},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        )
        result2 = SimpleNamespace(diagnostics=diag2)
        assert compute_cwres_npe_proxy(result2) == 0.0  # type: ignore[arg-type]

    def test_harness_drops_unmatched_observed_times(self) -> None:
        """Pin the time-match-truncation logic in
        ``.simulate_posterior_predictive``.

        rxode2's ``rxSolve`` sometimes produces fewer simulated rows
        than there are observation rows in the input data — the
        canonical case is trailing ``DV=0`` observations that
        rxode2 truncates after the last "useful" time. The
        previous strict ``nrow(s_rows) == n_obs`` check then
        silently marked every replicate as failed and the helper
        returned NULL.

        The harness now uses the union of TIMES the sim actually
        covers as the per-subject time axis: observations whose
        TIME the sim doesn't cover are dropped from t_observed /
        observed_dv / sims_at_observed for that subject. The
        per-subject helper documents the truncation as a clean
        column drop, not a silent failure.
        """
        from pathlib import Path

        import apmode

        harness = (Path(apmode.__file__).parent / "r" / "harness.R").read_text()
        # Pin the substring so a refactor that removes the
        # truncation re-introduces the silent NPE=None abort.
        assert "sim_times <- unique(as.numeric(subj_sims$time))" in harness, (
            "r/harness.R must build the per-subject time axis from the "
            "simulation's actual TIMES, not the data's, so partial "
            "rxSolve coverage does not silently mark every replicate "
            "as failed."
        )
        assert "keep_times <- as.numeric(subj_data$TIME) %in% sim_times" in harness, (
            "r/harness.R must filter subj_data to TIMES rxSolve covered."
        )
        assert "idx <- match(t_obs, as.numeric(s_rows$time))" in harness, (
            "r/harness.R must positionally match per-sim values back to "
            "observed times via match() — not by row count alone."
        )

    def test_harness_filters_evid_case_insensitively(self) -> None:
        """Pin the case-insensitive ``evid`` filter in
        ``.simulate_posterior_predictive``.

        rxode2's ``rxSolve`` returns the EVID column as ``evid``
        (lowercase) in the modern data-frame mode. The previous
        ``"EVID" %in% names(sim_df)`` check missed the lowercase
        column entirely, so EVID=2 (other-event) rows in datasets
        like Oral_1CPT inflated the per-subject sim row count past
        ``n_obs`` and the strict ``nrow(s_rows) == n_obs`` check
        silently marked every replicate as failed, returning NULL
        for the entire fixture and crashing Phase-1 on the
        ``BackendResult.diagnostics.npe_score is None`` check.
        """
        from pathlib import Path

        import apmode

        harness = (Path(apmode.__file__).parent / "r" / "harness.R").read_text()
        # Pin both halves of the case-insensitive intersect: the
        # canonical lowercase plus the legacy uppercase fallback.
        assert 'intersect(c("evid", "EVID"), names(sim_df))' in harness, (
            "r/harness.R must filter rxSolve output by both 'evid' "
            "(modern rxode2) and 'EVID' (legacy) so EVID=2 'other-event' "
            "rows do not inflate the per-sim row count and silently mark "
            "every replicate as failed."
        )
