# SPDX-License-Identifier: GPL-2.0-or-later
"""CLI-layer tests for ``apmode formular {fmt,lint,validate,explain,diff,lower,compat}``.

Formular sharpening plan §4 Phase 1 (P1.9). Mirrors the
``typer.testing.CliRunner`` convention used for the rest of the CLI surface
(``tests/unit/test_cli.py``, ``tests/unit/test_shell_completion.py``) rather
than duplicating it inline — each command is exercised end-to-end through
:data:`apmode.cli.app` (not the sub-app directly) so the registration wiring
in ``cli.py`` is covered too.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from apmode.cli import app

runner = CliRunner()

_FIXTURE_SPEC = """
model {
    metadata: { title = "CLI Formular Fixture", intent = "exploration", version = "1.0" }
    units: { time = h, amount = mg, concentration = mg/L, volume = L }
    absorption: FirstOrder(ka)
    distribution: OneCmt(V)
    elimination: Linear(CL)
    variability: IIV(params=[CL, V], structure=diagonal)
    covariates: { CL <- WT.power(theta=0.75, ref=70) }
    priors: { CL ~ LogNormal(mu=1.386, sigma=0.25) }
    observation: Proportional(sigma_prop=0.1)
    initial: { ka = 1.2, V = 70.0, CL = 5.0 }
}
"""

_FIXTURE_SPEC_NO_COVARIATES = """
model {
    absorption: FirstOrder(ka)
    distribution: OneCmt(V)
    elimination: Linear(CL)
    variability: IIV(params=[CL, V], structure=diagonal)
    observation: Proportional(sigma_prop=0.1)
    initial: { ka = 1.2, V = 70.0, CL = 5.0 }
}
"""

_FIXTURE_SPEC_INVALID = """
model {
    absorption: FirstOrder(ka)
    distribution: OneCmt(V)
    elimination: Linear(CL)
    variability: IIV(params=[CL, V], structure=diagonal)
    observation: Proportional(sigma_prop=0.1)
    initial: { ka = -1.0, V = 70.0, CL = 5.0 }
}
"""

_FRE_DATA_CSV = """NMID,TIME,WT
1,0,70
1,1,70
2,0,80
2,1,80
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


# ---------------------------------------------------------------------------
# fmt
# ---------------------------------------------------------------------------


class TestFmt:
    def test_prints_canonical_order_to_stdout(self, tmp_path: Path) -> None:
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC)
        result = runner.invoke(app, ["formular", "fmt", str(spec_file)])
        assert result.exit_code == 0, result.output
        assert "model {" in result.stdout
        # Canonical order: metadata before units before absorption before ...
        assert result.stdout.index("metadata:") < result.stdout.index("units:")
        assert result.stdout.index("units:") < result.stdout.index("absorption:")
        assert result.stdout.index("absorption:") < result.stdout.index("distribution:")
        assert result.stdout.index("elimination:") < result.stdout.index("variability:")
        assert result.stdout.index("variability:") < result.stdout.index("covariates:")
        assert result.stdout.index("covariates:") < result.stdout.index("priors:")
        assert result.stdout.index("priors:") < result.stdout.index("initial:")
        assert "reorders top-level blocks" in result.stderr

    def test_in_place_writes_file(self, tmp_path: Path) -> None:
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC)
        result = runner.invoke(app, ["formular", "fmt", str(spec_file), "--in-place"])
        assert result.exit_code == 0, result.output
        rewritten = spec_file.read_text()
        assert rewritten.index("metadata:") < rewritten.index("initial:")

    def test_migrate_rewrites_legacy_syntax(self, tmp_path: Path) -> None:
        legacy = """
        model {
            absorption: FirstOrder(ka=1.2)
            distribution: OneCmt(V=70.0)
            elimination: Linear(CL=5.0)
            variability: IIV(params=[CL, V], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
        }
        """
        spec_file = _write(tmp_path, "legacy.pk", legacy)
        result = runner.invoke(app, ["formular", "fmt", str(spec_file), "--migrate"])
        assert result.exit_code == 0, result.output
        assert "FirstOrder(ka)" in result.stdout
        assert "initial:" in result.stdout
        assert "ka = 1.2" in result.stdout
        # The migrated text must compile under the current grammar.
        from apmode.dsl.grammar import compile_dsl

        compile_dsl(result.stdout)

    def test_migrate_warns_on_unmigratable_construct(self, tmp_path: Path) -> None:
        legacy = """
        model {
            absorption: FirstOrder(ka=1.2)
            distribution: OneCmt(V=70.0)
            elimination: Linear(CL=5.0)
            variability: CovariateLink(param=CL, covariate=SEX, form=categorical)
            observation: Proportional(sigma_prop=0.1)
        }
        """
        spec_file = _write(tmp_path, "legacy_categorical.pk", legacy)
        result = runner.invoke(app, ["formular", "fmt", str(spec_file), "--migrate"])
        assert result.exit_code == 1
        normalized = " ".join(result.stderr.split())
        assert "could not auto-migrate" in normalized
        assert "please review manually" in normalized

    def test_migrate_missing_file_exits_1(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["formular", "fmt", str(tmp_path / "nope.pk"), "--migrate"])
        assert result.exit_code == 1
        assert "not found" in result.stderr

    def test_missing_file_exits_1(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["formular", "fmt", str(tmp_path / "nope.pk")])
        assert result.exit_code == 1
        assert "not found" in result.stderr


# ---------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------


class TestLint:
    def test_valid_spec_exits_0(self, tmp_path: Path) -> None:
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC)
        result = runner.invoke(app, ["formular", "lint", str(spec_file)])
        assert result.exit_code == 0, result.output
        assert "No findings" in result.stdout

    def test_invalid_spec_reports_span_anchored_error(self, tmp_path: Path) -> None:
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC_INVALID)
        result = runner.invoke(app, ["formular", "lint", str(spec_file)])
        assert result.exit_code == 1
        assert "FRM-" in result.stdout
        # Source-span line:col anchor is rendered, not "?:?".
        assert "?:?" not in result.stdout


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_all_levels_default(self, tmp_path: Path) -> None:
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC)
        result = runner.invoke(app, ["formular", "validate", str(spec_file)])
        assert result.exit_code == 1
        assert "syntax" in result.stdout
        assert "semantic" in result.stdout
        assert "data_bound" in result.stdout
        assert "backend_bound" in result.stdout
        assert "policy_bound" in result.stdout
        assert "skipped" in result.stdout

    def test_backend_bound_level_with_frem_reports_missing_capability(
        self, tmp_path: Path
    ) -> None:
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC)
        result = runner.invoke(
            app,
            [
                "formular",
                "validate",
                str(spec_file),
                "--level",
                "backend_bound",
                "--backend",
                "frem",
            ],
        )
        assert result.exit_code == 1
        assert "backend_bound" in result.stdout

    def test_composed_levels(self, tmp_path: Path) -> None:
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC)
        result = runner.invoke(
            app,
            [
                "formular",
                "validate",
                str(spec_file),
                "--level",
                "ast",
                "--level",
                "semantic",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "ast" in result.stdout
        assert "semantic" in result.stdout


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


class TestExplain:
    def test_summarizes_module_choices(self, tmp_path: Path) -> None:
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC)
        result = runner.invoke(app, ["formular", "explain", str(spec_file)])
        assert result.exit_code == 0, result.output
        assert "FirstOrder" in result.stdout
        assert "OneCmt" in result.stdout
        assert "Linear" in result.stdout
        assert "WT" in result.stdout
        assert "LogNormal" in result.stdout

    def test_equations_flag_renders_ode_system(self, tmp_path: Path) -> None:
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC)
        result = runner.invoke(app, ["formular", "explain", str(spec_file), "--equations"])
        assert result.exit_code == 0, result.output
        assert "Differential equations:" in result.stdout
        assert "depot" in result.stdout
        assert "centr" in result.stdout

    def test_equations_flag_reports_node_module_as_error(self, tmp_path: Path) -> None:
        spec_file = _write(
            tmp_path,
            "node_spec.pk",
            """
model {
    absorption: FirstOrder(ka)
    distribution: OneCmt(V)
    elimination: NODE_Elimination(dim=2, constraint_template=saturable)
    observation: Proportional(sigma_prop=0.1)
    initial: { ka = 1.2, V = 70.0 }
}
""",
        )
        result = runner.invoke(app, ["formular", "explain", str(spec_file), "--equations"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# signature
# ---------------------------------------------------------------------------


class TestSignature:
    def test_signature_is_one_line_and_pipeable(self, tmp_path: Path) -> None:
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC)
        result = runner.invoke(app, ["formular", "signature", str(spec_file)])
        assert result.exit_code == 0, result.output
        lines = result.stdout.strip().splitlines()
        assert len(lines) == 1
        assert lines[0] == ("FO absorption | 1CMT | Linear CL | IIV(CL,V) diag | Prop error")


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


class TestDiff:
    def test_identical_specs_no_diff_after_canonicalizing(self, tmp_path: Path) -> None:
        spec_a = _write(tmp_path, "a.pk", _FIXTURE_SPEC)
        spec_b = _write(tmp_path, "b.pk", _FIXTURE_SPEC)
        result = runner.invoke(app, ["formular", "diff", str(spec_a), str(spec_b)])
        assert result.exit_code == 0, result.output
        assert "No differences" in result.stdout

    def test_reordered_variability_is_not_a_diff(self, tmp_path: Path) -> None:
        spec_a = _write(tmp_path, "a.pk", _FIXTURE_SPEC)
        reordered = _FIXTURE_SPEC.replace(
            "variability: IIV(params=[CL, V], structure=diagonal)",
            "variability: IIV(params=[V, CL], structure=diagonal)",
        )
        spec_b = _write(tmp_path, "b.pk", reordered)
        result = runner.invoke(app, ["formular", "diff", str(spec_a), str(spec_b)])
        assert result.exit_code == 0, result.output
        assert "No differences" in result.stdout

    def test_real_difference_is_reported(self, tmp_path: Path) -> None:
        spec_a = _write(tmp_path, "a.pk", _FIXTURE_SPEC)
        changed = _FIXTURE_SPEC.replace("ka = 1.2", "ka = 2.4")
        spec_b = _write(tmp_path, "b.pk", changed)
        result = runner.invoke(app, ["formular", "diff", str(spec_a), str(spec_b)])
        assert result.exit_code == 1
        assert "initial" in result.stdout


# ---------------------------------------------------------------------------
# lower
# ---------------------------------------------------------------------------


class TestLower:
    def test_nlmixr2_emits_r_code(self, tmp_path: Path) -> None:
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC_NO_COVARIATES)
        result = runner.invoke(app, ["formular", "lower", str(spec_file), "--backend", "nlmixr2"])
        assert result.exit_code == 0, result.output
        assert "ini(" in result.stdout
        assert "model(" in result.stdout

    def test_nlmixr2_writes_out_file(self, tmp_path: Path) -> None:
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC_NO_COVARIATES)
        out_file = tmp_path / "model.R"
        result = runner.invoke(
            app,
            [
                "formular",
                "lower",
                str(spec_file),
                "--backend",
                "nlmixr2",
                "--out",
                str(out_file),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()
        assert "ini(" in out_file.read_text()

    def test_stan_emits_program(self, tmp_path: Path) -> None:
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC_NO_COVARIATES)
        result = runner.invoke(app, ["formular", "lower", str(spec_file), "--backend", "stan"])
        assert result.exit_code == 0, result.output
        assert "parameters" in result.stdout

    def test_frem_fails_fast_on_unsupported_capability(self, tmp_path: Path) -> None:
        # _FIXTURE_SPEC declares covariates: {} -- VARIABILITY_COVARIATE_LINK
        # is EXPLICITLY_UNSUPPORTED for frem, so this must fail the
        # capability pre-flight rather than attempt emission.
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC)
        result = runner.invoke(app, ["formular", "lower", str(spec_file), "--backend", "frem"])
        assert result.exit_code == 1
        assert "Cannot lower to" in result.stderr
        assert "backend_bound" in result.stdout
        assert "covariate" in result.stdout.lower()

    def test_semantically_invalid_spec_fails_before_emission(self, tmp_path: Path) -> None:
        # Negative CL fails FRM-SEM-001 (positivity) -- must be caught by
        # the same ast/semantic/lane_bound checks `formular lint` runs,
        # not silently lowered into broken backend code with exit 0.
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC_INVALID)
        result = runner.invoke(app, ["formular", "lower", str(spec_file), "--backend", "nlmixr2"])
        assert result.exit_code == 1
        assert "FRM-" in result.stdout

    def test_frem_without_data_flags_fails_fast(self, tmp_path: Path) -> None:
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC_NO_COVARIATES)
        result = runner.invoke(app, ["formular", "lower", str(spec_file), "--backend", "frem"])
        assert result.exit_code == 1
        assert "--data" in result.stderr
        assert "--frem-covariates" in result.stderr

    def test_frem_with_data_emits_joint_omega_model(self, tmp_path: Path) -> None:
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC_NO_COVARIATES)
        data_file = _write(tmp_path, "data.csv", _FRE_DATA_CSV)
        result = runner.invoke(
            app,
            [
                "formular",
                "lower",
                str(spec_file),
                "--backend",
                "frem",
                "--data",
                str(data_file),
                "--frem-covariates",
                "WT",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "FREM" in result.stdout
        assert "WT" in result.stdout


# ---------------------------------------------------------------------------
# compat
# ---------------------------------------------------------------------------


class TestCompat:
    def test_full_matrix_without_spec(self) -> None:
        result = runner.invoke(app, ["formular", "compat"])
        assert result.exit_code == 0, result.output
        assert "nlmixr2" in result.stdout
        assert "capability matrix" in result.stdout.lower()

    def test_scoped_to_spec(self, tmp_path: Path) -> None:
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC_NO_COVARIATES)
        result = runner.invoke(app, ["formular", "compat", str(spec_file)])
        assert result.exit_code == 0, result.output
        assert "absorption.first_order" in result.stdout

    def test_scoped_to_spec_and_backend(self, tmp_path: Path) -> None:
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC_NO_COVARIATES)
        result = runner.invoke(app, ["formular", "compat", str(spec_file), "--backend", "nlmixr2"])
        assert result.exit_code == 0, result.output
        assert "nlmixr2" in result.stdout
        assert "stan" not in result.stdout.lower()

    def test_unknown_backend_exits_1(self, tmp_path: Path) -> None:
        spec_file = _write(tmp_path, "spec.pk", _FIXTURE_SPEC_NO_COVARIATES)
        result = runner.invoke(
            app, ["formular", "compat", str(spec_file), "--backend", "nonsense"]
        )
        assert result.exit_code == 1
        assert "unknown --backend" in result.stderr
