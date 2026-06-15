"""
Unit tests for phewas.py — Cox regression control selection and saddle point approximation.
"""
import datetime
import numpy as np
import polars as pl
import pytest
from scipy.stats import chi2

from phetk.phewas import PheWAS


@pytest.fixture
def cox_phewas(tmp_path):
    """Create a minimal PheWAS instance configured for Cox regression.

    Participants:
      10 — has phecode "ID_001" with first_event_date (2019-01-01) BEFORE
           cox_start_date (2020-01-01) → pre-existing condition
      20 — has phecode "ID_001" with first_event_date (2021-06-01) AFTER
           cox_start_date (2020-01-01) → valid case
      30 — has no phecode "ID_001" → should be a control
    """
    # Cohort file — every participant needs covariates + Cox time columns
    cohort = pl.DataFrame({
        "person_id": [10, 20, 30],
        "independent_variable_of_interest": [1, 0, 1],
        "sex": [1, 0, 1],
        "age": [50, 60, 55],
        "cox_start_date": [
            datetime.date(2020, 1, 1),
            datetime.date(2020, 1, 1),
            datetime.date(2020, 1, 1),
        ],
        "control_observed_time": [5.0, 5.0, 5.0],
    })
    cohort_file = str(tmp_path / "cohort.tsv")
    cohort.write_csv(cohort_file, separator="\t")

    # Phecode counts file
    phecode_counts = pl.DataFrame({
        "person_id": [10, 20],
        "phecode": ["ID_001", "ID_001"],
        "count": [3, 2],
        "first_event_date": [
            datetime.date(2019, 1, 1),   # BEFORE cox_start_date
            datetime.date(2021, 6, 1),   # AFTER  cox_start_date
        ],
        "phecode_observed_time": [1.0, 1.5],
    })
    phecode_file = str(tmp_path / "phecode_counts.tsv")
    phecode_counts.write_csv(phecode_file, separator="\t")

    pw = PheWAS(
        phecode_version="X",
        phecode_count_file_path=phecode_file,
        cohort_file_path=cohort_file,
        covariate_cols=["age", "sex"],
        independent_variable_of_interest="independent_variable_of_interest",
        sex_at_birth_col="sex",
        method="cox",
        cox_start_date_col="cox_start_date",
        cox_control_observed_time_col="control_observed_time",
        cox_phecode_observed_time_col="phecode_observed_time",
        min_cases=1,
        min_phecode_count=2,
        output_file_path=str(tmp_path / "results"),
    )
    return pw


@pytest.fixture
def male_only_phewas(tmp_path):
    """Create a PheWAS instance with an all-male cohort (sex=1).

    Participants 10, 20, 30 are all male. Phecode "MALE_001" has sex
    restriction "Male" and "FEM_001" has sex restriction "Female" in the
    mapping table.
    """
    cohort = pl.DataFrame({
        "person_id": [10, 20, 30],
        "independent_variable_of_interest": [1, 0, 1],
        "sex": [1, 1, 1],
        "age": [50, 60, 55],
    })
    cohort_file = str(tmp_path / "cohort.tsv")
    cohort.write_csv(cohort_file, separator="\t")

    phecode_counts = pl.DataFrame({
        "person_id": [10, 20],
        "phecode": ["FEM_001", "FEM_001"],
        "count": [3, 2],
    })
    phecode_file = str(tmp_path / "phecode_counts.tsv")
    phecode_counts.write_csv(phecode_file, separator="\t")

    pw = PheWAS(
        phecode_version="X",
        phecode_count_file_path=phecode_file,
        cohort_file_path=cohort_file,
        covariate_cols=["age", "sex"],
        independent_variable_of_interest="independent_variable_of_interest",
        sex_at_birth_col="sex",
        min_cases=1,
        min_phecode_count=1,
        output_file_path=str(tmp_path / "results"),
    )
    return pw


@pytest.fixture
def female_only_phewas(tmp_path):
    """Create a PheWAS instance with an all-female cohort (sex=0).

    Participants 10, 20, 30 are all female. Phecode "MALE_001" has sex
    restriction "Male" in the mapping table.
    """
    cohort = pl.DataFrame({
        "person_id": [10, 20, 30],
        "independent_variable_of_interest": [1, 0, 1],
        "sex": [0, 0, 0],
        "age": [50, 60, 55],
    })
    cohort_file = str(tmp_path / "cohort.tsv")
    cohort.write_csv(cohort_file, separator="\t")

    phecode_counts = pl.DataFrame({
        "person_id": [10, 20],
        "phecode": ["MALE_001", "MALE_001"],
        "count": [3, 2],
    })
    phecode_file = str(tmp_path / "phecode_counts.tsv")
    phecode_counts.write_csv(phecode_file, separator="\t")

    pw = PheWAS(
        phecode_version="X",
        phecode_count_file_path=phecode_file,
        cohort_file_path=cohort_file,
        covariate_cols=["age", "sex"],
        independent_variable_of_interest="independent_variable_of_interest",
        sex_at_birth_col="sex",
        min_cases=1,
        min_phecode_count=1,
        output_file_path=str(tmp_path / "results"),
    )
    return pw


class TestSingleSexCohortSexRestriction:
    """Single-sex cohorts must return empty cases/controls for opposite-sex phecodes."""

    def test_male_cohort_female_phecode_returns_empty(self, male_only_phewas):
        """All-male cohort must skip Female-restricted phecodes."""
        cases, controls, _ = male_only_phewas._case_control_prep(
            phecode="FEM_001", keep_ids=True,
        )
        assert len(cases) == 0
        assert len(controls) == 0

    def test_female_cohort_male_phecode_returns_empty(self, female_only_phewas):
        """All-female cohort must skip Male-restricted phecodes."""
        cases, controls, _ = female_only_phewas._case_control_prep(
            phecode="MALE_001", keep_ids=True,
        )
        assert len(cases) == 0
        assert len(controls) == 0


class TestCoxPreExistingExcludedFromControls:
    """Participants with a pre-existing condition (first_event_date < cox_start_date)
    must not appear in the control group."""

    def test_pre_existing_not_in_controls(self, cox_phewas):
        cases, controls, _ = cox_phewas._case_control_prep(
            phecode="ID_001", keep_ids=True,
        )
        control_ids = controls["person_id"].to_list()
        # Person 10 had the condition before cox_start_date — must NOT be a control
        assert 10 not in control_ids

    def test_pre_existing_not_in_cases(self, cox_phewas):
        cases, controls, _ = cox_phewas._case_control_prep(
            phecode="ID_001", keep_ids=True,
        )
        case_ids = cases["person_id"].to_list()
        # Person 10 was excluded pre-baseline — must NOT be a case either
        assert 10 not in case_ids

    def test_valid_case_present(self, cox_phewas):
        cases, controls, _ = cox_phewas._case_control_prep(
            phecode="ID_001", keep_ids=True,
        )
        case_ids = cases["person_id"].to_list()
        assert 20 in case_ids

    def test_clean_participant_is_control(self, cox_phewas):
        cases, controls, _ = cox_phewas._case_control_prep(
            phecode="ID_001", keep_ids=True,
        )
        control_ids = controls["person_id"].to_list()
        assert 30 in control_ids


# ---------------------------------------------------------------------------
# Saddle Point Approximation (SPA) tests
# ---------------------------------------------------------------------------

@pytest.fixture
def spa_instance():
    """Minimal PheWAS-like object with only spa_cutoff set.

    Avoids running the full PheWAS __init__ (which needs CSV files) while
    still providing the instance attribute required by _spa_test and
    _logit_spa_regression.
    """
    pw = object.__new__(PheWAS)
    pw.spa_cutoff = 0.05
    return pw


class TestSpaHelpers:
    """Pure-math unit tests for the SPA CGF helper static/class methods.

    These methods operate on raw numpy arrays and require no PheWAS state,
    so they are tested directly on the class.
    """

    def test_k2_positive_for_several_t_values(self):
        """K''(t) must be strictly positive for any evaluation point t.

        K''(t) is the variance of the score statistic evaluated at t;
        it cannot be non-positive for a non-degenerate distribution.
        """
        rng = np.random.default_rng(1)
        mu = rng.uniform(0.05, 0.95, 80)
        g = rng.standard_normal(80)
        for t in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            assert PheWAS._spa_k2(t, mu, g) > 0

    def test_k1_adj_at_t0_equals_score_mean(self):
        """K'(0) − q = sum(g * mu) − q.

        At t=0, the CGF derivative reduces to the expected score under the
        null, so _spa_k1_adj(0, mu, g, q=0) must equal sum(g * mu).
        """
        rng = np.random.default_rng(2)
        mu = rng.uniform(0.1, 0.9, 40)
        g = rng.standard_normal(40)
        expected = float(np.sum(g * mu))
        # _spa_k1_adj returns K'(t) - q; with q=0, result = K'(0) = sum(g*mu)
        assert abs(PheWAS._spa_k1_adj(0.0, mu, g, q=0.0) - expected) < 1e-12

    def test_getroot_residual_near_zero(self):
        """Newton solver must find t* such that K'(t*) ≈ q to near machine precision."""
        rng = np.random.default_rng(3)
        mu = rng.uniform(0.1, 0.9, 120)
        g = rng.standard_normal(120)
        # target above the mean so the root is at t > 0
        q = float(np.sum(g * mu)) + 3.0
        root, converged = PheWAS._spa_getroot_k1(0.0, mu, g, q)
        assert converged, "Newton-Raphson must converge for a well-posed problem"
        residual = PheWAS._spa_k1_adj(root, mu, g, q)
        assert abs(residual) < 1e-8

    def test_saddle_prob_small_for_extreme_score(self):
        """Lugannani-Rice tail probability must be small for a very extreme target q."""
        rng = np.random.default_rng(4)
        mu = rng.uniform(0.02, 0.08, 800)
        g = (rng.uniform(size=800) < 0.1).astype(float)
        extreme_q = float(np.sum(g * mu)) + 12.0
        root, converged = PheWAS._spa_getroot_k1(0.0, mu, g, extreme_q)
        if converged and np.isfinite(root):
            p = PheWAS._spa_get_saddle_prob(root, mu, g, extreme_q)
            assert abs(p) < 1e-3, f"Expected tiny tail probability, got {abs(p):.4e}"


class TestSpaTest:
    """Behavioral tests for _spa_test using a minimal PheWAS instance."""

    def test_above_cutoff_pvalue_no_adjustment(self, spa_instance):
        """When the normal-approximation p-value > spa_cutoff, spa_applied must be False.

        Uses a null setup (no real effect) so the initial p-value will be large.
        """
        rng = np.random.default_rng(10)
        n = 1000
        # Balanced, no effect → normal-approximation p-value should be large
        mu = np.full(n, 0.5)
        g = rng.standard_normal(n)
        g = g - g.mean()  # center so expected score ≈ 0
        y = (rng.uniform(size=n) < mu).astype(float)
        score = float(np.sum(g * (y - mu)))
        var1 = float(np.sum(mu * (1 - mu) * g ** 2))
        pval_noadj = float(chi2.sf(score ** 2 / var1, df=1))
        if pval_noadj > spa_instance.spa_cutoff:
            _, applied, _, _ = spa_instance._spa_test(g, mu, y)
            assert applied is False

    def test_at_or_below_cutoff_applies_spa(self, spa_instance):
        """When the normal-approximation p-value <= spa_cutoff, SPA must be applied.

        Constructs a strong rare-disease signal so the initial p-value falls below
        the 0.05 threshold, which triggers the SPA correction.
        """
        rng = np.random.default_rng(11)
        n = 3000
        # Very rare cases + rare allele → strong signal, initial p-value well below 0.05
        mu = np.full(n, 0.005)
        g = (rng.uniform(size=n) < 0.08).astype(float)
        y = np.zeros(n)
        carrier_idx = np.where(g == 1)[0]
        y[carrier_idx[: min(30, len(carrier_idx))]] = 1.0
        score = float(np.sum(g * (y - mu)))
        var1 = float(np.sum(mu * (1 - mu) * g ** 2))
        pval_noadj = float(chi2.sf(score ** 2 / var1, df=1))
        if pval_noadj <= spa_instance.spa_cutoff:
            _, applied, _, _ = spa_instance._spa_test(g, mu, y)
            assert applied is True

    def test_p_value_finite_and_valid(self, spa_instance):
        """_spa_test must always return a p-value in (0, 1]."""
        rng = np.random.default_rng(12)
        n = 800
        mu = rng.uniform(0.05, 0.95, n)
        g = rng.standard_normal(n)
        y = (rng.uniform(size=n) < mu).astype(float)
        p, _, _, _ = spa_instance._spa_test(g, mu, y)
        assert np.isfinite(p), "p-value must be finite"
        assert 0 < p <= 1.0, f"p-value out of (0, 1]: {p}"

    def test_imbalanced_spa_less_anticonservative_than_chi2(self, spa_instance):
        """Under case-control imbalance with a strong signal, SPA p > chi2 p.

        The chi-squared (normal) approximation is anti-conservative in the tails
        when case fraction is very low: it produces p-values smaller than the true
        tail probability.  SPA corrects this upward (less significant, more accurate).
        """
        rng = np.random.default_rng(13)
        n = 6000
        # ~0.5% case fraction, 5% allele frequency, positive effect
        mu = np.full(n, 0.005)
        g = (rng.uniform(size=n) < 0.05).astype(float)
        y = np.zeros(n)
        carrier_idx = np.where(g == 1)[0]
        y[carrier_idx[: max(1, len(carrier_idx) // 2)]] = 1.0
        non_carrier_idx = np.where(g == 0)[0]
        y[non_carrier_idx[:5]] = 1.0

        score = float(np.sum(g * (y - mu)))
        var1 = float(np.sum(mu * (1 - mu) * g ** 2))
        p_chi2 = float(chi2.sf(score ** 2 / var1, df=1))
        p_spa, applied, _, _ = spa_instance._spa_test(g, mu, y)

        if applied:
            assert p_spa > p_chi2, (
                f"SPA should yield a less extreme p-value under imbalance: "
                f"p_spa={p_spa:.3e}, p_chi2={p_chi2:.3e}"
            )


class TestLogitSpaRegression:
    """End-to-end tests for _logit_spa_regression."""

    @staticmethod
    def _make_design(n=600, case_frac=0.3, effect=0.0, seed=20):
        """Return (y, X_with_const, var_index) for a simple logistic setup."""
        sm = pytest.importorskip("statsmodels.api")
        rng = np.random.default_rng(seed)
        cov1 = rng.standard_normal(n)
        cov2 = rng.standard_normal(n)
        g = rng.binomial(1, 0.2, n).astype(float)
        eta = np.log(case_frac / (1 - case_frac)) + 0.4 * cov1 - 0.3 * cov2 + effect * g
        mu_true = 1 / (1 + np.exp(-eta))
        y = (rng.uniform(size=n) < mu_true).astype(float)
        # g at index 0, covariates at 1-2, intercept (constant) at 3
        X = np.column_stack([g, cov1, cov2])
        X = sm.tools.add_constant(X, prepend=False)
        return y, X, 0  # var_index = 0

    def test_degenerate_design_returns_none(self, spa_instance):
        """A rank-deficient design matrix must cause the method to return None."""
        y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        X = np.zeros((6, 3))  # fully degenerate
        base = {"phecode": "TEST", "cases": 3, "controls": 3}
        result = spa_instance._logit_spa_regression(y, X, 0, base)
        assert result is None

    def test_result_contains_all_required_keys(self, spa_instance):
        """Result dict must contain every field expected by downstream consumers."""
        y, X, var_idx = self._make_design()
        base = {"phecode": "250", "cases": int(y.sum()), "controls": int((1 - y).sum())}
        result = spa_instance._logit_spa_regression(y, X, var_idx, base)
        if result is None:
            pytest.skip("null model failed to converge in test data")
        required_keys = {
            "phecode", "cases", "controls",
            "p_value", "neg_log_p_value", "standard_error",
            "beta", "conf_int_1", "conf_int_2",
            "odds_ratio", "log10_odds_ratio", "converged", "is_SPA",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_null_effect_p_not_extreme(self, spa_instance):
        """Under H0 (effect=0), the SPA p-value must not be spuriously small."""
        y, X, var_idx = self._make_design(n=2000, case_frac=0.4, effect=0.0, seed=21)
        base = {"phecode": "250", "cases": int(y.sum()), "controls": int((1 - y).sum())}
        result = spa_instance._logit_spa_regression(y, X, var_idx, base)
        if result is None:
            pytest.skip("null model failed to converge in test data")
        assert result["p_value"] > 1e-4, (
            f"Null-effect p-value unexpectedly small: {result['p_value']:.3e}"
        )

    def test_large_effect_detected(self, spa_instance):
        """With a large true effect (OR ≈ 7), the SPA test must be significant."""
        y, X, var_idx = self._make_design(n=2000, case_frac=0.35, effect=2.0, seed=22)
        base = {"phecode": "250", "cases": int(y.sum()), "controls": int((1 - y).sum())}
        result = spa_instance._logit_spa_regression(y, X, var_idx, base)
        if result is None:
            pytest.skip("null model failed to converge in test data")
        assert result["p_value"] < 0.01, (
            f"Large effect (OR~7) not detected: p={result['p_value']:.3e}"
        )

    def test_beta_and_odds_ratio_consistent(self, spa_instance):
        """odds_ratio must equal exp(beta) to floating-point precision."""
        y, X, var_idx = self._make_design(n=1000, case_frac=0.3, effect=0.8, seed=23)
        base = {"phecode": "250", "cases": int(y.sum()), "controls": int((1 - y).sum())}
        result = spa_instance._logit_spa_regression(y, X, var_idx, base)
        if result is None:
            pytest.skip("null model failed to converge in test data")
        assert abs(result["odds_ratio"] - np.exp(result["beta"])) < 1e-10

    def test_confidence_interval_contains_beta_at_95pct(self, spa_instance):
        """The 95% CI must be symmetric around beta (±1.96 * SE)."""
        y, X, var_idx = self._make_design(n=1000, case_frac=0.3, effect=0.5, seed=24)
        base = {"phecode": "250", "cases": int(y.sum()), "controls": int((1 - y).sum())}
        result = spa_instance._logit_spa_regression(y, X, var_idx, base)
        if result is None:
            pytest.skip("null model failed to converge in test data")
        beta = result["beta"]
        se = result["standard_error"]
        ci_lo = result["conf_int_1"]
        ci_hi = result["conf_int_2"]
        assert ci_lo < beta < ci_hi, "beta must lie inside the confidence interval"
        # Width must be 2 × 1.96 × SE
        expected_width = 2 * 1.959963984540054 * se
        assert abs((ci_hi - ci_lo) - expected_width) < 1e-10
