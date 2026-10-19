import numpy as np
import pytest

from vancotwin.calibration import ParameterSamples, prior_samples
from vancotwin.config import PopulationModel
from vancotwin.enkf import EnsembleKalmanFilter
from vancotwin.pk_model import Dose, concentration, regimen
from vancotwin.scenarios import Regimen, apply_regimen, evaluate_regimen, recommend
from vancotwin.sensitivity import population_sensitivity, sobol_indices, trough_steady_state


DOSES = [Dose(0.0, 2000.0, 2.0)] + regimen(1000.0, 12.0, 8, start=12.0)


def test_enkf_learns_clearance_and_reduces_uncertainty():
    rng = np.random.default_rng(3)
    pop = PopulationModel()
    cl_true, v_true = 2.0, 60.0  # clearance molto piu' bassa dell'atteso a priori
    prior = prior_samples(100.0, 80.0, 500, rng, pop)
    f = EnsembleKalmanFilter.from_samples(prior, DOSES, rng, pop, n=500)
    sd0 = np.std(f.log_cl)
    for t in (15.0, 35.5, 47.5, 59.5):
        f.assimilate(t, float(concentration([t], DOSES, cl_true, v_true)[0]))
    cl_est = np.exp(np.mean(f.log_cl))
    assert abs(cl_est - cl_true) / cl_true < 0.15
    assert np.std(f.log_cl) < 0.5 * sd0
    assert len(f.history) == 5


def test_enkf_skips_missing_observation():
    rng = np.random.default_rng(0)
    f = EnsembleKalmanFilter.from_samples(prior_samples(90, 70, 100, rng), DOSES, rng, n=100)
    before = f.log_cl.copy()
    f.assimilate(10.0, np.nan)
    np.testing.assert_array_equal(f.log_cl, before)
    assert f.history[-1].y is None


def test_recommendation_hits_target_for_known_clearance():
    cl = np.full(200, 3.0)
    samples = ParameterSamples(cl, np.full(200, 50.0), "test")
    best, table = recommend(samples, DOSES, 36.0)
    assert 400 <= best.daily_dose / 3.0 <= 600
    assert table["pta"].between(0, 1).all()
    assert table.iloc[0]["pta"] == table["pta"].max()


def test_apply_regimen_replaces_future_doses():
    new = apply_regimen(DOSES, Regimen(750.0, 8.0), 36.0)
    assert all(d.amount == 750.0 for d in new if d.time >= 36.0)
    assert [d for d in new if d.time < 36.0] == [d for d in DOSES if d.time < 36.0]


def test_evaluate_regimen_toxicity_increases_with_dose():
    rng = np.random.default_rng(1)
    s = prior_samples(80, 70, 400, rng)
    low = evaluate_regimen(s, DOSES, Regimen(500, 12), 36.0)
    high = evaluate_regimen(s, DOSES, Regimen(2000, 12), 36.0)
    assert high["p_auc_toxic"] > low["p_auc_toxic"]


def test_sobol_identifies_only_influential_factor():
    rng = np.random.default_rng(0)
    samplers = {"x1": lambda n, r: r.normal(size=n), "x2": lambda n, r: r.normal(size=n)}
    df = sobol_indices(lambda x: 3 * x["x1"], samplers, 4000, rng).set_index("factor")
    assert df.loc["x1", "S1"] == pytest.approx(1.0, abs=0.05)
    assert abs(df.loc["x2", "ST"]) < 0.02


def test_volume_does_not_affect_steady_state_auc():
    df = population_sensitivity(4000, np.random.default_rng(0)).set_index(["output", "factor"])
    assert df.loc[("AUC24", "eta_V"), "ST"] < 0.02
    assert df.loc[("AUC24", "peso"), "ST"] < 0.02


def test_trough_steady_state_matches_simulation():
    doses = regimen(1000.0, 12.0, 60)
    sim = concentration([12 * 50 - 1e-9], doses, 4.0, 50.0)[0]
    assert sim == pytest.approx(float(trough_steady_state(1000, 12, 1.0, 4.0, 50.0)), rel=1e-6)
