import numpy as np
import pytest

from vancotwin.calibration import fit_posterior
from vancotwin.pk_model import Dose, concentration, regimen


@pytest.mark.slow
def test_pymc_recovers_individual_parameters():
    doses = [Dose(0.0, 2000.0, 2.0)] + regimen(1000.0, 12.0, 6, start=12.0)
    cl_true, v_true = 2.5, 70.0
    t = np.array([3.0, 14.0, 23.5, 35.5])
    c = concentration(t, doses, cl_true, v_true)
    post = fit_posterior(80.0, 80.0, doses, t, c, draws=400, tune=400, chains=2, seed=1)
    assert abs(np.median(post.cl) - cl_true) / cl_true < 0.15
    assert abs(np.median(post.v) - v_true) / v_true < 0.25
    assert post.diagnostics["r_hat_max"] < 1.05
    assert np.quantile(post.cl, 0.05) < cl_true < np.quantile(post.cl, 0.95)


@pytest.mark.slow
def test_pymc_without_observations_returns_prior():
    doses = regimen(1000.0, 12.0, 2)
    post = fit_posterior(100.0, 70.0, doses, [5.0], [np.nan], draws=300, tune=300, chains=1, seed=0)
    assert np.median(post.cl) == pytest.approx(4.0, rel=0.1)
