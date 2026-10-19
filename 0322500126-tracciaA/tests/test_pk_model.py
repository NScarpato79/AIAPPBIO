import numpy as np
import pytest

from vancotwin.pk_model import Dose, auc, auc24_steady_state, concentration, regimen, simulate_ode


@pytest.fixture
def doses():
    return [Dose(0.0, 2000.0, 2.0)] + regimen(1000.0, 12.0, 6, start=12.0)


def test_ode_matches_analytic_solution(doses):
    t = np.linspace(0, 84, 337)
    cl, v = np.array([2.0, 4.0, 7.0]), np.array([40.0, 55.0, 80.0])
    np.testing.assert_allclose(simulate_ode(t, doses, cl, v), concentration(t, doses, cl, v), atol=1e-5)


def test_ode_restart_from_intermediate_state(doses):
    cl, v = 4.0, 50.0
    c_mid = concentration([30.0], doses, cl, v)
    t = np.array([31.0, 50.0, 70.0])
    resumed = simulate_ode(t, doses, cl, v, c0=c_mid, t0=30.0)[0]
    np.testing.assert_allclose(resumed, concentration(t, doses, cl, v), atol=1e-5)


def test_concentration_zero_before_first_dose():
    assert concentration([0.0], [Dose(1.0, 1000.0)], 4.0, 50.0)[0] == 0.0


def test_linearity_double_dose_doubles_concentration():
    t = np.linspace(0, 24, 50)
    c1 = concentration(t, [Dose(0, 1000.0)], 4.0, 50.0)
    c2 = concentration(t, [Dose(0, 2000.0)], 4.0, 50.0)
    np.testing.assert_allclose(c2, 2 * c1)


def test_auc24_at_steady_state_equals_daily_dose_over_clearance():
    cl = 3.0
    doses = regimen(1000.0, 12.0, 40)
    numeric = auc(doses, cl, 50.0, 24 * 15, 24 * 16)
    assert numeric == pytest.approx(auc24_steady_state(2000.0, cl), rel=1e-3)


def test_simulate_ode_rejects_past_times(doses):
    with pytest.raises(ValueError):
        simulate_ode([1.0], doses, 4.0, 50.0, t0=5.0)
