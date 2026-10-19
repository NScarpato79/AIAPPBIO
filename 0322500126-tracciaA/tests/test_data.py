import numpy as np
import pandas as pd
import pytest

from vancotwin.data import (
    age_band,
    apply_qc,
    cockcroft_gault,
    cohort_tables,
    generate_cohort,
    pseudonymize,
    quality_control,
)


def test_cockcroft_gault_known_values():
    assert cockcroft_gault(40, 72, 1.0, False) == pytest.approx(100.0)
    assert cockcroft_gault(40, 72, 1.0, True) == pytest.approx(85.0)


def test_pseudonymize_is_deterministic_salted_and_opaque():
    a = pseudonymize("MRN100001", "s1")
    assert a == pseudonymize("MRN100001", "s1")
    assert a != pseudonymize("MRN100001", "s2")
    assert "100001" not in a


def test_cohort_is_reproducible_and_minimized():
    p1 = generate_cohort(10, np.random.default_rng(1))
    p2 = generate_cohort(10, np.random.default_rng(1))
    assert [p.cl_true for p in p1] == [p.cl_true for p in p2]
    demo, obs = cohort_tables(p1)
    assert "age" not in demo.columns and "age_band" in demo.columns
    assert not demo["pid"].str.contains("MRN").any()
    assert obs["holdout"].sum() == 10
    assert age_band(67.3) == "60-69"


def test_quality_control_flags_and_removes_bad_values():
    obs = pd.DataFrame(
        {
            "pid": ["a", "a", "b", "b", "c"],
            "sample": [0, 1, 0, 1, 0],
            "time_h": [1.0, 2.0, 1.0, 1.0, 1.0],
            "conc_mg_l": [20.0, np.nan, 25000.0, 30.0, -1.0],
        }
    )
    qc, report = quality_control(obs)
    assert report == {"qc_missing": 1, "qc_negative": 1, "qc_implausible": 1, "qc_duplicate": 1, "n_total": 5, "n_pass": 1}
    assert qc["qc_pass"].tolist() == [True, False, False, False, False]


def test_apply_qc_sets_rejected_samples_to_nan():
    patients = generate_cohort(3, np.random.default_rng(0))
    patients[1].obs_conc[0] = 9999.0
    _, obs = cohort_tables(patients)
    qc, _ = quality_control(obs)
    apply_qc(patients, qc)
    assert np.isnan(patients[1].obs_conc[0])
