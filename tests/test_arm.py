"""Official Arm categories and private raw-MRI clinical preparation."""
import pytest

from mri_vla_jepa.arm import (
    OFFICIAL_ARMS, arm_id, visible_arm, make_arm_record,
    validate_arm_record, normalize_patient_key, load_official_clinical,
)


def test_exact_official_vocabulary_and_unknown():
    assert len(OFFICIAL_ARMS) == 13 and len(set(OFFICIAL_ARMS)) == 13
    assert [arm_id(v) for v in OFFICIAL_ARMS] == list(range(1, 14))
    assert arm_id(None) == 0
    with pytest.raises(ValueError, match="13 official"):
        arm_id("Paclitaxel + invented drug")


def test_explicit_semantics_and_time():
    record = make_arm_record(OFFICIAL_ARMS[4], 2, "prospective_verified")
    assert visible_arm(record, 0) == (0, False, -1)
    assert visible_arm(record, 2) == (5, True, 2)
    with pytest.raises(ValueError, match="explicit canonical"):
        make_arm_record(OFFICIAL_ARMS[0], None, "prospective_verified")
    with pytest.raises(ValueError, match="semantics"):
        make_arm_record(OFFICIAL_ARMS[0], 0, "retrospective_assumed_baseline")
    with pytest.raises(ValueError, match="null known-at"):
        validate_arm_record({"label": None, "known_at_stage": 0, "semantics": "assigned_arm_scenario"})


def test_private_id_join_normalization():
    assert normalize_patient_key("ISPY2-123456") == "123456"
    assert normalize_patient_key(123456.0) == "123456"
    with pytest.raises(ValueError, match="missing"):
        normalize_patient_key(float("nan"))


def test_clinical_csv_read_is_private_and_missing_aware(tmp_path):
    pytest.importorskip("pandas")
    path = tmp_path / "fake_clinical.csv"
    path.write_text("Patient_ID,Arm,HR,HER2,MP,pCR,Age_at_Screening\n123456,Paclitaxel,1,0,1,1,\n")
    rows = load_official_clinical(path)
    assert rows["123456"]["clinical_values"] == [None, 1., 0., 1.]
    assert rows["123456"]["pcr"] == 1
