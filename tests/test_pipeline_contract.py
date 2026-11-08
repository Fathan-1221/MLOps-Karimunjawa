"""Pastikan keluaran preprocessing tetap cocok dengan yang diharapkan train.py."""

from src import preprocess as pp
from src import train
from tests.helpers import make_frame


def test_train_features_exist_after_preprocessing():
    df = pp.create_label(pp.feature_engineering(pp.clean_data(make_frame(n=12))))
    missing = [f for f in train.FEATURES if f not in df.columns]
    assert not missing, f"Fitur hilang dari hasil preprocessing: {missing}"


def test_train_can_find_numeric_label_column():
    df = pp.create_label(pp.feature_engineering(pp.clean_data(make_frame(n=12))))
    assert train.pick_label_column(df) in ("kelayakan", "kelayakan_label")
