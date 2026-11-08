import pandas as pd
import pytest

from src import preprocess as pp
from tests.helpers import make_frame


def prepared(df):
    return pp.create_label(pp.feature_engineering(pp.clean_data(df)))


def test_strong_wind_is_labelled_tidak_layak():
    df = prepared(make_frame(n=6, wind_speed=pp.WIND_SPEED_THRESHOLD + 1))
    assert (df["kelayakan_label"] == 0).all()


def test_calm_weather_is_labelled_layak():
    df = prepared(make_frame(n=6)).dropna()
    assert (df["kelayakan_label"] == 1).all()


def test_heavy_rain_is_labelled_tidak_layak():
    df = prepared(make_frame(n=6, rain=pp.RAIN_THRESHOLD + 1))
    assert (df["kelayakan_label"] == 0).all()


def test_clean_data_keeps_latest_duplicate():
    old = make_frame(n=3, temperature=20.0)
    new = make_frame(n=3, temperature=30.0)
    cleaned = pp.clean_data(pd.concat([old, new], ignore_index=True))
    assert len(cleaned) == 3
    assert (cleaned["temperature"] == 30.0).all()


def test_features_are_not_computed_across_time_gap():
    first = make_frame(n=5, start="2026-01-01 00:00")
    second = make_frame(n=5, start="2026-03-01 00:00")
    df = pp.feature_engineering(pp.clean_data(pd.concat([first, second], ignore_index=True)))
    # 3 baris pertama tiap segmen belum punya data 3 jam sebelumnya
    assert df["pressure_drop_3h"].isna().sum() == 6


def test_load_data_fails_clearly_when_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "RAW_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        pp.load_data()
