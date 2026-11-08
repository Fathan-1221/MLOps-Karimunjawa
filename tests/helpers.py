import pandas as pd


def make_frame(n=6, start="2026-01-01 00:00", **overrides):
    """DataFrame cuaca per jam dengan nilai tenang, bisa ditimpa per kolom."""
    df = pd.DataFrame({
        "timestamp": pd.date_range(start, periods=n, freq="h"),
        "temperature": 28.0,
        "humidity": 70.0,
        "pressure": 1010.0,
        "wind_speed": 5.0,
        "rain": 0.0,
    })
    for column, value in overrides.items():
        df[column] = value
    return df
