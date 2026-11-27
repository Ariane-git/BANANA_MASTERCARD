"""Chargement, apprentissage et prévision de l'action Mastercard (MA)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from neuralprophet import NeuralProphet
from neuralprophet.utils import load as load_neuralprophet, save as save_neuralprophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


TICKER = "MA"
HORIZON_DEFAULT = 20
HORIZON_MAX = 40
HISTORY_PERIOD = "8y"


def fetch_history(period: str = HISTORY_PERIOD) -> pd.DataFrame:
    """Télécharge les cours ajustés, avec contrôle explicite des colonnes et dates."""
    raw = yf.download(TICKER, period=period, interval="1d", auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("Yahoo Finance n'a renvoyé aucune donnée pour MA.")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    result = pd.DataFrame({"ds": pd.to_datetime(close.index).tz_localize(None), "y": close.to_numpy()})
    result = result.dropna().drop_duplicates("ds").sort_values("ds").reset_index(drop=True)
    if len(result) < 300:
        raise RuntimeError("Historique trop court pour une validation temporelle fiable.")
    return result


def build_model(epochs: int = 80) -> NeuralProphet:
    return NeuralProphet(
        n_lags=0,
        n_forecasts=1,
        yearly_seasonality="auto",
        weekly_seasonality=True,
        daily_seasonality=False,
        quantiles=[0.10, 0.90],
        epochs=epochs,
        learning_rate=0.01,
        normalize="soft",
    )


def validation_split(data: pd.DataFrame, horizon: int = HORIZON_DEFAULT):
    if len(data) <= horizon + 100:
        raise ValueError("Il faut davantage d'historique que la fenêtre de validation.")
    return data.iloc[:-horizon].copy(), data.iloc[-horizon:].copy()


def forecast_frame(model: NeuralProphet, future_dates: pd.DatetimeIndex) -> pd.DataFrame:
    future = pd.DataFrame({"ds": pd.to_datetime(future_dates), "y": np.nan})
    predicted = model.predict(future)
    if "yhat1" not in predicted:
        raise RuntimeError("NeuralProphet n'a pas produit la colonne de prévision yhat1.")
    result = predicted[["ds", "yhat1"]].copy().rename(columns={"yhat1": "prediction"})
    for quantile in ("10.0%", "90.0%"):
        column = f"yhat1 {quantile}"
        if column in predicted:
            name = "borne_basse" if quantile == "10.0%" else "borne_haute"
            result[name] = predicted[column].to_numpy()
    return result


def make_trading_dates(last_observation, horizon: int = HORIZON_DEFAULT) -> pd.DatetimeIndex:
    """Retourne les prochaines séances du calendrier NYSE, sans week-ends ni fêtes."""
    if not 1 <= horizon <= HORIZON_MAX:
        raise ValueError(f"L'horizon doit être compris entre 1 et {HORIZON_MAX} séances.")
    start = pd.Timestamp(last_observation).normalize() + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=horizon * 2 + 20)
    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(start_date=start, end_date=end)
    dates = pd.DatetimeIndex(schedule.index[:horizon]).tz_localize(None)
    if len(dates) < horizon:
        raise RuntimeError("Impossible de construire toutes les séances de prévision.")
    return dates


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def train_validate_and_save(data: pd.DataFrame, models_dir: Path, epochs: int = 80):
    """Évalue sur les 20 dernières séances, puis réentraîne sur tout l'historique."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    train_data, test_data = validation_split(data)
    validation_model = build_model(epochs=epochs)
    validation_model.fit(train_data, freq="B", progress="none")
    validation_forecast = forecast_frame(validation_model, pd.DatetimeIndex(test_data["ds"]))
    validation_forecast["actual"] = test_data["y"].to_numpy()
    metrics = calculate_metrics(validation_forecast["actual"], validation_forecast["prediction"])

    final_model = build_model(epochs=epochs)
    final_model.fit(data, freq="B", progress="none")
    save_neuralprophet(final_model, str(models_dir / "mastercard_neuralprophet.np"))
    return final_model, validation_forecast, metrics


def load_model(models_dir: Path):
    path = Path(models_dir) / "mastercard_neuralprophet.np"
    if not path.exists():
        raise FileNotFoundError("Modèle boursier absent : exécute d'abord le notebook 02.")
    model = load_neuralprophet(str(path), map_location="cpu")

    # A model exported from Colab may keep its TensorBoard logger under /content.
    # Prediction needs no training metrics, so disable that logger and use a
    # writable local temp directory when rebuilding the Lightning trainer.
    model.metrics = {}
    model.metrics_logger = None
    model.trainer_config = {
        **(model.trainer_config or {}),
        "default_root_dir": str(Path(tempfile.gettempdir()) / "banana_mastercard_neuralprophet"),
    }
    model.restore_trainer(accelerator="cpu")
    return model
