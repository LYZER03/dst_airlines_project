"""
API Prédiction Retards DST Airlines - Temps réel
Reçoit les variables dynamiques d'un vol (via Kafka), reconstitue les ~30 features
attendues par le modèle XGBoost, et renvoie la probabilité de retard.

Lancement : uvicorn api:app --reload --port 8000
Documentation interactive : http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="API Prédiction Retards DST Airlines")

# --- 1. Chargement du modèle ML ---
model = None
try:
    model = joblib.load("./ml_engine/models/xgb_classifier.pkl")
    print("Modèle XGBoost chargé avec succès")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")

# --- 2. Valeurs par défaut des features NON envoyées par Kafka ---
# (médianes / modes du jeu d'entraînement). Les features dérivées du temps
# (sin/cos) et les flags ne figurent PAS ici : ils sont recalculés à chaque
# requête à partir des entrées, pour rester cohérents avec le vol réel.
default_features = {
    'wind_kmh': 15.0,
    'temp_c': 15.0,
    'daily_flight_count': 500,
    'duration_min': 120,
    'origin_airport': 'UNKNOWN',
}

# --- 3. Schéma d'entrée : variables dynamiques envoyées par Kafka ---
class FlightData(BaseModel):
    hourly_flight_count: float = Field(..., description="Trafic horaire à CDG")
    visibility_m: float = Field(..., description="Visibilité en mètres")
    route_delay_rate_30d: float = Field(..., description="Taux de retard historique de la route")
    aircraft_delay_rate_7d: float = Field(..., description="Taux de retard historique de l'avion")
    wind_gusts_kmh: float = Field(..., description="Rafales de vent (km/h)")
    arrival_hour: int = Field(..., ge=0, le=23, description="Heure d'arrivée prévue")
    day_of_week: Optional[int] = Field(0, ge=0, le=6, description="0 = lundi")
    month: Optional[int] = Field(6, ge=1, le=12)
    callsign: Optional[str] = Field(None, description="Identifiant du vol (pour traçage)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "hourly_flight_count": 55, "visibility_m": 400,
                "route_delay_rate_30d": 0.35, "aircraft_delay_rate_7d": 0.30,
                "wind_gusts_kmh": 80, "arrival_hour": 8,
                "day_of_week": 0, "month": 1, "callsign": "AF1234"
            }
        }
    }


def build_features(flight: dict) -> pd.DataFrame:
    """Reconstitue la ligne complète attendue par le modèle :
    défauts + variables dynamiques + features dérivées recalculées."""
    row = default_features.copy()

    hour = flight['arrival_hour']
    day = flight.get('day_of_week', 0)
    month = flight.get('month', 6)
    vis = flight['visibility_m']
    gusts = flight['wind_gusts_kmh']

    row['hourly_flight_count'] = flight['hourly_flight_count']
    row['visibility_m'] = vis
    row['route_delay_rate_30d'] = flight['route_delay_rate_30d']
    row['aircraft_delay_rate_7d'] = flight['aircraft_delay_rate_7d']
    row['wind_gusts_kmh'] = gusts
    row['arrival_hour'] = hour

    # Encodages cycliques recalculés (cohérents avec l'heure/jour/mois réels)
    row['arrival_hour_sin'] = np.sin(2 * np.pi * hour / 24)
    row['arrival_hour_cos'] = np.cos(2 * np.pi * hour / 24)
    row['day_sin'] = np.sin(2 * np.pi * day / 7)
    row['day_cos'] = np.cos(2 * np.pi * day / 7)
    row['month_sin'] = np.sin(2 * np.pi * month / 12)
    row['month_cos'] = np.cos(2 * np.pi * month / 12)

    # Flags opérationnels déduits des entrées
    row['is_lvp'] = int(vis < 600)
    row['is_high_wind'] = int(gusts > 55)
    row['is_peak_hour'] = int(hour in [6, 7, 8, 9, 17, 18, 19, 20])
    row['is_weekend'] = int(day >= 5)
    row['is_holiday'] = 0

    df = pd.DataFrame([row])

    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)

    if hasattr(model, "feature_names_in_"):
        df = df[list(model.feature_names_in_)]
    return df


@app.get("/")
def root():
    return {"service": "API Prédiction Retards DST Airlines",
            "documentation": "/docs", "model_loaded": model is not None}


@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "degraded",
            "model_loaded": model is not None}


@app.post("/predict")
def predict_delay(flight: FlightData):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    try:
        df_input = build_features(flight.model_dump())
        prediction = model.predict(df_input)[0]
        probabilite = model.predict_proba(df_input)[0][1]
        return {
            "callsign": flight.callsign,
            "is_delayed_prediction": bool(prediction),
            "delay_probability": float(probabilite),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))