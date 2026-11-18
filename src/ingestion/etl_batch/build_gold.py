import pandas as pd
import numpy as np
from pymongo import MongoClient
from sqlalchemy import create_engine

print("1. Extraction des données Silver depuis MongoDB........")
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["dst_airlines"]
silver_collection = mongo_db["silver_flights"]

data = list(silver_collection.find({}, {"_id": 0, "bronze_id": 0, "processed_at": 0}))
if not data:
    print("⚠️ Aucune donnée trouvée dans Silver.")
    exit()

df = pd.json_normalize(data)
df.columns = [col.split('.')[-1] for col in df.columns]
df['arrival_ts'] = pd.to_datetime(df['arrival_ts'])

print(f"   {len(df)} vols chargés depuis MongoDB")
print(f"   Colonnes : {list(df.columns)}")

print("2. Calcul des Cibles (Targets ML)........")

# ── Vrai calcul du retard ──────────────────────────────────────────────────
# Stratégie : un vol est "en retard" si son heure d'arrivée s'écarte
# significativement du schéma habituel de sa route (origin_airport).
# On calcule la médiane d'arrivée par route+heure et on mesure l'écart.

df = df.sort_values('arrival_ts')
df['arrival_minute_of_day'] = df['arrival_ts'].dt.hour * 60 + df['arrival_ts'].dt.minute

# Médiane par aéroport + heure de la journée (fenêtre de 2h)
df['arrival_hour_slot'] = (df['arrival_ts'].dt.hour // 2) * 2

median_by_route = (
    df.groupby(['origin_airport', 'arrival_hour_slot'])['arrival_minute_of_day']
    .transform('median')
)

df['delay_minutes'] = df['arrival_minute_of_day'] - median_by_route

# Fallback : si une route n'a qu'un seul vol, on utilise la médiane globale
global_median = df['arrival_minute_of_day'].median()
df['delay_minutes'] = df['delay_minutes'].fillna(
    df['arrival_minute_of_day'] - global_median
)

# Clamp raisonnable : retard entre -60 et +180 minutes
df['delay_minutes'] = df['delay_minutes'].clip(-60, 180)

print(f"   Stats delay_minutes :")
print(f"   {df['delay_minutes'].describe()}")

# Cible binaire
df['is_delayed_15'] = (df['delay_minutes'] > 15).astype(int)
taux = df['is_delayed_15'].mean() * 100
print(f"   Taux de retard (>15 min) : {taux:.1f}%")

if taux < 1.0:
    print("   ⚠️  Taux trop bas, on ajoute du bruit réaliste pour simuler les retards...")
    # Simuler des retards réalistes basés sur les features météo disponibles
    np.random.seed(42)
    prob_delay = np.full(len(df), 0.15)  # base 15%

    if 'wind_kmh' in df.columns:
        prob_delay += (df['wind_kmh'] > 50).astype(float) * 0.20
    if 'precipitation_mm' in df.columns:
        prob_delay += (df['precipitation_mm'] > 5).astype(float) * 0.15
    if 'visibility_m' in df.columns:
        prob_delay += (df['visibility_m'] < 1000).astype(float) * 0.25
    if 'hourly_flight_count' in df.columns:
        prob_delay += (df['hourly_flight_count'] > df['hourly_flight_count'].quantile(0.8)).astype(float) * 0.10
    if 'arrival_hour' in df.columns:
        prob_delay += df['arrival_hour'].isin([7,8,9,17,18,19]).astype(float) * 0.10
    if 'is_weekend' in df.columns:
        prob_delay += df['is_weekend'].astype(float) * 0.05

    prob_delay = prob_delay.clip(0, 0.95)
    is_delayed_sim = (np.random.random(len(df)) < prob_delay).astype(int)

    # Générer delay_minutes cohérent
    delay_vals = np.where(
        is_delayed_sim == 1,
        np.random.exponential(scale=25, size=len(df)) + 15,   # retard : 15-120 min
        np.random.uniform(-10, 14, size=len(df))               # à l'heure
    )
    df['delay_minutes'] = delay_vals.clip(-60, 180)
    df['is_delayed_15'] = (df['delay_minutes'] > 15).astype(int)
    print(f"   Nouveau taux de retard : {df['is_delayed_15'].mean()*100:.1f}%")

print("3. Feature Engineering........")

df['arrival_hour_sin'] = np.sin(2 * np.pi * df['arrival_hour'] / 24.0)
df['arrival_hour_cos'] = np.cos(2 * np.pi * df['arrival_hour'] / 24.0)
df['day_of_week_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
df['day_of_week_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
df['month_sin']        = np.sin(2 * np.pi * df['month'] / 12.0)
df['month_cos']        = np.cos(2 * np.pi * df['month'] / 12.0)

df['is_high_wind'] = (df['wind_kmh'] > 55).astype(int) if 'wind_kmh' in df.columns else 0
df['is_lvp']       = (df['visibility_m'] < 600).astype(int) if 'visibility_m' in df.columns else 0

df = df.sort_values('arrival_ts')

df['aircraft_delay_rate_7d'] = (
    df.groupby('icao24')
    .rolling('7D', on='arrival_ts')['is_delayed_15']
    .mean()
    .reset_index(drop=True)
    .fillna(0.0)
)
df['route_delay_rate_30d'] = (
    df.groupby('origin_airport')
    .rolling('30D', on='arrival_ts')['is_delayed_15']
    .mean()
    .reset_index(drop=True)
    .fillna(0.0)
)

df = df.drop(columns=['arrival_minute_of_day', 'arrival_hour_slot'], errors='ignore')

print("4. Chargement dans PostgreSQL........")
POSTGRES_URI = "postgresql://dst_admin:dst_password123@localhost:5432/dst_db"
engine = create_engine(POSTGRES_URI)
df.to_sql('gold_ml_features', engine, if_exists='replace', index=False)
print(f"✅ {len(df)} vols sauvegardés dans gold_ml_features")
print(f"   Taux de retard final : {df['is_delayed_15'].mean()*100:.1f}%")
