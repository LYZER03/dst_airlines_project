import pandas as pd
from pymongo import MongoClient
import holidays
from datetime import datetime, timezone
import numpy as np

# ==========================================
# 1. CONNEXION À MONGODB
# ==========================================
# On se connecte à la base de données qui tourne dans notre conteneur Docker
client = MongoClient("mongodb://localhost:27017/") 
db = client["dst_airlines"]

print("1. Extraction des données de la couche Bronze...")

# ==========================================
# 2. PRÉPARATION DE LA MÉTÉO (BRONZE -> PANDAS)
# ==========================================
weather_records = []
for doc in db.bronze_weather.find():
    hourly_data = doc['raw_data']['hourly']
    
    # Éclatement des données : 1 document de 24h -> 24 lignes (1 par heure)
    for i in range(len(hourly_data['time'])):
        weather_records.append({
            'time': pd.to_datetime(hourly_data['time'][i]),
            'temp_c': hourly_data['temperature_2m'][i],
            'precipitation_mm': hourly_data['precipitation'][i],
            'wind_kmh': hourly_data['wind_speed_10m'][i],
            'wind_gusts_kmh': hourly_data.get('wind_gusts_10m', [None]*24)[i],
            'visibility_m': hourly_data.get('visibility', [None]*24)[i],
            'weather_code': hourly_data.get('weather_code', [None]*24)[i]
        })

df_weather = pd.DataFrame(weather_records)

# --- RÈGLE MÉTIER : IMPUTATION DE LA VISIBILITÉ ---
def impute_visibility(row):
    if pd.notna(row['visibility_m']):
        return row['visibility_m']
    if row['precipitation_mm'] > 0:
        return 4000  # Imputation Pluie
    if row['weather_code'] in [45, 48]: 
        return 500   # Imputation Brouillard
    return 10000     # Imputation Temps Clair par défaut

df_weather['visibility_m'] = df_weather.apply(impute_visibility, axis=1)

# ==========================================
# 3. PRÉPARATION DES VOLS (BRONZE -> PANDAS)
# ==========================================
flights_records = []
for doc in db.bronze_flights.find():
    rd = doc['raw_data']
    flights_records.append({
        'bronze_id': doc['_id'],
        'icao24': rd.get('icao24'),
        'callsign': rd.get('callsign', ''),
        'firstSeen': rd.get('firstSeen'),
        'lastSeen': rd.get('lastSeen'),
        'estDepartureAirport': rd.get('estDepartureAirport'),
        'target_date_str': doc['target_date_str']
    })

df_flights = pd.DataFrame(flights_records)

print("2. Transformation, Nettoyage et Feature Engineering...")

# --- NETTOYAGE DE BASE ---
# Exclure les vols sans date d'arrivée
df_flights = df_flights.dropna(subset=['lastSeen'])

# Convertir lastSeen en date UTC lisible
df_flights['arrival_ts'] = pd.to_datetime(df_flights['lastSeen'], unit='s', utc=True)

# Clé de jointure pour la météo (arrondie à l'heure, sans timezone)
df_flights['join_hour_key'] = df_flights['arrival_ts'].dt.floor('h').dt.tz_localize(None)

# --- FEATURE KILLER : ENCOMBREMENT DE L'AÉROPORT ---
# Fait AVANT de filtrer Air France pour avoir la vraie densité de CDG
df_flights['hourly_flight_count'] = df_flights.groupby('join_hour_key')['bronze_id'].transform('count')
df_flights['daily_flight_count'] = df_flights.groupby('target_date_str')['bronze_id'].transform('count')

# --- FILTRE AIR FRANCE & DOUBLONS ---
df_flights['callsign'] = df_flights['callsign'].str.strip()
df_afr = df_flights[df_flights['callsign'].str.startswith('AFR', na=False)].copy()

# Suppression stricte des doublons (Même vol, même jour)
df_afr = df_afr.drop_duplicates(subset=['callsign', 'target_date_str'], keep='first')

# --- ENRICHISSEMENT (FEATURES TEMPORELLES) ---
df_afr['origin_airport'] = df_afr['estDepartureAirport'].fillna('UNKNOWN')
df_afr['duration_min'] = (df_afr['lastSeen'] - df_afr['firstSeen']) // 60
df_afr['duration_min'] = df_afr['duration_min'].replace({np.nan: None})

fr_holidays = holidays.France()
df_afr['day_of_week'] = df_afr['arrival_ts'].dt.weekday
df_afr['month'] = df_afr['arrival_ts'].dt.month
df_afr['is_weekend'] = df_afr['day_of_week'].isin([5, 6])
df_afr['is_holiday'] = df_afr['arrival_ts'].dt.date.apply(lambda d: d in fr_holidays)

# ==========================================
# 4. JOINTURE VOLS + MÉTÉO
# ==========================================
print("3. Jointure spatio-temporelle (Vols + Météo)...")
df_silver = pd.merge(df_afr, df_weather, left_on='join_hour_key', right_on='time', how='left')

# ==========================================
# 5. CHARGEMENT DANS MONGODB (SILVER)
# ==========================================
print("4. Écriture dans la collection silver_flights...")
silver_documents = []
processed_time = datetime.now(timezone.utc)

for _, row in df_silver.iterrows():
    doc = {
        "bronze_id": row['bronze_id'],
        "processed_at": processed_time,
        "flight": {
            "callsign": row['callsign'],
            "icao24": row['icao24'],
            "origin_airport": row['origin_airport'],
            "arrival_ts": row['arrival_ts'],
            "arrival_hour": row['arrival_ts'].hour,
            "duration_min": int(row['duration_min']) if pd.notna(row['duration_min']) else None
        },
        "weather": {
            "temp_c": round(row['temp_c'], 1) if pd.notna(row['temp_c']) else None,
            "wind_kmh": round(row['wind_kmh'], 1) if pd.notna(row['wind_kmh']) else None,
            "precipitation_mm": round(row['precipitation_mm'], 2) if pd.notna(row['precipitation_mm']) else None,
            "visibility_m": int(row['visibility_m']) if pd.notna(row['visibility_m']) else None,
            "wind_gusts_kmh": round(row['wind_gusts_kmh'], 1) if pd.notna(row['wind_gusts_kmh']) else None
        },
        "temporal": {
            "day_of_week": int(row['day_of_week']),
            "month": int(row['month']),
            "is_weekend": bool(row['is_weekend']),
            "is_holiday": bool(row['is_holiday'])
        },
        "context": {
            "daily_flight_count": int(row['daily_flight_count']),
            "hourly_flight_count": int(row['hourly_flight_count'])
        }
    }
    silver_documents.append(doc)

# Idempotence : On vide la collection avant d'insérer pour éviter les doublons si on relance le script
db.silver_flights.delete_many({})

if silver_documents:
    db.silver_flights.insert_many(silver_documents)
    print(f"✅ Succès : {len(silver_documents)} vols Air France ont été insérés en couche Silver !")
else:
    print("⚠️ Aucun vol Air France trouvé après filtrage.")