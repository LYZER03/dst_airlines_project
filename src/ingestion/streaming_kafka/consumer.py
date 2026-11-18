"""
Consumer Kafka - DST Airlines temps réel
Écoute le topic des vols, interroge l'API de prédiction, et sauvegarde
le résultat dans PostgreSQL (table live_predictions) pour le dashboard.

Lancement : python consumer.py
(reste en écoute ; Ctrl+C pour arrêter)
"""
import json
import requests
from datetime import datetime, timezone

from kafka import KafkaConsumer
from sqlalchemy import create_engine, text

# --- Configuration ---
KAFKA_TOPIC = 'live_flights_topic'
KAFKA_SERVERS = ['localhost:9092']
API_URL = "http://localhost:8000/predict"
POSTGRES_URI = "postgresql://dst_admin:dst_password123@localhost:5432/dst_db"

# --- Connexion PostgreSQL + création de la table si besoin ---
engine = create_engine(POSTGRES_URI, pool_pre_ping=True)

def init_table():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS live_predictions (
                id              SERIAL PRIMARY KEY,
                callsign        TEXT,
                received_at     TIMESTAMPTZ,
                arrival_hour    INTEGER,
                visibility_m    REAL,
                wind_gusts_kmh  REAL,
                is_delayed      BOOLEAN,
                delay_proba     REAL,
                raw_payload     JSONB
            );
        """))
    print("Table live_predictions prête")

def save_prediction(flight: dict, result: dict):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO live_predictions
                (callsign, received_at, arrival_hour, visibility_m,
                 wind_gusts_kmh, is_delayed, delay_proba, raw_payload)
            VALUES
                (:callsign, :received_at, :arrival_hour, :visibility_m,
                 :wind_gusts_kmh, :is_delayed, :delay_proba, :raw_payload)
        """), {
            "callsign": flight.get("callsign"),
            "received_at": datetime.now(timezone.utc),
            "arrival_hour": flight.get("arrival_hour"),
            "visibility_m": flight.get("visibility_m"),
            "wind_gusts_kmh": flight.get("wind_gusts_kmh"),
            "is_delayed": result["is_delayed_prediction"],
            "delay_proba": result["delay_probability"],
            "raw_payload": json.dumps(flight),
        })


def main():
    init_table()

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest',   # lit les nouveaux messages en direct
    )
    print(f"Écoute du topic Kafka '{KAFKA_TOPIC}' en cours...")

    for message in consumer:
        flight = message.value
        callsign = flight.get('callsign', 'Inconnu')
        print(f"\nVol reçu : {callsign}")

        try:
            response = requests.post(API_URL, json=flight, timeout=5)
            if response.status_code == 200:
                result = response.json()
                statut = "EN RETARD" if result['is_delayed_prediction'] else "À L'HEURE"
                print(f"Prédiction : {statut} (Proba: {result['delay_probability']:.2%})")
                save_prediction(flight, result)
                print("Sauvegardé dans PostgreSQL (live_predictions)")
            else:
                print(f"Erreur de l'API ({response.status_code}) : {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Erreur de connexion à l'API : {e}")
        except Exception as e:
            print(f"Erreur de sauvegarde : {e}")


if __name__ == "__main__":
    main()