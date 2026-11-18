"""
Producer temps réel - DST Airlines (Chantier 2 : APIs externes)
Combine la météo réelle (Open-Meteo) avec de vrais vols tirés de la base
gold_ml_features, et envoie le tout dans Kafka en boucle.

Pourquoi ce montage : l'API OpenSky (vols en direct) est filtrée depuis la VM.
On récupère donc les vols réels (routes, taux de retard historiques) dans notre
propre base, qu'on enrichit de la météo temps réel d'Open-Meteo. Le code OpenSky
est prêt en commentaire (fonction fetch_opensky) et s'activera dès que l'accès
réseau sera ouvert.

Lancement : python producer.py
Arrêt : Ctrl+C
"""
import json
import time
import signal
import sys

import requests
import pandas as pd
from sqlalchemy import create_engine, text
from kafka import KafkaProducer

# --- Configuration ---
KAFKA_TOPIC = "live_flights_topic"
KAFKA_SERVERS = ["localhost:9092"]
POSTGRES_URI = "postgresql://dst_admin:dst_password123@localhost:5432/dst_db"

# Coordonnées de Paris-CDG pour la météo
CDG_LAT, CDG_LON = 49.01, 2.55
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

CYCLE_MINUTES = 2      # un cycle toutes les 2 minutes
FLIGHTS_PER_CYCLE = 5  # nombre de vols envoyés par cycle

engine = create_engine(POSTGRES_URI, pool_pre_ping=True)

producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)


def fetch_weather():
    """Météo temps réel à CDG via Open-Meteo (visibilité, rafales, vent, temp)."""
    try:
        r = requests.get(OPEN_METEO_URL, params={
            "latitude": CDG_LAT, "longitude": CDG_LON,
            "current": "wind_gusts_10m,visibility,wind_speed_10m,temperature_2m",
        }, timeout=10)
        r.raise_for_status()
        cur = r.json()["current"]
        return {
            "visibility_m": float(cur.get("visibility", 9000)),
            "wind_gusts_kmh": float(cur.get("wind_gusts_10m", 15)),
            "wind_kmh": float(cur.get("wind_speed_10m", 10)),
            "temp_c": float(cur.get("temperature_2m", 15)),
        }
    except Exception as e:
        print(f"  [meteo] Open-Meteo indisponible ({e}) - valeurs neutres")
        return {"visibility_m": 9000, "wind_gusts_kmh": 15,
                "wind_kmh": 10, "temp_c": 15}


def fetch_flights(n):
    """Tire n vols réels au hasard dans gold_ml_features (routes + historiques)."""
    query = text(f"""
        SELECT callsign, origin_airport, arrival_hour, day_of_week, month,
               hourly_flight_count, route_delay_rate_30d, aircraft_delay_rate_7d
        FROM gold_ml_features
        ORDER BY RANDOM()
        LIMIT {int(n)};
    """)
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(query)]


# --- OpenSky : prêt à activer dès que l'accès réseau sera ouvert ---
# def fetch_opensky():
#     r = requests.get("https://opensky-network.org/api/states/all",
#                      params={"lamin":48.7,"lomin":2.1,"lamax":49.3,"lomax":3.0},
#                      timeout=15)
#     return r.json().get("states", [])


def build_message(flight, weather):
    """Assemble un vol + la météo du moment au format attendu par l'API."""
    return {
        "callsign": flight.get("callsign") or "AFXXXX",
        "hourly_flight_count": float(flight.get("hourly_flight_count") or 25),
        "visibility_m": weather["visibility_m"],
        "wind_gusts_kmh": weather["wind_gusts_kmh"],
        "route_delay_rate_30d": float(flight.get("route_delay_rate_30d") or 0.2),
        "aircraft_delay_rate_7d": float(flight.get("aircraft_delay_rate_7d") or 0.2),
        "arrival_hour": int(flight.get("arrival_hour") or 12),
        "day_of_week": int(flight.get("day_of_week") or 0),
        "month": int(flight.get("month") or 6),
    }


def run_cycle():
    weather = fetch_weather()
    print(f"  Météo CDG : visibilité {weather['visibility_m']:.0f} m, "
          f"rafales {weather['wind_gusts_kmh']:.0f} km/h")
    flights = fetch_flights(FLIGHTS_PER_CYCLE)
    for flight in flights:
        msg = build_message(flight, weather)
        producer.send(KAFKA_TOPIC, value=msg)
        print(f"  Envoyé : {msg['callsign']} "
              f"(origine {flight.get('origin_airport')}, "
              f"route_delay {msg['route_delay_rate_30d']:.2f})")
    producer.flush()
    print(f"  {len(flights)} vols envoyés dans Kafka")


def main():
    def stop(sig, frame):
        print("\nArrêt du producer.")
        producer.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, stop)

    print(f"Producer temps réel démarré (cycle toutes les {CYCLE_MINUTES} min). "
          "Ctrl+C pour arrêter.")
    while True:
        print(f"\n--- Nouveau cycle ({time.strftime('%H:%M:%S')}) ---")
        try:
            run_cycle()
        except Exception as e:
            print(f"  Erreur de cycle : {e}")
        time.sleep(CYCLE_MINUTES * 60)


if __name__ == "__main__":
    main()
