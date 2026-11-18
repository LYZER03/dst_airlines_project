"""
Producer de test - DST Airlines temps réel
Envoie un (ou plusieurs) vol(s) de test dans Kafka pour valider la chaîne.

Lancement : python test_producer.py
"""
import json
import time
import random
from kafka import KafkaProducer

KAFKA_TOPIC = 'live_flights_topic'
KAFKA_SERVERS = ['localhost:9092']

producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
)

# Deux scénarios réalistes (valeurs cohérentes, contrairement à des zéros)
vols_de_test = [
    {
        "callsign": "AF1234",
        "hourly_flight_count": 55,      # trafic saturé
        "visibility_m": 400,            # brouillard (LVP)
        "route_delay_rate_30d": 0.38,   # route souvent en retard
        "aircraft_delay_rate_7d": 0.32,
        "wind_gusts_kmh": 80,           # rafales fortes
        "arrival_hour": 8,              # heure de pointe
        "day_of_week": 0,
        "month": 1,
    },
    {
        "callsign": "AF5678",
        "hourly_flight_count": 12,      # trafic calme
        "visibility_m": 9000,           # ciel dégagé
        "route_delay_rate_30d": 0.06,
        "aircraft_delay_rate_7d": 0.04,
        "wind_gusts_kmh": 8,
        "arrival_hour": 14,
        "day_of_week": 2,
        "month": 5,
    },
]

for vol in vols_de_test:
    print(f"Envoi du vol {vol['callsign']} dans Kafka...")
    producer.send(KAFKA_TOPIC, value=vol)
    producer.flush()
    print(f"Vol {vol['callsign']} envoyé")
    time.sleep(1)

print("Terminé.")