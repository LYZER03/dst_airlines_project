from kafka import KafkaConsumer
import json

# Création du Consumer
consumer = KafkaConsumer(
    'live_flights_topic',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest' # Lit les messages depuis le début
)

print("En écoute des nouveaux vols...")

for message in consumer:
    print(f" Vol reçu via Kafka : {message.value}")