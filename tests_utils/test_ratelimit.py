import requests
import time

url = "http://localhost:8001/login"
# On envoie un payload au format attendu (JSON)
payload = {
    "email": "admin@dst.com", 
    "password": "Admin123!"
}

print("Lancement de l'attaque par force brute...\n")

for i in range(1, 8):
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print(f"Essai {i} : ✅ Succès (Code 200)")
    elif response.status_code == 429:
        print(f"Essai {i} : 🚨 BLOQUÉ par le Rate Limit ! (Code 429)")
    else:
        print(f"Essai {i} : Erreur (Code {response.status_code})")
        
    time.sleep(0.2)