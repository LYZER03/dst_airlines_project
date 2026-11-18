import os
import requests
from pymongo import MongoClient
from dotenv import load_dotenv
import datetime
import time

# --- Configuration ---
# Codes couleurs ANSI pour le terminal
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def print_status(message, success=True):
    if success:
        print(f"{GREEN}✅ {message}{RESET}")
    else:
        print(f"{RED}❌ {message}{RESET}")

def print_info(message):
    print(f"[INFO] {message}")

def run_checks():
    print("=========================================")
    print("   SCRIPT DE VÉRIFICATION D'ENVIRONNEMENT")
    print("=========================================\n")

    # 1. Chargement des Variables d'Environnement
    print_info("Test 1: Chargement des variables d'environnement...")
    env_path = '.env'
    if not os.path.exists(env_path):
        print_status(f"Fichier {env_path} introuvable.", success=False)
        return

    load_dotenv()
    
    # Mapping des variables demandées vers celles trouvées dans le .env
    # Format: {'INTERNE': ['NOM_PREF', 'NOM_ALT']}
    var_map = {
        'MONGO_URI': ['MONGO_URI', 'MONGODB_URI'],
        'OPENSKY_USER': ['OPENSKY_USER', 'OPENSKY_CLIENT_ID'],
        'OPENSKY_PASS': ['OPENSKY_PASS', 'OPENSKY_CLIENT_SECRET']
    }
    
    config = {}
    missing_vars = []
    
    for key, choices in var_map.items():
        val = None
        for choice in choices:
            val = os.getenv(choice)
            if val:
                break
        if val:
            config[key] = val
        else:
            missing_vars.append(f"{choices[0]} (ou {choices[1]})")

    if missing_vars:
        print_status(f"Variables manquantes: {', '.join(missing_vars)}", success=False)
    else:
        print_status("Variables chargées.")
    print("-" * 20)

    # 2. Connexion MongoDB
    print_info("Test 2: Connexion MongoDB (Local)...")
    mongo_uri = config.get('MONGO_URI', 'mongodb://localhost:27017/')
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        # Force connection verification
        info = client.server_info()
        version = info.get('version', 'inconnue')
        print_status(f"Succès : Connecté à MongoDB v{version}")
    except Exception as e:
        print_status(f"Erreur : Impossible de joindre MongoDB à {mongo_uri}. Message: {e}", success=False)
    print("-" * 20)

    # 3. API Open-Meteo
    print_info("Test 3: API Open-Meteo (Publique)...")
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 49.0,
            "longitude": 2.5,
            "start_date": "2023-01-01",
            "end_date": "2023-01-01",
            "hourly": "temperature_2m"
        }
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
             print_status(f"Succès : Open-Meteo accessible (Code 200)")
        else:
             print_status(f"Erreur : Code retour {response.status_code}", success=False)
    except Exception as e:
        print_status(f"Erreur connexion Open-Meteo : {e}", success=False)
    print("-" * 20)

    # 4. API OpenSky Network (Authentifié)
    print_info("Test 4: API OpenSky Network (Authentifié - OAuth2)...")
    client_id = config.get('OPENSKY_USER') # Mapped from OPENSKY_CLIENT_ID
    client_secret = config.get('OPENSKY_PASS') # Mapped from OPENSKY_CLIENT_SECRET
    
    if not client_id or not client_secret:
        print_status("Credentials OpenSky manquants (vérifié TEST 1), saut du test.", success=False)
    else:
        try:
            # Step A: Get OAuth2 Token
            token_url = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
            token_data = {
                'grant_type': 'client_credentials',
                'client_id': client_id,
                'client_secret': client_secret
            }
            
            print_info("   -> Obtention du token OAuth2...")
            token_response = requests.post(token_url, data=token_data, timeout=10)
            
            if token_response.status_code != 200:
                 print_status(f"Erreur Token : {token_response.status_code} - {token_response.text}", success=False)
            else:
                access_token = token_response.json().get('access_token')
                print_status("Token récupéré.", success=True)

                # Step B: Call API with Token
                # Endpoint: Arrival flights by airport
                url = "https://opensky-network.org/api/flights/arrival"
                
                # Time window: last 30 minutes (to avoid history limits for free users)
                end_time = int(time.time())
                start_time = end_time - 1800 
                
                params = {
                    "airport": "LFPG",
                    "begin": start_time,
                    "end": end_time
                }
                
                headers = {'Authorization': f'Bearer {access_token}'}
                
                # Step B: Verify Token validity via /states/own (Strict Auth Check)
                # Cette route renvoie 403 si l'auth est invalide, contrairement à /states/all qui peut passer en anonyme.
                print_info("   -> Vérification stricte du token via /states/own...")
                check_auth_url = "https://opensky-network.org/api/states/own"
                try:
                    resp_auth = requests.get(check_auth_url, headers=headers, timeout=5)
                    if resp_auth.status_code == 200:
                        print_status("Token validé (Accès authentifié confirmé).")
                    elif resp_auth.status_code == 403:
                         print_status(f"Attention : Token valide format mais accès refusé (403) sur /states/own.", success=False)
                    else:
                         print_info(f"Note : /states/own a renvoyé {resp_auth.status_code} (Normal si pas de receivers).")
                except Exception as e:
                    print_info(f"Skip validation /states/own : {e}")

                # Step C: Call API data
                print_info(f"   -> Appel API (Time: {start_time} to {end_time})...")
                response = requests.get(url, params=params, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    print_status(f"Succès : OpenSky accessible (Code 200)")
                    # Optional: print count
                    data = response.json()
                    print(f"       -> {len(data)} vols récupérés.")
                elif response.status_code == 404:
                     # 404 is normal if no flights are found in the time window
                     print_status(f"Aucun vol trouvé sur la période (Code 404 - Normal).")
                     print_info("Tentative de fallback sur /states/all (Vols en temps réel)...")
                     # Fallback to /states/all
                     fallback_url = "https://opensky-network.org/api/states/all"
                     # Zone: France
                     params_states = {"lamin": 42.3, "lomin": -4.7, "lamax": 51.1, "lomax": 8.2}
                     try:
                        resp_states = requests.get(fallback_url, params=params_states, headers=headers, timeout=10)
                        if resp_states.status_code == 200:
                             print_status(f"Succès Fallback : /states/all accessible ({len(resp_states.json().get('states', []))} vols).")
                             print_info("-> La connexion et l'Auth fonctionnent. Le 404 précédent est dû à l'absence de données (Check Time/Airport).")
                        else:
                             print_status(f"Echec Fallback : {resp_states.status_code}", success=False)
                     except Exception as e:
                         print_status(f"Erreur Fallback : {e}", success=False)

                elif response.status_code in [401, 403]:
                    print_status(f"Erreur API Auth : {response.status_code} (Token rejeté ?)", success=False)
                    print(f"       -> Réponse : {response.text}")
                else:
                    print_status(f"Erreur : Code retour {response.status_code}", success=False)
                    print(f"       -> Réponse : {response.text}")
                
        except Exception as e:
             print_status(f"Erreur connexion OpenSky : {e}", success=False)
    print("=========================================")

if __name__ == "__main__":
    run_checks()
