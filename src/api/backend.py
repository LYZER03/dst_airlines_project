"""
Backend applicatif - DST Airlines (Chantier 3 : API interne + Sécurité)
API de données centralisant l'accès aux bases (PostgreSQL + MongoDB).
Le dashboard et tout autre client passent par elle plutôt que de se
connecter directement aux bases : pattern Backend-for-Frontend.
"""
import os
from typing import Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Query, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
import re

from sqlalchemy import create_engine, text
from pymongo import MongoClient

# Imports pour la sécurité (Rate Limiting et JWT)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import jwt

# ==========================================
# 1. CONFIGURATION BASES DE DONNÉES
# ==========================================
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_USER = os.getenv("PG_USER", "dst_admin")
PG_PASSWORD = os.getenv("PG_PASSWORD", "dst_password123")
PG_DB = os.getenv("PG_DB", "dst_db")
POSTGRES_URI = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"

MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = os.getenv("MONGO_PORT", "27017")
MONGO_DB = os.getenv("MONGO_DB", "dst_db")
MONGO_URI = f"mongodb://{MONGO_HOST}:{MONGO_PORT}/"

GOLD_TABLE = "gold_ml_features"
LIVE_TABLE = "live_predictions"

engine = create_engine(POSTGRES_URI, pool_pre_ping=True, pool_size=5)
mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)

# ==========================================
# 2. CONFIGURATION SÉCURITÉ (JWT & Rate Limit)
# ==========================================
SECRET_KEY = "super_secret_key_dst_airlines" # À cacher dans un .env en prod !
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

limiter = Limiter(key_func=get_remote_address)
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
security = HTTPBearer()

# Fausse base utilisateurs (À remplacer par une table SQL plus tard)
FAUSSE_BDD_UTILISATEURS = {
    "admin@dst.com": {"mdp": "Admin123!", "role": "admin"},
    "agent@dst.com": {"mdp": "Agent123!", "role": "user"}
}

class UserLogin(BaseModel):
    email: str = Field(..., pattern="^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    password: str = Field(..., min_length=8)

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not re.match(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)", v):
            raise ValueError("Le mot de passe doit contenir au moins 1 majuscule, 1 minuscule et 1 chiffre.")
        return v

# ==========================================
# 3. INITIALISATION DE L'APPLICATION
# ==========================================
app = FastAPI(
    title="DST Airlines - Backend API Sécurisée",
    description="API de données sécurisée : vols historiques, prédictions temps réel.",
    version="1.1.0",
)

# Ajout du Limiter et du Middleware CORS
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# ==========================================
# 4. FONCTIONS UTILITAIRES (SÉCURITÉ)
# ==========================================
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Vérifie le token JWT pour toutes les routes protégées"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        role: str = payload.get("role")
        if email is None or role is None:
            raise HTTPException(status_code=401, detail="Identifiants invalides")
        return {"email": email, "role": role}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expiré")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Token invalide")


# ==========================================
# 5. ROUTES PUBLIQUES (Ouvertes à tous)
# ==========================================
@app.get("/", tags=["Général"])
def root():
    return {"service": "DST Airlines Backend API", "version": "1.1.0", "status": "secured"}

@app.get("/health", tags=["Général"])
def health():
    pg_ok, mongo_ok = True, True
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception:
        pg_ok = False
    try:
        mongo.admin.command("ping")
    except Exception:
        mongo_ok = False
    return {"status": "ok" if (pg_ok and mongo_ok) else "degraded",
            "postgres": pg_ok, "mongodb": mongo_ok}

@app.post("/login", tags=["Authentification"])
@limiter.limit("5/minute")
async def login(request: Request, user: UserLogin):
    user_db = FAUSSE_BDD_UTILISATEURS.get(user.email)
    if not user_db or user_db["mdp"] != user.password:
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")
    
    access_token = create_access_token(data={"sub": user.email, "role": user_db["role"]})
    return {"access_token": access_token, "token_type": "bearer", "role": user_db["role"]}


# ==========================================
# 6. ROUTES PROTÉGÉES (Nécessitent un Token)
# ==========================================
# Remarque : L'ajout de `current_user: dict = Depends(get_current_user)` 
# verrouille automatiquement la route !
@app.get("/users/me", tags=["Utilisateurs"])
def read_users_me(current_user: dict = Depends(get_current_user)):
    """Retourne les infos de l'utilisateur actuellement connecté."""
    return {
        "message": "Profil récupéré avec succès",
        "email": current_user["email"], 
        "role": current_user["role"]
    }

@app.get("/admin/logs", tags=["Administration"])
def get_admin_logs(current_user: dict = Depends(get_current_user)):
    """Route strictement réservée aux administrateurs."""
    # On vérifie manuellement si le rôle est bien 'admin'
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Accès refusé. Privilèges administrateur requis."
        )
    
    return {
        "message": "Accès autorisé", 
        "logs_systeme": [
            "08:00 - Redémarrage serveur Kafka",
            "09:15 - Mise à jour du modèle XGBoost"
        ]
    }

@app.get("/flights/alerts", tags=["Temps réel"])
def get_critical_alerts(current_user: dict = Depends(get_current_user)):
    """Renvoie uniquement les vols imminents avec une forte probabilité de retard (>80%)."""
    try:
        with engine.connect() as c:
            # On filtre directement en SQL pour soulager le réseau
            rows = [dict(r._mapping) for r in c.execute(
                text(f"SELECT callsign, arrival_hour, delay_proba "
                     f"FROM {LIVE_TABLE} "
                     f"WHERE delay_proba > 0.80 "
                     f"ORDER BY delay_proba DESC LIMIT 10")
            )]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Erreur DB: {e}")
    
    return {"alert_count": len(rows), "critical_flights": rows}

@app.get("/stats/overview", tags=["Statistiques"])
def stats_overview(current_user: dict = Depends(get_current_user)):
    """KPIs globaux du dataset historique."""
    try:
        with engine.connect() as c:
            total = c.execute(text(f"SELECT COUNT(*) FROM {GOLD_TABLE}")).scalar()
            delayed = c.execute(
                text(f"SELECT COUNT(*) FROM {GOLD_TABLE} WHERE is_delayed_15 = true")
            ).scalar()
            avg_delay = c.execute(
                text(f"SELECT AVG(delay_minutes) FROM {GOLD_TABLE} WHERE delay_minutes > 0")
            ).scalar()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"PostgreSQL: {e}")
    rate = (delayed / total * 100) if total else 0
    return {"total_flights": total, "delayed_flights": delayed,
            "delay_rate_pct": round(rate, 1),
            "avg_delay_min": round(float(avg_delay), 1) if avg_delay else 0}

@app.get("/flights/historical", tags=["Vols"])
def flights_historical(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    origin: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user) # <- Route protégée !
):
    where, params = "", {"limit": limit, "offset": offset}
    if origin:
        where = "WHERE origin_airport = :origin"
        params["origin"] = origin
    try:
        with engine.connect() as c:
            rows = [dict(r._mapping) for r in c.execute(
                text(f"SELECT callsign, origin_airport, arrival_hour, "
                     f"delay_minutes, is_delayed_15, route_delay_rate_30d "
                     f"FROM {GOLD_TABLE} {where} LIMIT :limit OFFSET :offset"),
                params)]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"PostgreSQL: {e}")
    return {"limit": limit, "offset": offset, "count": len(rows), "data": rows}

@app.get("/predictions/live", tags=["Temps réel"])
def predictions_live(limit: int = Query(50, ge=1, le=200), current_user: dict = Depends(get_current_user)):
    try:
        with engine.connect() as c:
            rows = [dict(r._mapping) for r in c.execute(
                text(f"SELECT callsign, received_at, arrival_hour, visibility_m, "
                     f"wind_gusts_kmh, is_delayed, delay_proba "
                     f"FROM {LIVE_TABLE} ORDER BY received_at DESC LIMIT :limit"),
                {"limit": limit})]
    except Exception:
        return {"count": 0, "data": []}
    
    for r in rows:
        val = r.get("received_at")
        if val is not None and hasattr(val, "isoformat"):
            r["received_at"] = val.isoformat()
    return {"count": len(rows), "data": rows}

@app.get("/predictions/stats", tags=["Temps réel"])
def predictions_stats(current_user: dict = Depends(get_current_user)):
    try:
        with engine.connect() as c:
            total = c.execute(text(f"SELECT COUNT(*) FROM {LIVE_TABLE}")).scalar()
            delayed = c.execute(
                text(f"SELECT COUNT(*) FROM {LIVE_TABLE} WHERE is_delayed = true")
            ).scalar()
    except Exception:
        return {"total": 0, "delayed": 0, "delay_rate_pct": 0}
    rate = (delayed / total * 100) if total else 0
    return {"total": total, "delayed": delayed, "delay_rate_pct": round(rate, 1)}

@app.get("/weather/latest", tags=["Météo"])
def weather_latest(current_user: dict = Depends(get_current_user)):
    try:
        db = mongo[MONGO_DB]
        col = next((c for c in db.list_collection_names()
                    if "weather" in c.lower() or "meteo" in c.lower()), None)
        if not col:
            return {"warning": "Aucune collection météo", "data": None}
        doc = db[col].find_one({}, {"_id": 0}, sort=[("_id", -1)])
        return {"collection": col, "data": doc}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MongoDB: {e}")