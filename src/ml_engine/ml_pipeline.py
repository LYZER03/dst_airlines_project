"""
DST Airlines — ml_pipeline.py

Pipeline ML de production — lit depuis PostgreSQL (gold_ml_features)

Usage :
    cd ~/dst_airlines_project
    source venv/bin/activate
    python3 ml_pipeline.py

Produit :
    models/xgb_classifier.pkl   → Modèle de classification (is_delayed_sim)
    models/xgb_regressor.pkl    → Modèle de régression (delay_minutes)
    model_metrics.json          → Métriques finales exportées
    MLflow                      → Tracking dans mlflow.db (Run loggué)
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sqlalchemy import create_engine
from sqlalchemy import text

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score,
    classification_report
)
from xgboost import XGBClassifier, XGBRegressor
from mlflow.models.signature import infer_signature
from scipy.stats import randint, uniform

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

POSTGRES_URI  = "postgresql://dst_admin:dst_password123@localhost:5432/dst_db"
MLFLOW_URI    = "sqlite:///mlflow.db"
MODELS_DIR    = "models"
RANDOM_STATE  = 42

# Split temporel : Jan-Sep = Train | Oct-Déc = Test
TRAIN_MONTHS  = list(range(1, 9))   
TEST_MONTHS   = [9]  

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ml_pipeline.log", mode="w"),
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT DEPUIS POSTGRESQL
# ─────────────────────────────────────────────────────────────────────────────

def load_from_postgres() -> pd.DataFrame:
    log.info("1. Chargement depuis PostgreSQL")
    engine = create_engine(POSTGRES_URI)

    # Vérification de la connexion
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM gold_ml_features")).scalar()
        log.info(f"   Connexion OK — gold_ml_features : {count:,} lignes")

    query = "SELECT * FROM gold_ml_features;"
    df = pd.read_sql(query, engine)

    # Normaliser les noms de colonnes (enlever les préfixes metadata./weather./etc.)
    df.columns = [
        c.replace("metadata.", "")
         .replace("weather.", "w_")
         .replace("flight_info.", "fi_")
         .replace("temporal_features.", "t_")
         .replace("context.", "ctx_")
        for c in df.columns
    ]

    # Assurer le type datetime sur le timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Assurer le type datetime sur arrival_ts si présent (issu de build_gold.py)
    if "arrival_ts" in df.columns:
        df["arrival_ts"] = pd.to_datetime(df["arrival_ts"])
        df = df.sort_values("arrival_ts").reset_index(drop=True)
        # Créer timestamp depuis arrival_ts si absent
        if "timestamp" not in df.columns:
            df["timestamp"] = df["arrival_ts"]

    log.info(f"   Colonnes disponibles : {list(df.columns)}")
    log.info(f"   Période : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    log.info(f"   Shape   : {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONSTRUCTION DE LA CIBLE (is_delayed_sim)
# ─────────────────────────────────────────────────────────────────────────────

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    log.info("2. Construction des cibles ML━")

    # Détecter les colonnes disponibles selon la source (CSV ou build_gold.py)
    has_visibility  = any(c for c in df.columns if "visibility" in c)
    has_wind_gusts  = any(c for c in df.columns if "wind_gust" in c or "gusts" in c)
    has_hourly      = any(c for c in df.columns if "hourly_flight" in c or "hourly" in c)
    has_holiday     = any(c for c in df.columns if "holiday" in c)
    has_weekend     = any(c for c in df.columns if "weekend" in c)

    # Noms réels des colonnes (compatibles build_gold.py ET CSV)
    col_visibility = next((c for c in df.columns if "visibility" in c), None)
    col_gusts      = next((c for c in df.columns if "gusts" in c or "wind_gust" in c), None)
    col_hourly     = next((c for c in df.columns if "hourly_flight" in c or ("hourly" in c and "count" in c)), None)
    col_holiday    = next((c for c in df.columns if "holiday" in c), None)
    col_weekend    = next((c for c in df.columns if "weekend" in c), None)
    col_condition  = next((c for c in df.columns if "condition" in c), None)

    np.random.seed(RANDOM_STATE)
    risk = np.zeros(len(df))

    # Météo — visibilité (Variable Reine)
    if col_visibility:
        risk += (df[col_visibility] < 600).astype(float) * 0.40
        risk += ((df[col_visibility] >= 600) & (df[col_visibility] < 1500)).astype(float) * 0.20

    # Météo — rafales
    if col_gusts:
        risk += (df[col_gusts] > 70).astype(float) * 0.25
        risk += ((df[col_gusts] > 50) & (df[col_gusts] <= 70)).astype(float) * 0.10

    # Météo — condition
    if col_condition:
        risk += df[col_condition].isin(["Foggy", "Snowy", "Stormy"]).astype(float) * 0.15

    # Trafic CDG (Feature Killer)
    if col_hourly:
        q75 = df[col_hourly].quantile(0.75)
        q90 = df[col_hourly].quantile(0.90)
        risk += (df[col_hourly] >= q90).astype(float) * 0.30
        risk += ((df[col_hourly] >= q75) & (df[col_hourly] < q90)).astype(float) * 0.15

    # Temporel
    if col_holiday:
        risk += df[col_holiday].astype(float) * 0.10
    if col_weekend:
        risk += df[col_weekend].astype(float) * 0.05

    # Seuil calibré à ~23% (Eurocontrol 2023)
    threshold = np.percentile(risk, 77)
    df["is_delayed_sim"] = (risk >= threshold).astype(int)
    df["risk_score"]     = risk

    # Cible régression enrichie
    noise = np.where(
        df["is_delayed_sim"] == 1,
        np.random.exponential(scale=18, size=len(df)) + 10,
        np.random.uniform(-8, 12, size=len(df))
    )
    # Utiliser delay_minutes existant si disponible
    col_delay = next((c for c in df.columns if c in ["fi_delay_minutes", "delay_minutes"]), None)
    base_delay = df[col_delay].values if col_delay else np.zeros(len(df))
    df["delay_minutes_ml"] = np.clip(
        base_delay + noise * df["is_delayed_sim"] * 0.7,
        -15, 120
    )

    pct = df["is_delayed_sim"].mean() * 100
    log.info(f"   is_delayed_sim — Retardés : {df['is_delayed_sim'].sum():,} ({pct:.1f}%)")
    log.info(f"   delay_minutes_ml — Moy : {df['delay_minutes_ml'].mean():.1f} min")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    log.info("3. Feature Engineering")

    # Détecter les colonnes temporelles
    col_hour    = next((c for c in df.columns if "arrival_hour" in c and "sin" not in c and "cos" not in c), None)
    col_dow     = next((c for c in df.columns if "day_of_week" in c and "sin" not in c and "cos" not in c), None)
    col_month   = next((c for c in df.columns if c in ["t_month", "month"] and "sin" not in c), None)

    # Encodage cyclique (évite la discontinuité 23h→0h, dim→lun)
    if col_hour:
        df["arrival_hour_sin"] = np.sin(2 * np.pi * df[col_hour] / 24)
        df["arrival_hour_cos"] = np.cos(2 * np.pi * df[col_hour] / 24)
    if col_dow:
        df["day_sin"] = np.sin(2 * np.pi * df[col_dow] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df[col_dow] / 7)
    if col_month:
        df["month_sin"] = np.sin(2 * np.pi * df[col_month] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df[col_month] / 12)

    # Features opérationnelles CDG
    col_visibility = next((c for c in df.columns if "visibility" in c), None)
    col_gusts      = next((c for c in df.columns if "gusts" in c or "wind_gust" in c), None)

    df["is_lvp"]       = (df[col_visibility] < 600).astype(int) if col_visibility else 0
    df["is_high_wind"] = (df[col_gusts] > 55).astype(int)       if col_gusts else 0
    if col_hour:
        df["is_peak_hour"] = df[col_hour].isin([6,7,8,9,17,18,19,20]).astype(int)

    # Features P1 — taux retard par appareil et par route (rolling avec shift anti-leakage)
    col_icao   = next((c for c in df.columns if "icao24" in c), None)
    col_origin = next((c for c in df.columns if "origin_airport" in c or "origin" in c), None)

    if col_icao:
        df["aircraft_delay_rate_7d"] = (
            df.groupby(col_icao)["is_delayed_sim"]
              .transform(lambda x: x.shift(1).rolling(50, min_periods=3).mean())
              .fillna(df["is_delayed_sim"].mean())
        )
    else:
        df["aircraft_delay_rate_7d"] = df["is_delayed_sim"].mean()

    if col_origin:
        df["route_delay_rate_30d"] = (
            df.groupby(col_origin)["is_delayed_sim"]
              .transform(lambda x: x.shift(1).rolling(200, min_periods=10).mean())
              .fillna(df["is_delayed_sim"].mean())
        )
    else:
        df["route_delay_rate_30d"] = df["is_delayed_sim"].mean()

    log.info(f"   Features créées : cycliques, is_lvp, is_high_wind, is_peak_hour, P1 (aircraft/route)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. SPLIT TEMPOREL
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame):
    log.info("4. Split temporel Train / Test")

    train_mask = df["timestamp"].dt.month.isin(TRAIN_MONTHS)
    test_mask  = df["timestamp"].dt.month.isin(TEST_MONTHS)

    df_train = df[train_mask].copy()
    df_test  = df[test_mask].copy()

    log.info(f"   Train (Jan-Sep) : {len(df_train):,} vols | "
             f"{df_train['timestamp'].min().date()} → {df_train['timestamp'].max().date()}")
    log.info(f"   Test  (Oct-Déc) : {len(df_test):,}  vols | "
             f"{df_test['timestamp'].min().date()} → {df_test['timestamp'].max().date()}")
    log.info(f"   Taux retard train : {df_train['is_delayed_sim'].mean()*100:.1f}% | "
             f"test : {df_test['is_delayed_sim'].mean()*100:.1f}%")

    return df_train, df_test


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRÉPARATION DES MATRICES X / Y
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame):
    """Sélectionne les features disponibles dynamiquement selon les colonnes."""

    # Features numériques (celles qui existent dans le DataFrame)
    candidates_num = [
        "w_visibility_m", "visibility_m",
        "w_wind_gusts_kmh", "wind_gusts_kmh",
        "w_wind_kmh", "wind_kmh",
        "w_temp_c", "temp_c",
        "ctx_hourly_flight_count", "hourly_flight_count",
        "ctx_daily_flight_count", "daily_flight_count",
        "fi_arrival_hour", "arrival_hour",
        "fi_duration_min", "duration_min",
        "arrival_hour_sin", "arrival_hour_cos",
        "day_sin", "day_cos",
        "month_sin", "month_cos",
        "aircraft_delay_rate_7d", "route_delay_rate_30d",
        "risk_score",
    ]
    candidates_bool = [
        "t_is_weekend", "is_weekend",
        "t_is_holiday", "is_holiday",
        "is_lvp", "is_high_wind", "is_peak_hour",
    ]
    candidates_ordinal = ["w_condition", "condition"]
    candidates_ohe     = ["origin_airport"]

    num_features  = [c for c in candidates_num      if c in df.columns]
    bool_features = [c for c in candidates_bool     if c in df.columns]
    ord_features  = [c for c in candidates_ordinal  if c in df.columns]
    ohe_features  = [c for c in candidates_ohe      if c in df.columns]

    # Dédoublonner (garder le premier nom trouvé)
    num_features  = list(dict.fromkeys(num_features))
TRAIN_MONTHS  = list(range(1, 9))    # Jan-Août (8 mois)
TEST_MONTHS   = [9]      bool_features = list(dict.fromkeys(bool_features))
    ord_features  = ord_features[:1]
    ohe_features  = ohe_features[:1]

    all_features = num_features + bool_features + ord_features + ohe_features

    log.info(f"   Numériques   : {len(num_features)} features")
    log.info(f"   Booléennes   : {len(bool_features)} features")
    log.info(f"   OrdinalEnc   : {ord_features}")
    log.info(f"   OneHotEnc    : {ohe_features} (top 30 catégories)")
    log.info(f"   Total        : {len(all_features)} features")

    return num_features, bool_features, ord_features, ohe_features, all_features


def build_preprocessor(num_features, bool_features, ord_features, ohe_features):
    CONDITION_ORDER = [["Clear", "Cloudy", "Rainy", "Foggy", "Snowy", "Stormy"]]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    bool_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    ordinal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=CONDITION_ORDER,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])
    ohe_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            max_categories=30,
        )),
    ])

    transformers = []
    if num_features:  transformers.append(("num",     num_pipe,     num_features))
    if bool_features: transformers.append(("bool",    bool_pipe,    bool_features))
    if ord_features:  transformers.append(("ordinal", ordinal_pipe, ord_features))
    if ohe_features:  transformers.append(("ohe",     ohe_pipe,     ohe_features))

    return ColumnTransformer(transformers, remainder="drop")


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRAÎNEMENT — CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def train_classification(X_train, y_train, X_test, y_test, preprocessor, class_weight):
    log.info("6. Classification (is_delayed_sim)")

    # ── Random Forest ────────────────────────────────────────────────────────
    rf_pipe = Pipeline([
        ("prep",  preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    rf_pipe.fit(X_train, y_train)
    rf_preds  = rf_pipe.predict(X_test)
    rf_probas = rf_pipe.predict_proba(X_test)[:, 1]
    rf_f1     = f1_score(y_test, rf_preds)
    log.info(f"   Random Forest     → F1={rf_f1:.4f} | "
             f"Précision={precision_score(y_test, rf_preds):.4f} | "
             f"Rappel={recall_score(y_test, rf_preds):.4f}")

    # ── XGBoost avec optimisation RandomizedSearchCV + TimeSeriesSplit ───────
    log.info("   XGBoost — Optimisation (30 iter × TimeSeriesSplit-5)...")

    param_dist = {
        "model__n_estimators":     randint(100, 500),
        "model__max_depth":        randint(3, 9),
        "model__learning_rate":    uniform(0.01, 0.19),
        "model__subsample":        uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__min_child_weight": randint(1, 8),
        "model__gamma":            uniform(0, 0.4),
        "model__reg_alpha":        uniform(0, 0.3),
        "model__reg_lambda":       uniform(0.5, 2.0),
    }

    xgb_pipe = Pipeline([
        ("prep",  preprocessor),
        ("model", XGBClassifier(
            scale_pos_weight=class_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
        )),
    ])

    search = RandomizedSearchCV(
        xgb_pipe,
        param_distributions=param_dist,
        n_iter=30,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best_xgb   = search.best_estimator_
    xgb_preds  = best_xgb.predict(X_test)
    xgb_probas = best_xgb.predict_proba(X_test)[:, 1]

    metrics_clf = {
        "f1":        f1_score(y_test, xgb_preds),
        "precision": precision_score(y_test, xgb_preds),
        "recall":    recall_score(y_test, xgb_preds),
        "accuracy":  accuracy_score(y_test, xgb_preds),
        "auc_roc":   roc_auc_score(y_test, xgb_probas),
    }

    log.info(f"   XGBoost optimisé  → F1={metrics_clf['f1']:.4f} | "
             f"Précision={metrics_clf['precision']:.4f} | "
             f"Rappel={metrics_clf['recall']:.4f} | "
             f"AUC={metrics_clf['auc_roc']:.4f}")

    # KPI check
    kpis = [
        ("F1 ≥ 70%",        metrics_clf["f1"]        >= 0.70),
        ("Précision ≥ 72%", metrics_clf["precision"] >= 0.72),
        ("Rappel ≥ 68%",    metrics_clf["recall"]    >= 0.68),
    ]
    for name, ok in kpis:
        log.info(f"   KPI {name}: {'' if ok else 'En cours'}")

    log.info("\n" + classification_report(y_test, xgb_preds,
                                          target_names=["À l'heure", "Retardé"]))

    return best_xgb, metrics_clf, search.best_params_


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENTRAÎNEMENT — RÉGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def train_regression(X_train, y_train, X_test, y_test, preprocessor):
    log.info("7. Régression (delay_minutes)")

    # ── Random Forest ────────────────────────────────────────────────────────
    rf_pipe = Pipeline([
        ("prep",  preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    rf_pipe.fit(X_train, y_train)
    rf_preds  = rf_pipe.predict(X_test)
    rf_mae    = mean_absolute_error(y_test, rf_preds)
    log.info(f"   Random Forest → MAE={rf_mae:.3f}min | "
             f"RMSE={np.sqrt(mean_squared_error(y_test, rf_preds)):.3f}min | "
             f"R²={r2_score(y_test, rf_preds):.4f}")

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb_pipe = Pipeline([
        ("prep",  preprocessor),
        ("model", XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=RANDOM_STATE,
            eval_metric="mae",
            verbosity=0,
        )),
    ])
    xgb_pipe.fit(X_train, y_train)
    xgb_preds = xgb_pipe.predict(X_test)

    metrics_reg = {
        "mae":  mean_absolute_error(y_test, xgb_preds),
        "rmse": np.sqrt(mean_squared_error(y_test, xgb_preds)),
        "r2":   r2_score(y_test, xgb_preds),
    }
    log.info(f"   XGBoost       → MAE={metrics_reg['mae']:.3f}min | "
             f"RMSE={metrics_reg['rmse']:.3f}min | "
             f"R²={metrics_reg['r2']:.4f}")

    # Retourner le meilleur (MAE la plus basse)
    best_pipe = xgb_pipe if metrics_reg["mae"] <= rf_mae else rf_pipe
    log.info(f"   Retenu : {'XGBoost' if best_pipe is xgb_pipe else 'Random Forest'}")

    return best_pipe, metrics_reg


# ─────────────────────────────────────────────────────────────────────────────
# 8. SAUVEGARDE — JOBLIB + MLFLOW
# ─────────────────────────────────────────────────────────────────────────────

def save_models(clf_pipe, reg_pipe, metrics_clf, metrics_reg,
                best_params, X_train, n_train, n_test):
    log.info("8. Sauvegarde des modèles")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Joblib
    clf_path = os.path.join(MODELS_DIR, "xgb_classifier.pkl")
    reg_path = os.path.join(MODELS_DIR, "xgb_regressor.pkl")
    joblib.dump(clf_pipe, clf_path)
    joblib.dump(reg_pipe, reg_path)
    log.info(f"   Sauvegardé : {clf_path}")
    log.info(f"   Sauvegardé : {reg_path}")

    # model_metrics.json
    all_metrics = {
        "classification": metrics_clf,
        "regression":     metrics_reg,
        "split":          "temporal (Jan-Sep train | Oct-Déc test)",
        "train_size":     n_train,
        "test_size":      n_test,
    }
    with open("model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("   Sauvegardé : model_metrics.json")

    # MLflow
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("dst_airlines_classification")

        with mlflow.start_run(run_name="XGBoost_TemporalSplit_Production") as run:

            # Paramètres
            mlflow.log_params({
                k.replace("model__", ""): v
                for k, v in best_params.items()
            })
            mlflow.log_param("split_type",   "temporal")
            mlflow.log_param("train_period", "Jan-Sep 2025")
            mlflow.log_param("test_period",  "Oct-Dec 2025")
            mlflow.log_param("train_size",   n_train)
            mlflow.log_param("test_size",    n_test)
            mlflow.log_param("cv_strategy",  "TimeSeriesSplit(5)")
            mlflow.log_param("source",       "PostgreSQL gold_ml_features")

            # Métriques
            mlflow.log_metrics({
                "f1_score":  metrics_clf["f1"],
                "precision": metrics_clf["precision"],
                "recall":    metrics_clf["recall"],
                "accuracy":  metrics_clf["accuracy"],
                "auc_roc":   metrics_clf["auc_roc"],
                "reg_mae":   metrics_reg["mae"],
                "reg_rmse":  metrics_reg["rmse"],
                "reg_r2":    metrics_reg["r2"],
            })

            # Modèle
            signature = infer_signature(X_train, clf_pipe.predict(X_train))
            mlflow.sklearn.log_model(
                clf_pipe, "model",
                signature=signature,
                registered_model_name="dst_airlines_delay_classifier",
            )
            mlflow.log_artifact("model_metrics.json")

            log.info(f"   MLflow Run ID : {run.info.run_id}")
            log.info("   Modèle enregistré : dst_airlines_delay_classifier (Production)")

    except Exception as e:
        log.warning(f"   MLflow non disponible : {e}")
        log.warning("   → Modèles sauvegardés localement uniquement")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("DST Airlines — Pipeline ML (Split Temporel)")
    log.info("  Source    : PostgreSQL gold_ml_features")
    log.info("  Train     : Janvier → Septembre 2025")
    log.info("  Test      : Octobre → Décembre 2025")
    log.info("=" * 60)

    # 1. Chargement
    df = load_from_postgres()

    # 2. Cibles
    df = build_targets(df)

    # 3. Feature Engineering
    df = feature_engineering(df)

    # 4. Split temporel
    df_train, df_test = temporal_split(df)

    # 5. Préparation des features
    log.info("5. Préparation des features")
    num_f, bool_f, ord_f, ohe_f, all_f = prepare_features(df_train)
    preprocessor = build_preprocessor(num_f, bool_f, ord_f, ohe_f)

    # Convertir booléens
    for c in bool_f:
        df_train[c] = df_train[c].astype(int)
        df_test[c]  = df_test[c].astype(int)

    X_train = df_train[all_f]
    X_test  = df_test[all_f]
    y_clf_train = df_train["is_delayed_sim"].values
    y_clf_test  = df_test["is_delayed_sim"].values
    y_reg_train = df_train["delay_minutes_ml"].values
    y_reg_test  = df_test["delay_minutes_ml"].values

    class_weight = (y_clf_train == 0).sum() / (y_clf_train == 1).sum()
    log.info(f"   X_train : {X_train.shape} | X_test : {X_test.shape}")
    log.info(f"   scale_pos_weight : {class_weight:.2f}")

    # 6. Classification
    clf_pipe, metrics_clf, best_params = train_classification(
        X_train, y_clf_train, X_test, y_clf_test, preprocessor, class_weight
    )

    # 7. Régression
    reg_pipe, metrics_reg = train_regression(
        X_train, y_reg_train, X_test, y_reg_test, preprocessor
    )

    # 8. Sauvegarde
    save_models(
        clf_pipe, reg_pipe, metrics_clf, metrics_reg,
        best_params, X_train, len(X_train), len(X_test)
    )

    log.info("=" * 60)
    log.info("Pipeline ML terminé avec succès")
    log.info(f"   F1-score (test Oct-Déc) : {metrics_clf['f1']:.4f}")
    log.info(f"   MAE régression          : {metrics_reg['mae']:.3f} min")
    log.info(f"   Modèles dans            : {MODELS_DIR}/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
