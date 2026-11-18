"""
DST Airlines - Livrable 5
Entraînement des modèles de prédiction de retard (classification + régression).

Pipeline :
  1. Chargement de gold_ml_features depuis PostgreSQL
  2. Construction des cibles (is_delayed_sim, delay_minutes_ml)
  3. Feature engineering (encodage cyclique, flags CDG, taux historiques)
  4. Split temporel Train (Jan-Août) / Test (Septembre)
  5. Comparaison Random Forest vs XGBoost
  6. Optimisation XGBoost (RandomizedSearchCV + TimeSeriesSplit)
  7. Export : modèles .pkl, model_metrics.json, model_meta.json, figures
  8. Versioning MLflow (tracking + registre de modèles)

Lancement sur la VM : python3 train_model.py
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')  # pas d'affichage : la VM n'a pas d'environnement graphique
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)
from xgboost import XGBClassifier, XGBRegressor
from scipy.stats import randint, uniform

warnings.filterwarnings('ignore')

POSTGRES_URI = 'postgresql://dst_admin:dst_password123@localhost:5432/dst_db'
RANDOM_STATE = 42
TRAIN_MONTHS = list(range(1, 9))   # Janvier - Août
TEST_MONTHS  = [9]                 # Septembre (dernier mois complet)
OUTPUT_DIR   = 'models'

BLUE, LBLUE  = '#4a6fa5', '#9db8e0'   # bleus pastel (neutre, train)
GREEN, LGREEN = '#6bbf9a', '#8fd3b6'  # verts pastel (situation normale)
ORANGE, RED   = '#ef9b4f', '#e35d5b'  # vigilance / alerte

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#f8f9fc',
    'axes.grid': True, 'grid.color': '#e8ecf4', 'grid.linewidth': 0.8,
    'axes.edgecolor': '#e3e7f0', 'text.color': '#33415c',
    'axes.labelcolor': '#33415c', 'xtick.color': '#5a6781',
    'ytick.color': '#5a6781',
    'font.family': 'DejaVu Sans', 'axes.titlesize': 12, 'axes.labelsize': 11,
})


# ------------------------------------------------------------------
# 1. Chargement des données
# ------------------------------------------------------------------

def load_data():
    engine = create_engine(POSTGRES_URI)
    query = "SELECT * FROM gold_ml_features;"
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    print(f'Connexion PostgreSQL OK - {len(df):,} vols chargés')
    return df


def find_col(df, *candidates):
    """Retourne le premier nom de colonne trouvé dans df, sinon None.
    Le pipeline supporte deux conventions de nommage (build_gold.py / CSV préfixé)."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def map_columns(df):
    cols = {
        'ts':         find_col(df, 'arrival_ts', 'timestamp'),
        'hour':       find_col(df, 'arrival_hour', 'fi_arrival_hour'),
        'visibility': find_col(df, 'visibility_m', 'w_visibility_m'),
        'gusts':      find_col(df, 'wind_gusts_kmh', 'w_wind_gusts_kmh'),
        'wind':       find_col(df, 'wind_kmh', 'w_wind_kmh'),
        'temp':       find_col(df, 'temp_c', 'w_temp_c'),
        'condition':  find_col(df, 'condition', 'w_condition'),
        'hourly':     find_col(df, 'hourly_flight_count', 'ctx_hourly_flight_count'),
        'daily':      find_col(df, 'daily_flight_count', 'ctx_daily_flight_count'),
        'dow':        find_col(df, 'day_of_week', 't_day_of_week'),
        'month':      find_col(df, 'month', 't_month'),
        'weekend':    find_col(df, 'is_weekend', 't_is_weekend'),
        'holiday':    find_col(df, 'is_holiday', 't_is_holiday'),
        'duration':   find_col(df, 'duration_min', 'fi_duration_min'),
        'origin':     find_col(df, 'origin_airport', 'metadata.origin_airport'),
        'icao24':     find_col(df, 'icao24', 'metadata.icao24'),
        'delay':      find_col(df, 'delay_minutes', 'fi_delay_minutes'),
    }
    return cols


# ------------------------------------------------------------------
# 2. Cibles ML
# ------------------------------------------------------------------

def build_targets(df, C):
    """Construit is_delayed_sim (score de risque opérationnel calibré à ~23%
    de retards, référence Eurocontrol 2023) et delay_minutes_ml.
    Le risk_score n'est pas réutilisé comme feature (data leakage)."""
    np.random.seed(RANDOM_STATE)
    risk = np.zeros(len(df))

    # Visibilité : LVP (Low Visibility Procedure) sous 600 m à CDG
    if C['visibility']:
        risk += (df[C['visibility']] < 600).astype(float) * 0.40
        risk += ((df[C['visibility']] >= 600) & (df[C['visibility']] < 1500)).astype(float) * 0.20

    # Rafales : > 70 km/h on approche des limites opérationnelles
    if C['gusts']:
        risk += (df[C['gusts']] > 70).astype(float) * 0.25
        risk += ((df[C['gusts']] > 50) & (df[C['gusts']] <= 70)).astype(float) * 0.10

    if C['condition']:
        risk += df[C['condition']].isin(['Foggy', 'Snowy', 'Stormy']).astype(float) * 0.15

    # Saturation du trafic à CDG (au-delà du 90e percentile : retards en cascade)
    if C['hourly']:
        q75 = df[C['hourly']].quantile(0.75)
        q90 = df[C['hourly']].quantile(0.90)
        risk += (df[C['hourly']] >= q90).astype(float) * 0.30
        risk += ((df[C['hourly']] >= q75) & (df[C['hourly']] < q90)).astype(float) * 0.15

    if C['holiday']:
        risk += df[C['holiday']].astype(float) * 0.10
    if C['weekend']:
        risk += df[C['weekend']].astype(float) * 0.05

    # Aléas opérationnels non observés (rotations, ATC, technique) :
    # évite une cible parfaitement déterministe vis-à-vis des features
    risk += np.random.normal(0, 0.08, len(df))

    threshold = np.percentile(risk, 77)
    df['is_delayed_sim'] = (risk >= threshold).astype(int)

    # Cible de régression : retard en base + bruit réaliste
    base = df[C['delay']].values if C['delay'] else np.zeros(len(df))
    noise = np.where(
        df['is_delayed_sim'] == 1,
        np.random.exponential(scale=18, size=len(df)) + 10,
        np.random.uniform(-8, 12, size=len(df))
    )
    df['delay_minutes_ml'] = np.clip(base + noise * df['is_delayed_sim'] * 0.7, -15, 120)

    vc = df['is_delayed_sim'].value_counts()
    print(f'Cibles : {vc[1]:,} retardés ({vc[1]/len(df)*100:.1f}%) / '
          f'{vc[0]:,} à l\'heure ({vc[0]/len(df)*100:.1f}%)')
    return df


# ------------------------------------------------------------------
# 3. Feature engineering
# ------------------------------------------------------------------

def build_features(df, C):
    # Encodage cyclique : 23h et 0h sont voisines, le sin/cos les rapproche
    if C['hour']:
        df['arrival_hour_sin'] = np.sin(2 * np.pi * df[C['hour']] / 24)
        df['arrival_hour_cos'] = np.cos(2 * np.pi * df[C['hour']] / 24)
    if C['dow']:
        df['day_sin'] = np.sin(2 * np.pi * df[C['dow']] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df[C['dow']] / 7)
    if C['month']:
        df['month_sin'] = np.sin(2 * np.pi * df[C['month']] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df[C['month']] / 12)

    # Flags opérationnels CDG
    df['is_lvp']       = (df[C['visibility']] < 600).astype(int) if C['visibility'] else 0
    df['is_high_wind'] = (df[C['gusts']] > 55).astype(int) if C['gusts'] else 0
    df['is_peak_hour'] = (df[C['hour']].isin([6, 7, 8, 9, 17, 18, 19, 20]).astype(int)
                          if C['hour'] else 0)

    # Taux de retard historiques (rolling). shift(1) exclut le vol courant
    # pour éviter le leakage ; fallback sur le taux global si historique court.
    df['delayed_proxy'] = df['is_delayed_sim'].copy()
    global_rate = df['is_delayed_sim'].mean()

    if C['icao24']:
        df['aircraft_delay_rate_7d'] = (
            df.groupby(C['icao24'])['delayed_proxy']
              .transform(lambda x: x.shift(1).rolling(50, min_periods=3).mean())
              .fillna(global_rate)
        )
    else:
        df['aircraft_delay_rate_7d'] = global_rate

    if C['origin']:
        df['route_delay_rate_30d'] = (
            df.groupby(C['origin'])['delayed_proxy']
              .transform(lambda x: x.shift(1).rolling(200, min_periods=10).mean())
              .fillna(global_rate)
        )
    else:
        df['route_delay_rate_30d'] = global_rate

    df = df.drop(columns=['delayed_proxy'])
    print(f'Feature engineering terminé - {df.shape[1]} colonnes')
    return df


def define_feature_sets(df, C):
    num_features = [c for c in [
        C['visibility'], C['gusts'], C['wind'], C['temp'],
        C['hourly'], C['daily'], C['hour'], C['duration'],
        'arrival_hour_sin', 'arrival_hour_cos',
        'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'aircraft_delay_rate_7d', 'route_delay_rate_30d',
    ] if c is not None and c in df.columns]

    bool_features = [c for c in [
        C['weekend'], C['holiday'], 'is_lvp', 'is_high_wind', 'is_peak_hour',
    ] if c is not None and c in df.columns]

    cat_ordinal = [C['condition']] if C['condition'] and C['condition'] in df.columns else []
    cat_ohe     = [C['origin']]    if C['origin']    and C['origin']    in df.columns else []

    return num_features, bool_features, cat_ordinal, cat_ohe


def build_preprocessor(num_features, bool_features, cat_ordinal, cat_ohe):
    """Les scalers/encodeurs sont fittés sur X_train uniquement (anti-leakage)."""
    condition_order = [['Clear', 'Cloudy', 'Rainy', 'Foggy', 'Snowy', 'Stormy']]

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])
    bool_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])
    ordinal_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=condition_order,
                                   handle_unknown='use_encoded_value',
                                   unknown_value=-1)),
    ])
    ohe_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False,
                                  handle_unknown='ignore',
                                  max_categories=30)),
    ])

    transformers = [
        ('num',  num_pipe,  num_features),
        ('bool', bool_pipe, bool_features),
    ]
    if cat_ordinal:
        transformers.append(('ordinal', ordinal_pipe, cat_ordinal))
    if cat_ohe:
        transformers.append(('ohe', ohe_pipe, cat_ohe))

    return ColumnTransformer(transformers, remainder='drop')


# ------------------------------------------------------------------
# 4 à 6. Split, entraînement, optimisation
# ------------------------------------------------------------------

def temporal_split(df, C):
    df[C['ts']] = pd.to_datetime(df[C['ts']])
    df = df.sort_values(C['ts']).reset_index(drop=True)

    train_mask = df[C['ts']].dt.month.isin(TRAIN_MONTHS)
    test_mask  = df[C['ts']].dt.month.isin(TEST_MONTHS)
    df_train, df_test = df[train_mask].copy(), df[test_mask].copy()

    print(f'Split temporel : train {len(df_train):,} vols (Jan-Août) / '
          f'test {len(df_test):,} vols (Sep)')
    return df, df_train, df_test


def train_classifiers(preprocessor, X_train, y_train, X_test, y_test, class_weight):
    results = {}
    for name, model in [
        ('Random Forest', RandomForestClassifier(
            n_estimators=200, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1)),
        ('XGBoost', XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            scale_pos_weight=class_weight, random_state=RANDOM_STATE,
            eval_metric='logloss', verbosity=0)),
    ]:
        pipe = Pipeline([('prep', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        preds  = pipe.predict(X_test)
        probas = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            'f1':        f1_score(y_test, preds),
            'precision': precision_score(y_test, preds),
            'recall':    recall_score(y_test, preds),
            'accuracy':  accuracy_score(y_test, preds),
            'auc_roc':   roc_auc_score(y_test, probas),
            'pipeline':  pipe, 'preds': preds, 'probas': probas,
        }
        print(f'  {name:<15} F1={results[name]["f1"]:.4f} '
              f'AUC={results[name]["auc_roc"]:.4f}')
    return results


def train_regressors(preprocessor, X_train, y_train, X_test, y_test):
    results = {}
    for name, model in [
        ('Random Forest', RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ('XGBoost', XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=RANDOM_STATE, eval_metric='mae', verbosity=0)),
    ]:
        pipe = Pipeline([('prep', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = {
            'mae':  mean_absolute_error(y_test, preds),
            'rmse': float(np.sqrt(mean_squared_error(y_test, preds))),
            'r2':   r2_score(y_test, preds),
            'pipeline': pipe, 'preds': preds,
        }
        print(f'  {name:<15} MAE={results[name]["mae"]:.2f} min '
              f'R2={results[name]["r2"]:.4f}')
    return results


def optimize_xgboost(preprocessor, X_train, y_train, class_weight):
    """RandomizedSearchCV sur 30 combinaisons, validation croisée TimeSeriesSplit
    (5 folds) pour respecter l'ordre chronologique du train."""
    param_dist = {
        'model__n_estimators':     randint(100, 500),
        'model__max_depth':        randint(3, 9),
        'model__learning_rate':    uniform(0.01, 0.19),
        'model__subsample':        uniform(0.6, 0.4),
        'model__colsample_bytree': uniform(0.6, 0.4),
        'model__min_child_weight': randint(1, 8),
        'model__gamma':            uniform(0, 0.4),
        'model__reg_alpha':        uniform(0, 0.3),
        'model__reg_lambda':       uniform(0.5, 2.0),
    }
    xgb_pipe = Pipeline([
        ('prep',  preprocessor),
        ('model', XGBClassifier(scale_pos_weight=class_weight,
                                random_state=RANDOM_STATE,
                                eval_metric='logloss', verbosity=0)),
    ])
    search = RandomizedSearchCV(
        xgb_pipe, param_distributions=param_dist,
        n_iter=30, cv=TimeSeriesSplit(n_splits=5),
        scoring='f1', random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
    )
    print('Optimisation XGBoost (30 itérations x 5 folds temporels)...')
    search.fit(X_train, y_train)
    print('Meilleurs paramètres :')
    for k, v in search.best_params_.items():
        print(f'  {k.replace("model__", ""):20s} = {v}')
    return search


# ------------------------------------------------------------------
# 7. Figures
# ------------------------------------------------------------------

def plot_classification(clf_results, y_test, best_name, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Classification - Train: Jan-Août | Test: Septembre',
                 fontsize=12, fontweight='bold', color=BLUE)

    names = list(clf_results.keys())
    x, w = np.arange(len(names)), 0.2
    for i, (metric, color, label) in enumerate([
        ('f1', LBLUE, 'F1'), ('precision', LGREEN, 'Précision'),
        ('recall', ORANGE, 'Rappel'), ('auc_roc', GREEN, 'AUC-ROC'),
    ]):
        vals = [clf_results[n][metric] for n in names]
        axes[0].bar(x + (i - 1.5) * w, vals, w, label=label, color=color, edgecolor='white')
    axes[0].axhline(0.70, color=RED, linestyle='--', lw=1.5, label='KPI F1 >= 70%')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(fontsize=8)
    axes[0].set_title('Métriques sur le jeu de test')

    cm = confusion_matrix(y_test, clf_results[best_name]['preds'])
    ConfusionMatrixDisplay(cm, display_labels=["À l'heure", 'Retardé']).plot(
        ax=axes[1], colorbar=False, cmap='Blues')
    axes[1].set_title(f'Matrice de confusion - {best_name}')

    for name, color in zip(names, [LBLUE, ORANGE]):
        RocCurveDisplay.from_predictions(
            y_test, clf_results[name]['probas'],
            name=f"{name} (AUC={clf_results[name]['auc_roc']:.3f})",
            ax=axes[2], color=color)
    axes[2].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[2].set_title('Courbes ROC')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_feature_importances(fi_df, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Feature Importances - XGBoost optimisé', fontweight='bold', color=BLUE)

    ax1.barh(fi_df['feature'][::-1], fi_df['importance'][::-1],
             color=LBLUE, edgecolor='white')
    ax1.set_title('Top 20 features (gain XGBoost)')
    ax1.set_xlabel('Importance')

    cumul = fi_df['importance'].cumsum() / fi_df['importance'].sum() * 100
    ax2.plot(range(1, len(fi_df) + 1), cumul.values, marker='o', color=BLUE, linewidth=2)
    ax2.fill_between(range(1, len(fi_df) + 1), cumul.values, alpha=0.15, color=BLUE)
    ax2.axhline(80, color=RED, linestyle='--', lw=1.5, label='80%')
    ax2.axhline(95, color=ORANGE, linestyle='--', lw=1.5, label='95%')
    ax2.set_title('Importance cumulée')
    ax2.set_xlabel('Rang')
    ax2.set_ylabel('%')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------
# 8. Exports + MLflow
# ------------------------------------------------------------------

def export_artifacts(df, C, best_opt, reg_results, best_reg, clf_results,
                     metrics_final, fi_df, X_train, X_test,
                     num_features, bool_features, cat_ordinal, cat_ohe):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    joblib.dump(best_opt, f'{OUTPUT_DIR}/xgb_classifier.pkl')
    joblib.dump(reg_results[best_reg]['pipeline'], f'{OUTPUT_DIR}/xgb_regressor.pkl')
    fi_df.to_csv(f'{OUTPUT_DIR}/feature_importances.csv', index=False)

    # Métriques pour le dashboard (onglet Performances ML)
    metrics_out = {
        'classification': {
            'XGBoost optimisé': metrics_final,
            **{name: {k: float(r[k]) for k in
                      ('f1', 'precision', 'recall', 'accuracy', 'auc_roc')}
               for name, r in clf_results.items()},
        },
        'regression': {name: {k: float(r[k]) for k in ('mae', 'rmse', 'r2')}
                       for name, r in reg_results.items()},
        'best_regressor': best_reg,
        'split': 'temporal (Jan-Août train | Sep test)',
        'train_size': len(X_train),
        'test_size': len(X_test),
    }
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=2)

    # Métadonnées d'inférence : permet au dashboard de reconstruire une
    # ligne de features complète (médianes/modes en valeurs par défaut)
    defaults = {}
    for c in num_features:
        defaults[c] = float(df[c].median())
    for c in bool_features:
        defaults[c] = int(df[c].mode().iloc[0])
    for c in cat_ordinal + cat_ohe:
        defaults[c] = str(df[c].mode().iloc[0])

    meta = {
        'columns': C,
        'num_features': num_features,
        'bool_features': bool_features,
        'cat_ordinal': cat_ordinal,
        'cat_ohe': cat_ohe,
        'all_features': num_features + bool_features + cat_ordinal + cat_ohe,
        'defaults': defaults,
        'conditions': ['Clear', 'Cloudy', 'Rainy', 'Foggy', 'Snowy', 'Stormy'],
        'origins': (sorted(df[C['origin']].dropna().unique().tolist())[:50]
                    if C['origin'] else []),
        'hourly_q75': float(df[C['hourly']].quantile(0.75)) if C['hourly'] else None,
        'hourly_q90': float(df[C['hourly']].quantile(0.90)) if C['hourly'] else None,
    }
    with open(f'{OUTPUT_DIR}/model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print('Artefacts exportés : models/*.pkl, model_metrics.json, models/model_meta.json')
    return metrics_out


def log_mlflow(search, metrics_final, reg_results, best_reg, X_train, X_test, best_opt):
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.models.signature import infer_signature

        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        mlflow.set_experiment('dst_airlines_classification')

        with mlflow.start_run(run_name='XGBoost_TemporalSplit_Optimized') as run:
            mlflow.log_params({k.replace('model__', ''): v
                               for k, v in search.best_params_.items()})
            mlflow.log_param('split_type', 'temporal')
            mlflow.log_param('train_period', 'Jan-Août 2025')
            mlflow.log_param('test_period', 'Sep 2025')
            mlflow.log_param('train_size', len(X_train))
            mlflow.log_param('test_size', len(X_test))
            mlflow.log_param('source', 'PostgreSQL gold_ml_features')

            mlflow.log_metrics({
                'f1_score':  metrics_final['f1'],
                'precision': metrics_final['precision'],
                'recall':    metrics_final['recall'],
                'accuracy':  metrics_final['accuracy'],
                'auc_roc':   metrics_final['auc_roc'],
                'reg_mae':   reg_results[best_reg]['mae'],
                'reg_rmse':  reg_results[best_reg]['rmse'],
            })

            for img in ('ml_classification.png', 'ml_features.png'):
                if os.path.exists(img):
                    mlflow.log_artifact(img)
            mlflow.log_artifact('model_metrics.json')

            sig = infer_signature(X_train, best_opt.predict(X_train))
            mlflow.sklearn.log_model(
                best_opt, 'model', signature=sig,
                registered_model_name='dst_airlines_delay_classifier')

            print(f'MLflow run : {run.info.run_id}')
            print('Modèle enregistré : dst_airlines_delay_classifier')
    except Exception as e:
        print(f'MLflow indisponible ({e}) - modèles sauvegardés localement')


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    df = load_data()
    C = map_columns(df)

    df, df_train, df_test = temporal_split(df, C)
    df = build_targets(df, C)
    df = build_features(df, C)
    # le split a eu lieu avant les cibles/features : on re-filtre proprement
    df_train = df[df[C['ts']].dt.month.isin(TRAIN_MONTHS)].copy()
    df_test  = df[df[C['ts']].dt.month.isin(TEST_MONTHS)].copy()

    num_f, bool_f, cat_ord, cat_ohe = define_feature_sets(df, C)
    all_features = num_f + bool_f + cat_ord + cat_ohe

    X_train = df_train[all_features].copy()
    X_test  = df_test[all_features].copy()
    for c in bool_f:
        X_train[c] = X_train[c].astype(int)
        X_test[c]  = X_test[c].astype(int)

    y_clf_train = df_train['is_delayed_sim'].values
    y_clf_test  = df_test['is_delayed_sim'].values
    y_reg_train = df_train['delay_minutes_ml'].values
    y_reg_test  = df_test['delay_minutes_ml'].values

    # Ratio négatifs/positifs pour rééquilibrer XGBoost (déséquilibre ~77/23)
    class_weight = (y_clf_train == 0).sum() / (y_clf_train == 1).sum()

    preprocessor = build_preprocessor(num_f, bool_f, cat_ord, cat_ohe)

    print('\nClassification (baseline) :')
    clf_results = train_classifiers(preprocessor, X_train, y_clf_train,
                                    X_test, y_clf_test, class_weight)

    print('\nRégression :')
    reg_results = train_regressors(preprocessor, X_train, y_reg_train,
                                   X_test, y_reg_test)
    best_reg = min(reg_results, key=lambda k: reg_results[k]['mae'])

    print()
    search = optimize_xgboost(preprocessor, X_train, y_clf_train, class_weight)
    best_opt = search.best_estimator_

    preds_opt  = best_opt.predict(X_test)
    probas_opt = best_opt.predict_proba(X_test)[:, 1]
    metrics_final = {
        'f1':        float(f1_score(y_clf_test, preds_opt)),
        'precision': float(precision_score(y_clf_test, preds_opt)),
        'recall':    float(recall_score(y_clf_test, preds_opt)),
        'accuracy':  float(accuracy_score(y_clf_test, preds_opt)),
        'auc_roc':   float(roc_auc_score(y_clf_test, probas_opt)),
    }

    print('\nXGBoost optimisé - test (Septembre) :')
    print(f'  F1        : {metrics_final["f1"]:.4f} (KPI >= 70%)')
    print(f'  Précision : {metrics_final["precision"]:.4f}')
    print(f'  Rappel    : {metrics_final["recall"]:.4f}')
    print(f'  AUC-ROC   : {metrics_final["auc_roc"]:.4f}')
    print()
    print(classification_report(y_clf_test, preds_opt,
                                target_names=["À l'heure", 'Retardé']))

    # Feature importances
    try:
        ohe_cols = (best_opt.named_steps['prep']
                    .named_transformers_['ohe']
                    .named_steps['encoder']
                    .get_feature_names_out(cat_ohe).tolist())
    except Exception:
        ohe_cols = []
    feature_names = num_f + bool_f + cat_ord + ohe_cols
    importances = best_opt.named_steps['model'].feature_importances_
    fi_df = (pd.DataFrame({'feature': feature_names, 'importance': importances})
             .sort_values('importance', ascending=False)
             .head(20).reset_index(drop=True))

    # Figures pour le rapport et MLflow
    clf_results_plot = dict(clf_results)
    clf_results_plot['XGBoost optimisé'] = {
        **metrics_final, 'preds': preds_opt, 'probas': probas_opt}
    plot_classification(clf_results_plot, y_clf_test, 'XGBoost optimisé',
                        'ml_classification.png')
    plot_feature_importances(fi_df, 'ml_features.png')

    export_artifacts(df, C, best_opt, reg_results, best_reg, clf_results,
                     metrics_final, fi_df, X_train, X_test,
                     num_f, bool_f, cat_ord, cat_ohe)
    log_mlflow(search, metrics_final, reg_results, best_reg,
               X_train, X_test, best_opt)

    print('\nEntraînement terminé. Lancer ensuite : python3 dashboard.py')


if __name__ == '__main__':
    main()

