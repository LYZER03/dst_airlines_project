# DST Airlines - Livrable 5

Algorithme de Machine Learning fonctionnel et optimisé + Dashboard interactif.

## Contenu

| Fichier | Rôle |
|---|---|
| `Livrable5_ML_DSTAirlines.ipynb` | Notebook d'entraînement complet : cibles, feature engineering, split temporel, RF vs XGBoost, optimisation (RandomizedSearchCV + TimeSeriesSplit), export des artefacts, versioning MLflow |
| `train_model.py` | Même pipeline en version script (équivalent à exécuter tout le notebook), pratique pour ré-entraîner sans Jupyter |
| `dashboard.py` | Dashboard Dash (4 onglets) branché sur PostgreSQL et sur le modèle entraîné |

Artefacts produits par le notebook (ou `train_model.py`) :

- `models/xgb_classifier.pkl` - classifieur XGBoost optimisé (pipeline complet, prétraitement inclus)
- `models/xgb_regressor.pkl` - régresseur du retard en minutes
- `models/model_meta.json` - métadonnées d'inférence utilisées par le prédicteur du dashboard
- `model_metrics.json` - métriques affichées dans l'onglet Performances ML
- `ml_classification.png`, `ml_features.png` - figures pour le rapport
- `mlflow.db` - tracking MLflow (modèle enregistré sous `dst_airlines_delay_classifier`)

## Déploiement sur la VM

```bash
# 1. Connexion
ssh -i "C:/Users/celes/Downloads/jan25_alt2_de_airlines.pem" ubuntu@35.181.54.145

# 2. Dépendances
pip install pandas numpy scikit-learn xgboost sqlalchemy psycopg2-binary \
            matplotlib joblib mlflow dash dash-bootstrap-components plotly

# 3. Entraînement (PostgreSQL doit être démarré)
# Option notebook : lancer Jupyter sur la VM puis exécuter toutes les cellules
pip install notebook
jupyter notebook --no-browser --port 8888
# (tunnel depuis le poste local : ssh ... -L 8888:localhost:8888, puis ouvrir
#  le lien avec token affiché dans le terminal)
# Option script, équivalente :
python3 train_model.py

# 4. Dashboard
python3 dashboard.py
```

## Accès au dashboard depuis le poste local

Le port 8050 n'étant pas forcément ouvert sur la VM, le plus simple est un tunnel SSH :

```bash
ssh -i "C:/Users/celes/Downloads/jan25_alt2_de_airlines.pem" -L 8050:localhost:8050 ubuntu@35.181.54.145
```

Puis ouvrir http://localhost:8050 dans le navigateur.

Pour consulter l'interface MLflow (versioning des modèles) :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# avec tunnel : ssh ... -L 5000:localhost:5000 ...
```

## Choix de modélisation

- **Cible** : `is_delayed_15` en base étant à 100 % False (heuristique ETL V1), une
  cible simulée `is_delayed_sim` est construite à partir d'un score de risque
  opérationnel basé sur les vraies données météo et trafic, calibré à ~23 % de
  retards (référence Eurocontrol 2023). Le score de risque n'est pas réutilisé
  comme feature pour éviter le data leakage.
- **Split temporel** : train Janvier-Août, test Septembre. On entraîne sur le
  passé, on évalue sur le futur ; un split aléatoire serait une erreur sur des
  données temporelles.
- **Features** : météo (visibilité, rafales, condition), trafic CDG, encodages
  cycliques (heure, jour, mois), flags opérationnels (LVP < 600 m, vent fort,
  heures de pointe), taux de retard historiques par avion et par origine
  (rolling avec shift pour exclure le vol courant).
- **Modèles** : Random Forest vs XGBoost en baseline, puis optimisation du
  XGBoost par RandomizedSearchCV (30 combinaisons) avec validation croisée
  TimeSeriesSplit (5 folds) qui respecte l'ordre chronologique.
- **KPI** : F1-score >= 70 % sur le jeu de test (classe "retardé"), suivi de la
  précision, du rappel et de l'AUC-ROC.

## Notes

- Si la base est injoignable, le dashboard bascule sur un jeu de démonstration
  et l'indique en console ; si `models/xgb_classifier.pkl` est absent, le
  prédicteur utilise un score heuristique de secours et l'affiche clairement
  dans l'interface. Dans les deux cas, lancer `train_model.py` sur la VM rétablit
  le fonctionnement nominal.
- Le serveur Dash intégré convient pour la démonstration ; pour un vrai
  déploiement, passer par gunicorn (`gunicorn dashboard:app.server -b 0.0.0.0:8050`).
