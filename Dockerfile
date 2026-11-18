# On part d'une image Jupyter de base
FROM jupyter/minimal-notebook:latest

# On passe en mode administrateur pour installer les librairies
USER root

# Installation des librairies nécessaires pour le projet
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    xgboost \
    sqlalchemy \
    psycopg2-binary \
    matplotlib \
    seaborn

# On revient à l'utilisateur par défaut de Jupyter
USER jovyan