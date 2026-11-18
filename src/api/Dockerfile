# Utiliser une image Python officielle, légère et optimisée
FROM python:3.10-slim

# Définir le dossier de travail à l'intérieur du conteneur
WORKDIR /app

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les dépendances (sans garder le cache pour alléger l'image)
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le reste du code (dont le dossier api/) dans le conteneur
COPY . .

# Exposer le port 8001 sur lequel l'API va écouter
EXPOSE 8001

# Commande par défaut pour lancer l'API au démarrage du conteneur
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8001"]