# ---------- Étape 1 : image de base avec Node.js et Python ----------
FROM node:20-bullseye-slim

# Installer Python, pip et les dépendances système pour Prophet
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    build-essential \
    libpython3-dev \
    gfortran \
    libblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------- Étape 2 : créer le dossier de l'app ----------
WORKDIR /app

# Copier les fichiers package.json et package-lock.json
COPY package*.json ./

# Installer les dépendances Node.js
RUN npm install

# Copier le reste de l'application
COPY . ./

# ---------- Étape 3 : installer les dépendances Python ----------
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --upgrade pip
RUN pip install pandas numpy scikit-learn prophet openpyxl

# ---------- Étape 4 : Exposer le port ----------
EXPOSE 3000

# ---------- Étape 5 : Commande pour démarrer le serveur ----------
CMD ["node", "rag-server.js"]
