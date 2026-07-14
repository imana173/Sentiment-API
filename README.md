# SocialMetrics AI — API d'Analyse de Sentiments

API Flask qui évalue le sentiment de tweets à l'aide d'une **régression logistique**
(scikit-learn) entraînée sur des tweets annotés stockés dans **MySQL**.
Chaque tweet reçoit un score entre **-1 (très négatif)** et **1 (très positif)**.

## Architecture

```
sentiment-api/
├── app.py                     # API Flask (POST /analyze, POST /tweets, GET /health)
├── model.py                   # Modèle ML : TF-IDF + 2 régressions logistiques
├── db.py                      # Connexion MySQL et accès à la table `tweets`
├── requirements.txt
├── scripts/
│   ├── setup_database.sql     # Création base + table + utilisateur MySQL
│   ├── seed_data.py           # Insertion du dataset annoté de démonstration
│   ├── train_model.py         # Entraînement / réentraînement + matrices de confusion
│   ├── generate_report.py     # Génération du rapport d'évaluation PDF
│   ├── retrain.sh             # Script appelé par le cron hebdomadaire
│   └── crontab.example        # Ligne cron à installer
├── models/                    # Modèle sauvegardé (joblib) — généré
└── reports/                   # Matrices de confusion, métriques, rapport PDF — généré
```

**Fonctionnement du modèle** : deux régressions logistiques indépendantes sont
entraînées sur des caractéristiques TF-IDF (unigrammes + bigrammes) — l'une prédit
le label `positive`, l'autre le label `negative`. Le score renvoyé par l'API est
`P(positif) − P(négatif)`, naturellement borné dans `[-1, 1]`.

## 1. Installation

### Prérequis
- Python ≥ 3.10
- MySQL ≥ 8.0 (ou MariaDB ≥ 10.6)

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-compte>/sentiment-api.git
cd sentiment-api

# 2. Environnement virtuel + dépendances
python3 -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt

# 3. Créer la base de données et la table `tweets`
mysql -u root -p < scripts/setup_database.sql

# 4. Insérer le dataset annoté de démonstration
python scripts/seed_data.py

# 5. Entraîner le modèle (génère models/ et reports/)
python scripts/train_model.py

# 6. Lancer l'API
python app.py
```

L'API écoute sur `http://localhost:5000`.

### Configuration

La connexion MySQL se règle par variables d'environnement (valeurs par défaut
entre parenthèses) : `DB_HOST` (localhost), `DB_USER` (sentiment_user),
`DB_PASSWORD` (sentiment_password), `DB_NAME` (socialmetrics).

## 2. Utilisation de l'API

### `POST /analyze` — analyser une liste de tweets

**Requête**

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"tweets": ["J'\''adore ce restaurant, service excellent !",
                   "Ce film est une catastrophe, quelle déception",
                   "Le magasin ouvre demain à 10h"]}'
```

**Réponse** (`200 OK`)

```json
{
  "J'adore ce restaurant, service excellent !": 0.4026,
  "Ce film est une catastrophe, quelle déception": -0.4815,
  "Le magasin ouvre demain à 10h": -0.0103
}
```

**Gestion des erreurs** (`400`, `405`, `500`, `503`)

| Cas | Code | Réponse |
|---|---|---|
| Liste vide `{"tweets": []}` | 400 | `{"error": "La liste de tweets est vide."}` |
| Champ `tweets` manquant | 400 | `{"error": "Champ 'tweets' manquant."}` |
| `tweets` n'est pas un tableau | 400 | `{"error": "Le champ 'tweets' doit être un tableau de chaînes (string[])."}` |
| Élément non textuel dans la liste | 400 | `{"error": "Tous les éléments de 'tweets' doivent être des chaînes de caractères."}` |
| Corps non JSON | 400 | `{"error": "JSON invalide. ..."}` |
| Mauvaise méthode HTTP (GET) | 405 | `{"error": "Méthode non autorisée. Utilisez POST."}` |
| Modèle non entraîné | 503 | `{"error": "Modèle non entraîné. ..."}` |

### `POST /tweets` — ajouter un tweet annoté (enrichit le dataset)

```bash
curl -X POST http://localhost:5000/tweets \
  -H "Content-Type: application/json" \
  -d '{"text": "Superbe expérience aujourd'\''hui", "positive": 1, "negative": 0}'
```

Réponse `201` : `{"id": 741, "message": "Tweet enregistré."}`

### `GET /health` — état du service

```bash
curl http://localhost:5000/health
# {"status": "ok", "model_loaded": true, "database_connected": true}
```

## 3. Réentraînement hebdomadaire

Le script `scripts/train_model.py` recharge **toutes les données les plus
récentes** de la table `tweets`, réentraîne le modèle, le sauvegarde et
régénère les matrices de confusion et les métriques.

### Automatisation avec cron (Linux/macOS)

```bash
chmod +x scripts/retrain.sh
crontab -e
# ajouter (réentraînement chaque lundi à 3h) :
0 3 * * 1 /bin/bash /chemin/vers/sentiment-api/scripts/retrain.sh
```

Les logs sont écrits dans `logs/retrain.log`.

### Équivalent Windows (Planificateur de tâches)

```
schtasks /create /tn "RetrainSentimentModel" ^
  /tr "python C:\sentiment-api\scripts\train_model.py" ^
  /sc weekly /d MON /st 03:00
```

> Après un réentraînement, redémarrer l'API pour charger le nouveau modèle.

## 4. Rapport d'évaluation

```bash
python scripts/train_model.py       # produit reports/confusion_matrix_*.png + metrics.json
python scripts/generate_report.py   # produit reports/rapport_evaluation.pdf
```

Le rapport (`reports/rapport_evaluation.pdf`) contient les deux matrices de
confusion (classes positive et négative) avec interprétation, les mesures de
précision / rappel / F1-score, une analyse des forces, faiblesses et biais du
modèle, ainsi que des recommandations d'amélioration.

## 5. Résultats actuels (jeu de validation, 148 tweets)

| Classe | Accuracy | Précision | Rappel | F1-score |
|---|---|---|---|---|
| Positive | 0.946 | 1.000 | 0.857 | 0.923 |
| Négative | 0.987 | 0.983 | 0.983 | 0.983 |
