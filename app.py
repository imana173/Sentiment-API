"""API Flask d'analyse de sentiments - SocialMetrics AI.

Endpoint principal :
    POST /analyze
    Body JSON : {"tweets": ["tweet 1", "tweet 2", ...]}
    Réponse   : {"tweet 1": 0.83, "tweet 2": -0.55, ...}   (scores ∈ [-1, 1])
"""
from flask import Flask, request, jsonify

import db
from model import SentimentModel, MODEL_PATH

app = Flask(__name__)
app.json.ensure_ascii = False

# Chargement du modèle au démarrage
try:
    model = SentimentModel.load(MODEL_PATH)
except FileNotFoundError:
    model = None


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Route inconnue. Utilisez POST /analyze."}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Méthode non autorisée. Utilisez POST."}), 405


@app.route("/health", methods=["GET"])
def health():
    """Vérifie l'état de l'API, du modèle et de la connexion MySQL."""
    db_ok = True
    try:
        db.count_tweets()
    except Exception:
        db_ok = False
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "database_connected": db_ok,
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyse le sentiment d'une liste de tweets."""
    if model is None:
        return jsonify({"error": "Modèle non entraîné. Lancez scripts/train_model.py."}), 503

    # --- Validation du corps de la requête -----------------------------
    if not request.is_json:
        return jsonify({"error": "Le corps de la requête doit être au format JSON."}), 400

    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({"error": "JSON invalide. Format attendu : {\"tweets\": [\"...\"]}"}), 400

    tweets = data.get("tweets")
    if tweets is None:
        return jsonify({"error": "Champ 'tweets' manquant."}), 400
    if not isinstance(tweets, list):
        return jsonify({"error": "Le champ 'tweets' doit être un tableau de chaînes (string[])."}), 400
    if len(tweets) == 0:
        return jsonify({"error": "La liste de tweets est vide."}), 400
    if not all(isinstance(t, str) for t in tweets):
        return jsonify({"error": "Tous les éléments de 'tweets' doivent être des chaînes de caractères."}), 400
    if any(t.strip() == "" for t in tweets):
        return jsonify({"error": "Les tweets vides ne sont pas autorisés."}), 400

    # --- Prédiction -----------------------------------------------------
    try:
        scores = model.predict_scores(tweets)
    except Exception as exc:  # erreur inattendue du modèle
        return jsonify({"error": f"Erreur interne lors de l'analyse : {exc}"}), 500

    return jsonify({tweet: score for tweet, score in zip(tweets, scores)})


@app.route("/tweets", methods=["POST"])
def add_tweet():
    """Ajoute un tweet annoté à la base (enrichit le dataset d'entraînement).

    Body JSON : {"text": "...", "positive": 1, "negative": 0}
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "JSON invalide."}), 400

    text = data.get("text")
    positive = data.get("positive")
    negative = data.get("negative")

    if not isinstance(text, str) or text.strip() == "":
        return jsonify({"error": "Champ 'text' manquant ou vide."}), 400
    if positive not in (0, 1) or negative not in (0, 1):
        return jsonify({"error": "'positive' et 'negative' doivent valoir 0 ou 1."}), 400
    if positive == 1 and negative == 1:
        return jsonify({"error": "Un tweet ne peut pas être à la fois positif et négatif."}), 400

    try:
        tweet_id = db.insert_tweet(text, positive, negative)
    except Exception as exc:
        return jsonify({"error": f"Erreur base de données : {exc}"}), 500

    return jsonify({"id": tweet_id, "message": "Tweet enregistré."}), 201


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
