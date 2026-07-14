
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.joblib")


class SentimentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.clf_positive = LogisticRegression(max_iter=1000, C=2.0)
        self.clf_negative = LogisticRegression(max_iter=1000, C=2.0)
        self.is_trained = False
        self.metrics = {}

    # ------------------------------------------------------------------ #
    # Entraînement
    # ------------------------------------------------------------------ #
    def train(self, texts, y_positive, y_negative, test_size=0.2, random_state=42):
        """Entraîne les deux classifieurs et évalue sur un jeu de validation."""
        X_train_txt, X_val_txt, yp_train, yp_val, yn_train, yn_val = train_test_split(
            texts,
            y_positive,
            y_negative,
            test_size=test_size,
            random_state=random_state,
            stratify=y_positive,
        )

        X_train = self.vectorizer.fit_transform(X_train_txt)
        X_val = self.vectorizer.transform(X_val_txt)

        self.clf_positive.fit(X_train, yp_train)
        self.clf_negative.fit(X_train, yn_train)
        self.is_trained = True

        # Évaluation sur le jeu de validation
        yp_pred = self.clf_positive.predict(X_val)
        yn_pred = self.clf_negative.predict(X_val)

        self.metrics = {
            "n_train": len(X_train_txt),
            "n_validation": len(X_val_txt),
            "positive": self._evaluate(yp_val, yp_pred),
            "negative": self._evaluate(yn_val, yn_pred),
        }
        return self.metrics

    @staticmethod
    def _evaluate(y_true, y_pred):
        return {
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        }

    # ------------------------------------------------------------------ #
    # Prédiction
    # ------------------------------------------------------------------ #
    def predict_scores(self, tweets):
        """Retourne un score ∈ [-1, 1] pour chaque tweet."""
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas entraîné. Lancez scripts/train_model.py.")
        X = self.vectorizer.transform(tweets)
        p_pos = self.clf_positive.predict_proba(X)[:, 1]
        p_neg = self.clf_negative.predict_proba(X)[:, 1]
        scores = p_pos - p_neg
        return [round(float(s), 4) for s in scores]

   
    # Persistance
   
    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path=MODEL_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Modèle introuvable ({path}). Entraînez-le d'abord : python scripts/train_model.py"
            )
        return joblib.load(path)
