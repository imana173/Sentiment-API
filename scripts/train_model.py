
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import db
from model import SentimentModel, MODEL_PATH

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def plot_confusion_matrix(cm, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Prédit 0", "Prédit 1"])
    ax.set_yticks([0, 1], labels=["Réel 0", "Réel 1"])
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=16, fontweight="bold")
    ax.set_title(title)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Début de l'entraînement…")

    rows = db.fetch_all_tweets()
    if len(rows) < 20:
        print(f"Erreur : seulement {len(rows)} tweets en base — insuffisant pour entraîner.")
        sys.exit(1)

    texts = [r["text"] for r in rows]
    y_pos = [int(r["positive"]) for r in rows]
    y_neg = [int(r["negative"]) for r in rows]

    print(f"Dataset : {len(texts)} tweets "
          f"({sum(y_pos)} positifs, {sum(y_neg)} négatifs, "
          f"{len(texts) - sum(y_pos) - sum(y_neg)} neutres)")

    model = SentimentModel()
    metrics = model.train(texts, y_pos, y_neg)
    model.save(MODEL_PATH)
    print(f"Modèle sauvegardé : {MODEL_PATH}")

    # --- Rapports --------------------------------------------------------
    os.makedirs(REPORTS_DIR, exist_ok=True)

    plot_confusion_matrix(
        metrics["positive"]["confusion_matrix"],
        "Matrice de confusion - classe POSITIVE",
        os.path.join(REPORTS_DIR, "confusion_matrix_positive.png"),
    )
    plot_confusion_matrix(
        metrics["negative"]["confusion_matrix"],
        "Matrice de confusion - classe NEGATIVE",
        os.path.join(REPORTS_DIR, "confusion_matrix_negative.png"),
    )

    metrics["trained_at"] = datetime.now().isoformat()
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Matrices de confusion et métriques générées dans reports/.")
    for label in ("positive", "negative"):
        m = metrics[label]
        print(f"  [{label:8s}] precision={m['precision']:.3f}  "
              f"recall={m['recall']:.3f}  f1={m['f1_score']:.3f}  "
              f"accuracy={m['accuracy']:.3f}")
    print("Entraînement terminé avec succès.")


if __name__ == "__main__":
    main()
