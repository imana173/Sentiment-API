
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
OUT = os.path.join(REPORTS_DIR, "rapport_evaluation.pdf")

with open(os.path.join(REPORTS_DIR, "metrics.json"), encoding="utf-8") as f:
    M = json.load(f)

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], spaceAfter=10)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=15,
                      alignment=4, spaceAfter=8)

doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=2 * cm, bottomMargin=2 * cm)
story = []

# ------------------------------------------------------------------ Titre
story.append(Paragraph("Rapport d'Évaluation — API d'Analyse de Sentiments", styles["Title"]))
story.append(Paragraph(
    f"SocialMetrics AI — Modèle de régression logistique (scikit-learn)<br/>"
    f"Dernier entraînement : {M['trained_at'][:19].replace('T', ' ')} — "
    f"{M['n_train']} tweets d'entraînement, {M['n_validation']} tweets de validation (split 80/20)",
    ParagraphStyle("sub", parent=styles["Normal"], alignment=1, textColor=colors.grey)))
story.append(Spacer(1, 18))

# ------------------------------------------------------- 1. Méthodologie
story.append(Paragraph("1. Méthodologie", H1))
story.append(Paragraph(
    "Le système repose sur deux régressions logistiques indépendantes entraînées sur des "
    "caractéristiques TF-IDF (unigrammes et bigrammes, accents normalisés). Le premier "
    "classifieur prédit le label <b>positive</b> et le second le label <b>negative</b> de la table "
    "MySQL <font face='Courier'>tweets</font>. Le score final retourné par l'API pour chaque tweet "
    "est la différence P(positif) − P(négatif), borné dans l'intervalle [−1, 1] : une valeur proche "
    "de 1 indique un sentiment très positif, proche de −1 un sentiment très négatif, et proche de 0 "
    "un contenu neutre. Le jeu de données est découpé en 80 % d'entraînement et 20 % de validation "
    "avec stratification, et toutes les métriques ci-dessous sont mesurées sur le jeu de validation, "
    "jamais vu pendant l'entraînement.", BODY))

# --------------------------------------------- 2. Matrices de confusion
story.append(Paragraph("2. Matrices de confusion", H1))

def bloc_matrice(nom, fichier, m, interpretation):
    story.append(Paragraph(f"2.{1 if nom == 'positive' else 2} Classe {nom.upper()}", H2))
    img = Image(os.path.join(REPORTS_DIR, fichier), width=9.5 * cm, height=7.6 * cm)
    story.append(img)
    cm_ = m["confusion_matrix"]
    story.append(Paragraph(
        f"Vrais négatifs : {cm_[0][0]} — Faux positifs : {cm_[0][1]} — "
        f"Faux négatifs : {cm_[1][0]} — Vrais positifs : {cm_[1][1]}.", BODY))
    story.append(Paragraph(interpretation, BODY))

p = M["positive"]
bloc_matrice(
    "positive", "confusion_matrix_positive.png", p,
    f"<b>Interprétation :</b> le classifieur positif atteint une précision de {p['precision']:.2f} : "
    f"lorsqu'il prédit qu'un tweet est positif, il ne se trompe pratiquement jamais "
    f"({p['confusion_matrix'][0][1]} faux positif(s)). En revanche, son rappel de {p['recall']:.2f} "
    f"montre qu'il manque {p['confusion_matrix'][1][0]} tweets réellement positifs, classés à tort "
    "comme non positifs. Le modèle est donc <b>conservateur</b> sur la classe positive : il préfère "
    "ne pas prédire « positif » en cas de doute, notamment pour des formulations positives implicites "
    "ou peu marquées lexicalement.")

n = M["negative"]
bloc_matrice(
    "negative", "confusion_matrix_negative.png", n,
    f"<b>Interprétation :</b> le classifieur négatif est très équilibré, avec une précision et un "
    f"rappel identiques de {n['precision']:.2f}. Il ne commet que {n['confusion_matrix'][0][1]} faux "
    f"positif(s) et {n['confusion_matrix'][1][0]} faux négatif(s) sur {M['n_validation']} tweets de "
    "validation. Le vocabulaire négatif (« catastrophe », « déception », « arnaque »…) est fortement "
    "discriminant, ce qui explique cette excellente séparation des classes.")

story.append(PageBreak())

# ------------------------------------------------------- 3. Métriques
story.append(Paragraph("3. Précision, rappel et F1-score", H1))

table_data = [
    ["Classe", "Accuracy", "Précision", "Rappel", "F1-score"],
    ["Positive", f"{p['accuracy']:.3f}", f"{p['precision']:.3f}", f"{p['recall']:.3f}", f"{p['f1_score']:.3f}"],
    ["Négative", f"{n['accuracy']:.3f}", f"{n['precision']:.3f}", f"{n['recall']:.3f}", f"{n['f1_score']:.3f}"],
]
t = Table(table_data, colWidths=[3.5 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecf0f1")]),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(t)
story.append(Spacer(1, 10))
story.append(Paragraph(
    "<b>Lecture des métriques.</b> La <b>précision</b> mesure la fiabilité des prédictions "
    "« classe 1 » (parmi les tweets prédits positifs/négatifs, combien le sont réellement). "
    "Le <b>rappel</b> mesure la couverture (parmi les tweets réellement positifs/négatifs, combien "
    "sont détectés). Le <b>F1-score</b> est leur moyenne harmonique et résume l'équilibre entre les "
    "deux. Pour un outil de veille d'opinion comme celui de SocialMetrics AI, un rappel élevé sur la "
    "classe négative est particulièrement important : rater des tweets négatifs signifierait passer "
    "à côté d'une crise de réputation naissante. Avec un rappel négatif de "
    f"{n['recall']:.2f}, le modèle répond bien à ce besoin.", BODY))

# ------------------------------------------------------- 4. Analyse
story.append(Paragraph("4. Analyse des performances : forces, faiblesses, biais", H1))
story.append(Paragraph(
    "<b>Forces.</b> Les deux classifieurs affichent des F1-scores supérieurs à 0,92, avec une "
    "détection quasi parfaite du sentiment négatif. Les scores retournés par l'API sont cohérents et "
    "gradués : un tweet enthousiaste obtient un score nettement positif, un tweet factuel (horaires, "
    "annonces) reste proche de 0, et un tweet hostile descend franchement sous 0. Le pipeline complet "
    "(MySQL → entraînement → sauvegarde joblib → API Flask) est reproductible et réentraînable "
    "automatiquement.", BODY))
story.append(Paragraph(
    "<b>Faiblesses observées.</b> L'erreur dominante est un déficit de rappel sur la classe positive : "
    f"{p['confusion_matrix'][1][0]} tweets positifs sur {sum(p['confusion_matrix'][1])} ne sont pas "
    "reconnus. Ces erreurs concernent typiquement des tweets positifs formulés sans mots fortement "
    "polarisés (« je reviendrai », « conforme à mes attentes ») que le TF-IDF ne relie pas assez au "
    "vocabulaire positif appris. Par ailleurs, un modèle sac-de-mots ne capture ni la <b>négation</b> "
    "(« pas mauvais du tout » risque d'être vu comme négatif), ni l'<b>ironie</b> ou le second degré, "
    "très fréquents sur X/Twitter.", BODY))
story.append(Paragraph(
    "<b>Biais éventuels.</b> (1) <b>Biais de domaine</b> : le dataset d'entraînement est construit "
    "autour d'avis de consommation (restaurants, films, applications…) ; les performances peuvent "
    "chuter sur des sujets politiques ou d'actualité. (2) <b>Biais lexical</b> : une partie du corpus "
    "est générée par gabarits, ce qui sur-représente certaines tournures et gonfle probablement les "
    "métriques par rapport à des tweets réels, plus bruités (fautes, argot, émojis, hashtags). "
    "(3) <b>Biais de classe</b> : les neutres sont sous-représentés (24 % du corpus), ce qui peut "
    "pousser le modèle à polariser des messages factuels. (4) <b>Biais de langue</b> : le modèle est "
    "monolingue français et n'a aucune garantie sur des tweets en anglais ou multilingues.", BODY))

# ------------------------------------------------- 5. Recommandations
story.append(Paragraph("5. Recommandations d'amélioration", H1))
story.append(Paragraph(
    "1. <b>Enrichir le dataset avec des tweets réels annotés</b> (via l'endpoint POST /tweets), en "
    "visant plusieurs milliers d'exemples incluant émojis, hashtags et fautes de frappe, afin de "
    "réduire le biais de génération par gabarits.", BODY))
story.append(Paragraph(
    "2. <b>Traiter la négation et les émojis</b> dans le prétraitement (par exemple fusionner "
    "« pas_bon » en un seul token, mapper les émojis vers des tokens de sentiment), ce qui devrait "
    "directement améliorer le rappel de la classe positive.", BODY))
story.append(Paragraph(
    "3. <b>Rééquilibrer les classes</b> (sur-échantillonnage des neutres ou pondération "
    "class_weight='balanced') et ajuster le seuil de décision du classifieur positif pour arbitrer "
    "précision/rappel selon le besoin métier.", BODY))
story.append(Paragraph(
    "4. <b>Suivre la dérive du modèle</b> : conserver l'historique des métriques à chaque "
    "réentraînement hebdomadaire et déclencher une alerte si le F1-score baisse, plutôt que de "
    "remplacer le modèle aveuglément.", BODY))
story.append(Paragraph(
    "5. <b>À plus long terme</b>, comparer la régression logistique à des représentations plus riches "
    "(embeddings type fastText ou CamemBERT) qui capturent le contexte et l'ironie, tout en gardant "
    "la régression logistique comme référence simple et interprétable.", BODY))

doc.build(story)
print(f"Rapport généré : {OUT}")
