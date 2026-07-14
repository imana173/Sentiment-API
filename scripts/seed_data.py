
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import db

# (texte, positive, negative)
POSITIFS = [
    "J'adore ce nouveau téléphone, il est incroyable !",
    "Quelle journée magnifique, je suis trop content",
    "Le service client était parfait, merci beaucoup",
    "Ce film est un chef d'oeuvre, à voir absolument",
    "Super expérience, je recommande à 100%",
    "Bravo à toute l'équipe pour cette victoire éclatante",
    "Le concert d'hier soir était juste génial",
    "Je suis fier de ce qu'on a accompli ensemble",
    "Excellente nouvelle, le projet est validé !",
    "Ce restaurant est une pépite, plats délicieux",
    "Merci pour ce cadeau, il me fait tellement plaisir",
    "L'application est fluide et super bien pensée",
    "Vacances de rêve, paysages à couper le souffle",
    "Enfin diplômé, quel bonheur immense",
    "La mise à jour améliore vraiment tout, top",
    "Accueil chaleureux et personnel adorable",
    "Je souris rien qu'en y repensant, moment parfait",
    "Produit de qualité exceptionnelle, très satisfait",
    "Cette série est passionnante du début à la fin",
    "Quel plaisir de retrouver mes amis, soirée géniale",
    "Le livre m'a transporté, une lecture merveilleuse",
    "Résultats au-delà de nos espérances, formidable",
    "J'aime tellement cette chanson, elle me rend heureux",
    "Livraison rapide et emballage soigné, parfait",
    "Le nouveau parc est superbe, les enfants ont adoré",
    "Une équipe à l'écoute et efficace, je suis ravi",
    "Ce gâteau est une tuerie, délicieux",
    "Très bonne surprise, bien mieux que prévu",
    "Le match était splendide, quelle remontada fantastique",
    "Interface claire, tout fonctionne à merveille",
    "Je me sens en pleine forme, motivation au top",
    "Un grand merci, votre aide a été précieuse",
    "Magnifique coucher de soleil ce soir, sublime",
    "Promotion obtenue, je suis aux anges",
    "L'hôtel était propre, confortable et charmant",
    "Performance impressionnante, chapeau l'artiste",
    "Ce café est excellent, meilleur du quartier",
    "Réunion très productive, super ambiance d'équipe",
    "Adorable petit chaton, il illumine ma journée",
    "Quel talent, cette exposition est splendide",
]

NEGATIFS = [
    "Ce produit est une arnaque totale, à fuir",
    "Je suis vraiment déçu par ce service lamentable",
    "Quelle horreur ce film, j'ai perdu mon temps",
    "Le pire restaurant de ma vie, plats immangeables",
    "Service client catastrophique, personne ne répond",
    "Je déteste cette application, elle plante sans arrêt",
    "Livraison en retard et colis abîmé, inadmissible",
    "Cette journée est un cauchemar, tout va mal",
    "Déception totale, rien ne fonctionne comme prévu",
    "L'hôtel était sale et bruyant, à éviter absolument",
    "Prix exorbitant pour une qualité médiocre",
    "Je suis furieux, ma commande a encore été annulée",
    "Quel désastre ce match, jeu catastrophique",
    "La mise à jour a tout cassé, c'est nul",
    "Personnel désagréable et incompétent, honteux",
    "Ce téléphone est une catastrophe, batterie morte en 2h",
    "Trahi par mes collègues, ambiance toxique au travail",
    "Nourriture froide et fade, expérience horrible",
    "Je regrette cet achat, qualité déplorable",
    "Encore une panne, ce train est toujours en retard",
    "Film ennuyeux et prévisible, énorme déception",
    "Arnaque pure et simple, ne commandez jamais ici",
    "Ma journée est gâchée, tout s'écroule",
    "Interface confuse et bugs partout, insupportable",
    "Le concert était nul, son horrible et retard énorme",
    "Vendeur malhonnête, produit défectueux non remboursé",
    "C'est inadmissible d'être traité comme ça, scandaleux",
    "Je suis épuisé et démoralisé par cette situation",
    "Ce livre est d'un ennui mortel, illisible",
    "Aucun respect pour les clients, boycott immédiat",
    "Le wifi ne marche jamais, connexion exécrable",
    "Résultats désastreux, tout est à refaire",
    "Ce café est infect, imbuvable",
    "Application inutile qui vole nos données, honteux",
    "Pire expérience d'achat de ma vie, fuyez",
    "Le service après-vente m'a raccroché au nez, minable",
    "Chambre minuscule, moisissure au plafond, répugnant",
    "Je suis en colère, promesse non tenue encore une fois",
    "Ambiance sinistre et attente interminable, nul",
    "Produit cassé dès la première utilisation, pathétique",
]

NEUTRES = [
    "Le magasin ouvre à 9h du lundi au samedi",
    "La réunion est prévue demain à 14h en salle B",
    "Il pleut aujourd'hui sur Paris",
    "Le nouveau modèle sort le mois prochain",
    "J'ai pris le bus pour aller au travail",
    "La conférence aura lieu en visioconférence",
    "Le colis est en cours de livraison",
    "Ils ont annoncé une mise à jour pour la semaine prochaine",
    "Le prix du billet est de vingt euros",
    "Je regarde les informations à la télévision",
    "Le document est disponible en téléchargement",
    "La boulangerie est fermée le dimanche",
    "Le train part de la gare de Lyon à 8h15",
    "Il y a une nouvelle boutique dans le centre commercial",
    "Le rapport contient une cinquantaine de pages",
    "La météo annonce du vent pour demain",
    "Mon rendez-vous a été déplacé à jeudi",
    "Le musée propose une visite guidée à 11h",
    "Ils recrutent un développeur en CDI",
    "L'application est disponible sur Android et iOS",
]


# ---------------------------------------------------------------------
# Génération combinatoire : sujets x expressions de sentiment.
# Permet d'obtenir un dataset suffisamment grand pour entraîner un
# modèle TF-IDF + régression logistique de manière robuste.
# ---------------------------------------------------------------------
SUJETS = [
    "ce restaurant", "ce film", "cette application", "ce téléphone",
    "cet hôtel", "ce concert", "ce livre", "ce service client",
    "cette série", "ce produit", "cette formation", "ce café",
    "cette compagnie aérienne", "ce jeu vidéo", "cette boutique",
    "ce musée", "cette voiture", "ce site web", "cette banque", "ce coiffeur",
]

EXPRESSIONS_POSITIVES = [
    "est vraiment excellent, je recommande",
    "est génial, j'ai adoré du début à la fin",
    "m'a agréablement surpris, quelle qualité",
    "est parfait, rien à redire",
    "est une superbe découverte, bravo",
    "est top, expérience très agréable",
    "mérite cinq étoiles, service impeccable",
    "est formidable, je suis ravi",
    "est magnifique, un vrai bonheur",
    "est incroyable, j'en suis très satisfait",
    "est au top, merci pour ce moment",
    "est fantastique, je reviendrai avec plaisir",
]

EXPRESSIONS_NEGATIVES = [
    "est vraiment décevant, à éviter",
    "est nul, j'ai détesté",
    "est une catastrophe, fuyez",
    "est horrible, quelle perte de temps",
    "est médiocre, très mauvaise qualité",
    "est lamentable, je suis furieux",
    "ne vaut rien, grosse arnaque",
    "est insupportable, plus jamais",
    "est affreux, expérience désastreuse",
    "est pitoyable, je regrette amèrement",
    "m'a beaucoup déçu, c'est honteux",
    "est exécrable, service catastrophique",
]

EXPRESSIONS_NEUTRES = [
    "ouvre à neuf heures tous les jours",
    "se trouve dans le centre-ville",
    "propose une nouvelle offre ce mois-ci",
    "sera fermé pendant les travaux",
    "a changé d'adresse récemment",
    "publie ses horaires sur son site",
    "accepte les paiements par carte",
    "organise un événement la semaine prochaine",
]


def generer_dataset():
    """Combine les phrases manuelles et les phrases générées."""
    dataset = []
    for t in POSITIFS:
        dataset.append((t, 1, 0))
    for t in NEGATIFS:
        dataset.append((t, 0, 1))
    for t in NEUTRES:
        dataset.append((t, 0, 0))

    for sujet in SUJETS:
        for expr in EXPRESSIONS_POSITIVES:
            dataset.append((f"Franchement {sujet} {expr}", 1, 0))
        for expr in EXPRESSIONS_NEGATIVES:
            dataset.append((f"Franchement {sujet} {expr}", 0, 1))
        for expr in EXPRESSIONS_NEUTRES:
            dataset.append((f"Pour info {sujet} {expr}", 0, 0))

    return dataset


def main():
    n = db.count_tweets()
    if n > 0:
        print(f"La table contient déjà {n} tweets. Aucune insertion (évite les doublons).")
        return

    dataset = generer_dataset()
    for text, pos, neg in dataset:
        db.insert_tweet(text, pos, neg)

    print(f"{len(dataset)} tweets annotés insérés dans la table `tweets`.")


if __name__ == "__main__":
    main()
