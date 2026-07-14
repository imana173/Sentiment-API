
import os
import pymysql

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "sentiment_user"),
    "password": os.getenv("DB_PASSWORD", "sentiment_password"),
    "database": os.getenv("DB_NAME", "socialmetrics"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}


def get_connection():
    """Ouvre une connexion MySQL à partir des variables d'environnement."""
    return pymysql.connect(**DB_CONFIG)


def fetch_all_tweets():
    """Récupère tous les tweets annotés (dataset d'entraînement)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, text, positive, negative FROM tweets ORDER BY id")
            return cur.fetchall()


def insert_tweet(text: str, positive: int, negative: int) -> int:
    """Insère un tweet annoté et retourne son id."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)",
                (text, int(positive), int(negative)),
            )
        conn.commit()
        return cur.lastrowid


def count_tweets() -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM tweets")
            return cur.fetchone()["n"]
