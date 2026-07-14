

CREATE DATABASE IF NOT EXISTS socialmetrics
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE socialmetrics;


CREATE TABLE IF NOT EXISTS tweets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT NOT NULL,
    positive TINYINT(1) NOT NULL DEFAULT 0,
    negative TINYINT(1) NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE USER IF NOT EXISTS 'sentiment_user'@'localhost' IDENTIFIED BY 'sentiment_password';
GRANT SELECT, INSERT, UPDATE, DELETE ON socialmetrics.* TO 'sentiment_user'@'localhost';
FLUSH PRIVILEGES;
