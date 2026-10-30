-- SQLite schema. The Flask backend also creates these tables automatically.
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS users (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 username TEXT UNIQUE NOT NULL,
 full_name TEXT NOT NULL,
 email TEXT UNIQUE NOT NULL,
 phone TEXT,
 password_hash TEXT NOT NULL,
 created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 user_id INTEGER,
 input_json TEXT NOT NULL,
 prediction TEXT NOT NULL,
 probabilities TEXT,
 created_at TEXT NOT NULL,
 FOREIGN KEY(user_id) REFERENCES users(id)
);
