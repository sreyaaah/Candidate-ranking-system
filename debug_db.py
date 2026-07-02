import sqlite3
conn = sqlite3.connect('data/candidates.db')
print(conn.execute("SELECT COUNT(1) FROM candidates WHERE skills LIKE '%python%'").fetchone())
