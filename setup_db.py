import duckdb
import os

DB_PATH = "gefcom.duckdb"
conn = duckdb.connect(DB_PATH)

conn.execute(
    """
    CREATE TABLE IF NOT EXISTS load AS
    SELECT * FROM read_csv('data/GEFCOM2012_Data/Load/Load_history.csv', header = True)
    """
)

conn.execute(
    """
    CREATE TABLE IF NOT EXISTS temperature AS
    SELECT * FROM read_csv('data/GEFCOM2012_Data/Load/temperature_history.csv', header = True)
    """
)

print(conn.execute("SHOW TABLES").fetchall())
print(conn.execute("SELECT COUNT(*) FROM load").fetchall())
print(conn.execute("DESCRIBE load").fetchall())

conn.close()