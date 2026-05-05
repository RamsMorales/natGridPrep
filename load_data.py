import duckdb
import polars as pd

database_connection = duckdb.connect()
database_connection.execute("CREATE TABLE Load_history AS SELECT * FROM read_csv('data/GEFCOM2012_Data/Load/Load_history.csv')")

load_history = database_connection.execute(
    """
    SELECT *
    FROM Load_history
    """
).pl()

print(load_history)