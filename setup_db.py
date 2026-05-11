import duckdb
import os
import polars as pl

DB_PATH = "gefcom.duckdb"
conn = duckdb.connect(DB_PATH)

# conn.execute(
#     """
#     CREATE TABLE IF NOT EXISTS load AS
#     SELECT * FROM read_csv('data/GEFCOM2012_Data/Load/Load_history.csv', header = True)
#     """
# )

# conn.execute(
#     """
#     CREATE TABLE IF NOT EXISTS temperature AS
#     SELECT * FROM read_csv('data/GEFCOM2012_Data/Load/temperature_history.csv', header = True)
#     """
# )

# print(conn.execute("SHOW TABLES").fetchall())
# print(conn.execute("SELECT COUNT(*) FROM load").fetchall())
# print(conn.execute("DESCRIBE load").fetchall())

# conn.close()

load = conn.execute(
    """
    SELECT * FROM load;
    """
).pl()

# Data prerocessing

## Converting to long format with timestamp as ds and load in float
hour_columns = [f"h{i}" for i in range(1, 25)]

load_long = (
    (
        load.with_columns(
            [
                pl.col(c).str.replace_all(",", "").cast(pl.Float64, strict=False)
                for c in hour_columns
            ]
        )
    )
    .unpivot(
        index=["zone_id", "year", "month", "day"],
        variable_name="hour",
        value_name="load",
    )
    .with_columns(pl.col("hour").str.replace("h", "").cast(pl.Int64))
)

load_long_hr_shift = load_long.with_columns(pl.col("hour") - 1)
updated_load = load_long_hr_shift.with_columns(
    pl.datetime(year="year", month="month", day="day", hour="hour").alias("ds")
).drop(["year", "month", "day", "hour"]).drop_nulls()

conn.execute("DROP TABLE IF EXISTS load_long")
conn.register("load_long_df", updated_load)
conn.execute("CREATE TABLE load_long AS SELECT * FROM load_long_df")
conn.close()