import duckdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
from statsmodels.tsa.seasonal import STL, seasonal_decompose
import numpy as np
import polars as pl

database_connection = duckdb.connect("gefcom.duckdb")
load_table = database_connection.execute(
    """
    SELECT * FROM load
    """
).pl()

# Noticed that the data is organized in a table with zone_id + year + month + day
# as primary key. Then the rest of the colums are the observations.
# There are some null values present in the data for 2008 zone 20,
# but I don't know the full extent from this view. I am going to focus
# on zone 1 arbitrarily just to practice this skill.

zone_1_load = database_connection.execute(
    """
    SELECT * FROM load
    WHERE zone_id == 1;
    """
).pl()

# Data prerocessing

## Converting to long format with timestamp as ds and load in float
hour_columns = [f"h{i}" for i in range(1, 25)]

z1_long = (
    (
        zone_1_load.with_columns(
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

z1_long_hr_shift = z1_long.with_columns(pl.col("hour") -1)
z1_timestamp_shcema = z1_long_hr_shift.with_columns(
    pl.datetime(
        year="year",
        month="month",
        day="day",
        hour="hour"
    ).alias("ds")
).drop(["year","month","day","hour"])

## Checking for Daylight savings duplicates

daylight_savings_debug_view = z1_timestamp_shcema.group_by(['zone_id', pl.col("ds").dt.date() \
                                                            .alias("date")]).agg(pl.len().alias("n_hours")).sort("date")

# print(daylight_savings_debug_view)
# print(daylight_savings_debug_view.filter(pl.col("n_hours") != 24))


# TODO:
# - find a solution for the null values
# - setup the STL decomposition
# - descriptive statistivs appropriate? I think maybe for the response once aggregated.

# I am arbitrarily deciding to fill the NA values with a local rolling average since the data is typically
# seasonal and largely autocorrelated based on experience with ISO data.
# NOTE: assumption is made here about handling Null values in the data.

print(z1_timestamp_shcema.filter(pl.col("load").is_nan() | pl.col("load").is_null()))

# zone_1_filled_NA_rolling_avg = zone_1_load.fillna(zone_1_load.rolling())
