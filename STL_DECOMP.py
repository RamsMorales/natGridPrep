import duckdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
from statsmodels.tsa.seasonal import MSTL, seasonal_decompose
import numpy as np
import polars as pl
import calplot

database_connection = duckdb.connect("gefcom.duckdb")
load_table = database_connection.execute(
    """
    SELECT * FROM load
    """
).pl()

# Noticed that the data is organized in a table with
#  zone_id + year + month + day
# as primary key. Then the rest of the colums are the observations.
# There are some null values present in the data for 2008 zone 20,
# but I don't know the full extent from this view.
# I am going to focus on zone 1 arbitrarily just to
# practice this skill.

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

z1_long_hr_shift = z1_long.with_columns(pl.col("hour") - 1)
z1_timestamp_shcema = z1_long_hr_shift.with_columns(
    pl.datetime(year="year", month="month", day="day", hour="hour").alias("ds")
).drop(["year", "month", "day", "hour"])

## Checking for Daylight savings duplicates

daylight_savings_debug_view = (
    z1_timestamp_shcema.group_by(["zone_id", pl.col("ds").dt.date().alias("date")])
    .agg(pl.len().alias("n_hours"))
    .sort("date")
)

# print(daylight_savings_debug_view)
# print(daylight_savings_debug_view.filter(pl.col("n_hours") != 24))

# I am arbitrarily deciding to fill the NA values with a local rolling average since the data is typically
# seasonal and largely autocorrelated based on experience with ISO data.
# NOTE: assumption is made here about handling Null values in the data.

## Checking for null values
null_debug_view = (
    z1_timestamp_shcema.filter(pl.col("load").is_nan() | pl.col("load").is_null())
    .group_by(["zone_id", pl.col("ds").dt.date().alias("date")])
    .agg(pl.len())
    .sort("date")
)

# null_debug_view.show(None)

null_dates = null_debug_view.to_pandas().set_index("date")["len"]
null_dates.index = pd.to_datetime(null_dates.index)

# calplot.calplot(null_dates, cmap="Reds", figsize=(16, 10))
# plt.savefig("figs/null_pattern.png", dpi = 150)
# plt.show() #okay... this is sick!!

# found a structured pattern for most of the nulls and visualized
# after. Seems like most of the nulls are the in the benchmark
# Except for those on 2008 06 30.

# z1_timestamp_shcema.filter(
#     pl.col("ds").dt.date() == pl.date(2006,6,30)
# ).sort("ds").show()

# This last one represents a data cutoff from the source for
# whatever reason. I think I am just going to use the data for the
# first 4 years based on what is said here:
# https://blog.drhongtao.com/2016/07/gefcom2012-load-forecasting-data.html

# TODO:
# - setup the STL decomposition
# - descriptive statistivs appropriate? I think maybe for the response once aggregated.

## STL decomposition
# print(z1_timestamp_shcema.describe())
project_data = (
    z1_timestamp_shcema.filter(
        pl.col("ds").is_between(pl.date(2004, 4, 4), pl.date(2008, 4, 5), closed="left")
    )
).sort("ds").drop_nulls()

mstl = MSTL(project_data.to_pandas()["load"],periods=(24,168,8760))
res = mstl.fit()
res.plot()
plt.tight_layout()
plt.show()


