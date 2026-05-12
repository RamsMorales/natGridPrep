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


## STL decomposition
# print(z1_timestamp_shcema.describe())
project_data = (
    (
        z1_timestamp_shcema.filter(
            pl.col("ds").is_between(
                pl.date(2004, 4, 4), pl.date(2008, 4, 5), closed="left"
            )
        )
    )
    .sort("ds")
    .drop_nulls()
)

# mstl = MSTL(project_data.to_pandas()["load"],periods=(24,168,8760))
# res = mstl.fit()
# res.plot()
# plt.tight_layout()
# plt.savefig("figs/mstl.png", dpi=150)
# plt.show()

## Expanding EDA to see trends better

### Monthly average load over time

rolling_monthly_average = project_data.with_columns(
    rolling_mean=pl.col("load").rolling_mean_by(
        "ds", window_size="1mo", min_samples=28
    )  # added min_samples to account for non leap year. This way full
    # month is included. I have not put to much thought into wether this
    # matters or not though.
).drop_nulls()

trend_diagnostic_plot = (
    ggplot(rolling_monthly_average, aes(x="ds", y="rolling_mean"))
    + geom_line()
    + geom_hline(
        yintercept=rolling_monthly_average.select(pl.mean("rolling_mean")),
        linetype="dashed",
        color="orange",
    )
    + labs(x="Date", y="Monthly Rolling average load")
)

# trend_diagnostic_plot.show()

### Monthly peak load by year
monthly_peak_load_df = project_data.group_by_dynamic(
    "ds",
    every="1mo",
    closed="both"
).agg(
    pl.col("load").max()
)

monthly_peak_load_df.show()

"""
Two things: -----scratch that 3
1) doing this daily is sick! seems language like in that you start to synthesize the syntax  like you do with sentences. vocab build then new sentances spawn in context despite a given permutation of words not necessarily encountered previously. XD
2) polars is EFFFINGGG SIIIICCKKK!! seems to make solving common problems 
easy enough.
3) I think that the remaining skill is a bit higher order. Engineering seems to come with practice, (I am not implying that this will be trivial. In fact, I think that it follows in the same way that modeling does. You kinda need to build a vocabulary and understand the properties in the same way. Its just that modeling is less salient. I can read the docs on the lib and figure out a solution from examples. I imagine the modelling skill will build the same way, but modelling is documented via papers and encapsulated in academic language that, although captures important nuances, increases the friction to learning it) but modeling decisions, design takes a little bit of domain know how from my perspective, a bit of experience understanding what questions to ask and what view of the data will get you what you want. Also, some detective work to understand how to handle your instance. Next, having some math background and understanding of the properties of the tools, may elevate the modeling skills. In conclusion, I would like to deepen the math and engineering skills as well as practice more modeling problems in context. I think that may be the biggest lever at getting good at this. For the internship and working in general, it may help to learn from more senior data scientists. I don't know how to build that kind of mentorship relationship tho..... 
"""
monthly_peak_load_plot = ggplot(
    data= monthly_peak_load_df,
    aes(x ="ds",y="load")
)

# TODO:
# - finish this plot
# - pivot df to make into separate  years to categorize for box plot
# - figure out the index that lets me separate into series and response 