import duckdb
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from plotnine import *
import numpy as np
import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import (
    Naive,
    SeasonalNaive,
)

from utilsforecast.losses import mae, mape, rmse
from utilsforecast.evaluation import evaluate

database_connection = duckdb.connect("gefcom.duckdb")

load_table = database_connection.execute(
    """
    SELECT * FROM load_long
    """
).pl()

project_data = (
    (
        load_table.filter(
            pl.col("ds").is_between(
                pl.date(2004, 4, 4), pl.date(2008, 4, 5), closed="left"
            )
        )
    )
    .sort("ds")
    .drop_nulls()
).rename({"zone_id": "unique_id", "load": "y"})

if __name__ == "__main__":
    # project_data.show()
    # print(project_data.null_count())
    methods = [
        Naive(alias="naive"),
        SeasonalNaive(season_length=24, alias="24hour_naive"),
        SeasonalNaive(season_length=168, alias="week_naive"),
        SeasonalNaive(season_length=8070, alias="year_naive"),
    ]

    ## Train test splitting on validation set for naive fits testing
    train = project_data.filter(
        (pl.col("ds") < pl.date(2008, 1, 1)) & (pl.col("unique_id") == 1)
    )
    test = project_data.filter(
        (pl.col("ds") >= pl.date(2008, 1, 1)) & (pl.col("unique_id") == 1)
    )
    # print(train.tail())
    # print("----------------------------------------------------------------------------------------")
    # test.show()
    ## Fitting Naive forecasts as baseline models for this project
    forecaster = StatsForecast(models=methods, freq="h")
    forecaster.fit(train.to_pandas())
    forecasts = forecaster.predict(h=2280).assign(y=test["y"].to_numpy())

    error_metrics = (
        evaluate(forecasts, metrics=[mae, mape, rmse])
        .drop(columns=["unique_id"])
        .set_index("metric")
        .rename_axis("", axis="rows")
        .rename_axis("Method", axis="columns")
        .transpose()
        .reset_index()
    )

    print(error_metrics)
    mae_plot = (
        ggplot(error_metrics, aes(x="Method", fill="Method"))
        + geom_col(aes(y="mae"))
    )

    mape_plot = (
        ggplot(error_metrics, aes(x="Method", fill="Method"))
        + geom_col(aes(y="mape"))
    )

    rmse_plot = (
        ggplot(error_metrics, aes(x="Method", fill="Method"))
        + geom_col(aes(y="rmse"))
    )
mae_plot.show()
mape_plot.show()
rmse_plot.show()