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

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

import random

random.seed(23)
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


def get_random_visualization_period(fitted: pd.DataFrame, delta: int):
    #### get a random week to visualize from data
    left_date = fitted["ds"].min()
    right_date = fitted["ds"].max() - pd.Timedelta(weeks=delta)
    random_start = left_date + (right_date - left_date) * random.random()

    period = fitted[
        (fitted["ds"] >= random_start)
        & (fitted["ds"] < random_start + pd.Timedelta(weeks=delta))
    ]
    return period


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
    # forecaster.fit(train.to_pandas())
    # forecasts = forecaster.predict(h=2280).assign(y=test["y"].to_numpy())

    # error_metrics = (
    #     evaluate(forecasts, metrics=[mae, mape, rmse])
    #     .drop(columns=["unique_id"])
    #     .set_index("metric")
    #     .rename_axis("", axis="rows")
    #     .rename_axis("Method", axis="columns")
    #     .transpose()
    #     .reset_index()
    # )

    ## Residual diagnositics on the predictors
    # facet_length = len(methods) / 2 if len(methods) % 2 == 0 else len(methods) // 2 + 1
    forecaster.forecast(h=2280, fitted=True, df=train.to_pandas(), level=[80, 95])
    fitted = forecaster.forecast_fitted_values()
    # print(fitted.isna().sum()) # note to self: if you encounter NaN check the count and see if there is
    # a pattern
    print(fitted.columns)
    resid_names = []

    for method in forecaster.models:
        name = str(method)
        fitted[f"resid_{name}"] = fitted["y"] - fitted[name]
        resid_names.append(f"resid_{name}")

    ### Histogram of residuals for each method
    for residuals in resid_names:
        histogram = (
            ggplot(data=fitted, mapping=aes(x=residuals))
            + geom_histogram(fill="blue", colour="black")
            + labs(x="", y="", title=f"Residuals for {residuals}")
        )
        # histogram.show()

        ### ACF Plots for the residuals looking for them to not have autocorrelation
        acf_resid = fitted[f"{residuals}"]
        if np.any(acf_resid):
            acf_resid = acf_resid.dropna()

        acf_plot = plot_acf(
            x=acf_resid,
            zero=False,
            auto_ylims=True,
            bartlett_confint=False,
            title=f"{residuals} - ACF plot",
        )

        # acf_plot.show()

        ### residuals time series plot
        ts_plot_resid = (
            ggplot(data=fitted, mapping=aes(x="ds", y=residuals))
            + geom_line()
            + labs(x="", y="", title=f"Residuals for {residuals}")
        )
        # ts_plot_resid.show()

        # histogram.show()
        # acf_plot.show()
        # ts_plot_resid.show()

    ## Prediction intervals for naive models

    period_map = {
        "naive": 1,
        "24hour_naive": 1,
        "week_naive": 2,
        "year_naive": 60,
    }
    for method in forecaster.models:
        name = str(method)
        delta = period_map[f"{method}"]
        period = get_random_visualization_period(fitted, delta)

        prediction_interval_plot = (
            ggplot(data=period, mapping=aes(x="ds", y="y"))
            + geom_line(aes(color='"y"'))
            + geom_line(aes(y=f"{method}", color=f'"{method}"'))
            + geom_ribbon(
                aes(
                    ymin=f"{method}-lo-95",
                    ymax=f"{method}-hi-95",
                    fill=f'"{method}-hi-95"',
                ),
                alpha=0.15,
            )
            + geom_ribbon(
                aes(
                    ymin=f"{method}-lo-80",
                    ymax=f"{method}-hi-80",
                    fill=f'"{method}-hi-80"',
                ),
                alpha=0.3,
            )
            + scale_color_manual(values={"y": "black", f"{method}": "blue"})
            + scale_fill_manual(
                values={f"{method}_level_95": "blue", f"{method}_level_80": "blue"}
            )
            + labs(
                x="ds",
                y="",
                title=f"{method} - Fitted values and prediction interval",
                color="Series",
                fill="Level",
            )
            + theme_bw()
        )
        prediction_interval_plot.show()
    ## Evaluation of naive models using error metrics
#     print(error_metrics)
#     mae_plot = (
#         ggplot(error_metrics, aes(x="Method", fill="Method"))
#         + geom_col(aes(y="mae"))
#     )

#     mape_plot = (
#         ggplot(error_metrics, aes(x="Method", fill="Method"))
#         + geom_col(aes(y="mape"))
#     )

#     rmse_plot = (
#         ggplot(error_metrics, aes(x="Method", fill="Method"))
#         + geom_col(aes(y="rmse"))
#     )
# mae_plot.show()
# mape_plot.show()
# rmse_plot.show()
