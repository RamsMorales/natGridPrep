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

    ## Residual diagnositics on the predictors
    # facet_length = len(methods) / 2 if len(methods) % 2 == 0 else len(methods) // 2 + 1
    forecaster.forecast(h=2280, fitted=True, df=train.to_pandas())
    fitted = forecaster.forecast_fitted_values()
    # print(fitted.isna().sum()) # note to self: if you encounter NaN check the count and see if there is
    # a pattern

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
        histogram.show()

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

        acf_plot.show()

        ### residuals time series plot
        ts_plot_resid = (
            ggplot(data=fitted, mapping=aes(x="ds", y=residuals))
            + geom_line()
            + labs(x="", y="", title=f"Residuals for {residuals}")
        )
        ts_plot_resid.show()


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
