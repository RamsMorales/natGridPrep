from functools import partial
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from utilsforecast.evaluation import evaluate
from utilsforecast.feature_engineering import pipeline, trend
from utilsforecast.losses import rmse, mae, mape as _mape, mase, quantile_loss, mqloss
from statsforecast import StatsForecast
from statsforecast.models import (
    HistoricAverage,
    Naive,
    RandomWalkWithDrift,
    SeasonalNaive,
    SklearnModel,
)
import pandas as pd
from plotnine import *

def mape(df, models, id_col = "unique_id", target_col = "y"):
    df_mape = _mape(df, models, id_col=id_col, target_col=target_col)
    df_mape.loc[:, df_mape.select_dtypes(include="number").columns] *= 100
    return df_mape

def quantile_score(df, models, q=0.5, id_col="unique_id", target_col="y"):
    df_qs = quantile_loss(df, models, q=q, id_col=id_col, target_col=target_col)
    df_qs.loc[:, df_qs.select_dtypes(include="number").columns] *= 2
    return df_qs
## TIDY step - filter and select and fill nulls etc. data preproc
gdp_df = (
    pd.read_csv("data/global_economy.csv",parse_dates=["ds"])[[
        "unique_id","ds","GDP","Population"
    ]].assign(
        GDP = lambda x: x["GDP"].interpolate(),
        Population = lambda x: x["Population"].interpolate(),
        y=lambda x: x["GDP"] / x["Population"]
    )
)
if __name__ == "__main__":
## Visualize - many of these but visualize the series is 
# good practice as step 0
    series_plot = ggplot(gdp_df[gdp_df["unique_id"] == "Sweden"],aes(x="ds",y="y"))+geom_line() + labs(x="Year [1Y]", y="$US", title="GDP per capita for Sweden")
    series_plot.show()
    