import os
import warnings
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
random.seed(1)
from plotnine import *

beer = pd.read_csv("data/aus_production.csv", parse_dates=["ds"])[["ds","Beer"]].loc[lambda x: x["ds"] >= "2000"].assign(Season=lambda x: "Q" + x["ds"].dt.quarter.astype("string")).rename(columns={"Beer":"y"})

# print(beer)

acf = sm.tsa.acf(beer["y"], nlags=9, fft=False, bartlett_confint=False)
acf_df = pd.Series(acf, name="ACF").to_frame().rename_axis("lag")
print(acf_df[1:])
fig, ax = plt.subplots(figsize=(8,3))
plot_acf(beer["y"],lags=16,ax=ax,
         zero=False,bartlett_confint=False,auto_ylims=True)
ax.set(
    title="Autocorrelation Function for Beer",
    xlabel="lag[1Q]", ylabel="acf"
)

plt.show()

plot_acf(beer["y"],lags=48,ax=ax,
         zero=False,bartlett_confint=False,auto_ylims=True)
ax.set(
    title="Autocorrelation Function for Beer",
    xlabel="lag[1Q]", ylabel="acf"
)