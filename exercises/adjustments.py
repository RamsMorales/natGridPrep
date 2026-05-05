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
from statsmodels.tsa.seasonal import STL, seasonal_decompose

global_economy = pd.read_csv("data/global_economy.csv")

# df = (
#     global_economy.loc[lambda x: x["unique_id"] == "Australia"].assign(
#         y = lambda x: x["GDP"] / x["Population"]
#     )
# )

df = (
    global_economy.loc[lambda x: x["unique_id"] == "Australia"]
)
series_plot = ggplot(df, aes(x="ds", y="Exports")) + geom_line() + labs(x = "Year [1Y]", y="$ of GDP", title="Total Australian Exports")

# series_plot.show()

## Rolling can do this in pandas. 
# Polars also has rolling and seems to allow for more statistics on a rolling window. 
# I think that sql concept of defining a window might be carried over here.

aus_exports = (
    global_economy.loc[lambda x: x["unique_id"] == "Australia"].assign(
        MA_5 = lambda x: x["Exports"].rolling(5, center=True).mean()
    )
) 
# rolling here just seems to do the aggregation part. the .mean() in pandas 
# seems to apply the actual statistical calculation we want meaning it is similar to 
# polars and probably sql in function despite differences in syntax **** citation needed


fig, ax = plt.subplots()
sns.lineplot(data=aus_exports, x="ds",y="Exports",color = "gray",label="Data")
sns.lineplot(data=aus_exports, x = "ds", y = "MA_5", color="#D55E00",label = "5 - MA")
ax.set(
    title="Total Australian exports",
    xlabel = "Year",
    ylabel = "% of GDP"
)
plt.show()


 

us_employment = pd.read_csv("data/us_employment.csv", parse_dates=["ds"])
us_retail_employment = us_employment.loc[lambda x: (x["unique_id"] == "Retail Trade") & (x["ds"] >= "1990")]

# series_plot = ggplot(us_retail_employment, aes(x="ds", y="y")) + geom_line() + labs(x = "Year [1Y]", y="Persons (thousands)", title="Total employment in us retail") + theme_bw()

# # series_plot.show()

# ### Doing an STL decomposition
stl = STL(us_retail_employment["y"], period =12, seasonal = 13, trend = 21, robust = True)
res = stl.fit()
dcmp = pd.DataFrame({
    "ds": us_retail_employment["ds"],
    "data": us_retail_employment["y"],
    "trend": res.trend,
    "seasonal": res.seasonal,
    "remainder": res.resid,
}).reset_index(drop=True)

# # print(dcmp.head())

# #### ploting data vs trend
# fig, ax = plt.subplots()
# sns.lineplot(data=dcmp, x="ds",y="data",color = "gray",label="Data")
# sns.lineplot(data=dcmp, x = "ds", y = "trend", color="#D55E00",label = "Trend")
# ax.set(
#     title="Total employment in US retail",
#     xlabel = "Month [1M]",
#     ylabel = "Persons (thousands)"
# )
# # plt.show()

#### Plotting STL
fig, axes = plt.subplots(4,1,sharex=True, figsize=(8,6))
sns.lineplot(data=dcmp,x="ds",y="data",ax=axes[0])
sns.lineplot(data=dcmp,x="ds",y="trend",ax=axes[1])
sns.lineplot(data=dcmp,x="ds",y="seasonal",ax=axes[2])
sns.lineplot(data=dcmp,x="ds",y="remainder",ax=axes[3])
axes[0].set_title("Employed = trend + seasonal + remainder",
                  size="medium",loc="left")
axes[0].set(ylabel = "Employed")
axes[3].set(xlabel="")
fig.suptitle("STL decomposition")
plt.show()