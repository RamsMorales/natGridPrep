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

# Reviewing data frame
df = pd.DataFrame({
    "Year": list(range(2015,2020)), # synthetic yearly time series
    "Observation": [123,39,78,52,110],
})

# print(df.index)
# print(df.columns)
# print(df.dtypes)

# print(type(df["Year"]))
# print(df["Year"])

# year_df = df.set_index("Year")
# print(year_df)

# TImestamps and Periods
# print(repr(pd.Timestamp("2020-01")))
# print(repr(pd.Period("2020-01")))

# print(repr(pd.Timestamp("2020")))
# print(repr(pd.Timestamp("2020-01-01 12:34")))
# print(repr(pd.Period("2020-01-01")))
# print(repr(pd.Period("2020-01-01",freq="M")))

## Converting between lists of strings and time objects
ts_few = pd.to_datetime(["2020-01-01","2020-01-02","2020-01-03"])
ts_range = pd.date_range("2020-01-01","2020-01-02",freq= "8h")
# print(repr(ts_few)) # note to self the to_datetime object is DatetimeIndex like and its repr is the same as to_str in terms of behavior. Who knows how under hood :P
# print(ts_range)

## converting between datetime, period, and timestamp objects
# print(ts_range.to_period()) # it infers for you
# print(ts_range.to_period(freq="D")) # or you can specify
# print(ts_range.to_period(freq="W")) # or you can specify
# print(ts_range.to_period().to_timestamp()) 

## Converting Date objects to string
# print(ts_few.strftime("%m/%d/%Y"))
# print(ts_range.to_period().strftime("%Y %b ~ %H:%M"))

# Series and Index objects
# df = pd.DataFrame({"ts":ts_few})
# df = df.assign(
#     period = df["ts"].dt.to_period(), # here we are creating a new column by accessing the series object and mutating to a period object using the to_period method
#     yr = df["ts"].dt.year,
#     str = df["ts"].dt.strftime("%A, %B %-d"),
# ).set_index("ts")
'''
The block above takes a data frame with the days then creates a new data frame with 
the time series as a period object for the day, stores the year, and a string format. I don't know what %A %B or %-d mean yet 

Then it sets the index to entries of ts_few itself. I don't understand why you would do this. Sometimes an example is just an example, but hard for me to know the difference.

Sometimes one wants to do best practice and understand how to do code at a high level
This one may just be for learning for right now though.
'''

# Plotting
mel_to_syd_economy = (
    pd.read_csv("data/ansett.csv", parse_dates=["ds"]).loc[lambda x: (x["Airports"] == "MEL-SYD") & (x["Class"] == "Economy")].rename(columns={"Airports":"unique_id"}).assign(y=lambda x: x["y"] / 1000)
)
# print(mel_to_syd_economy)
series_plot = ggplot(mel_to_syd_economy,aes(x="ds",y="y")) +geom_line() + labs(x="Week 1 [1W]", y="Passengers ('000)",title="Ansett airlines economy class: Melbourne-Sydney")
#series_plot.show()

'''
Learned plotting via grammer of graphics in my data visualization class this semester. I really like it due to familiarity. Plus the plotting method that the book describes doesn't seem to work without modifying some things. This produces a similar output. Since I will be faster at producing plots this way, I am going to use this method. However, building a function that does this automatically may be helpful. 

Additionally, the utilsforcast documentation seems kinda lean in a way that i find a bit prohibitive relative to my timeline (bottlenecks and all that jazz)
'''
# Defining total_cost_df for reuse
pbs = (
    pd.read_csv("data/PBS_unparsed.csv",parse_dates=["Month"])[["Month","Concession","Type","ATC1","ATC2","Scripts","Cost"]]
)

total_cost_df = (
    pbs.loc[pbs["ATC2"] == "A10"].drop(columns=["ATC1","ATC2"]).groupby("Month", as_index=False).agg({"Cost":"sum"}).rename(columns={"Cost": "TotalC"})
)
# print(total_cost_df)
# Seasonal plots

seaonal_plot_df = total_cost_df.assign(
    Month_name=total_cost_df["Month"].dt.strftime("%b"),
    Year=total_cost_df["Month"].dt.year,
    Month_num=total_cost_df["Month"].dt.month,
)
"""
Seems like we make a copy and extract the series components since the entries are datetime objects it makes it easy to decomponse by accessing the elements of the object at the respective grain.
"""
unique_years = seaonal_plot_df["Year"].unique()
year_palette = sns.color_palette("husl",n_colors=len(unique_years))
fig, ax = plt.subplots()
sns.lineplot(
    data=seaonal_plot_df, x="Month_num", y="TotalC", hue="Year",palette=year_palette,legend=False,ax=ax
)

ax.set(
    title="Seasonal Plot: Antidabetic Drug Sales",
    xlabel="Month",
    ylabel="$ (millions)",
    xticks=range(1,13),
    xticklabels=[
        "Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","SEP","Oct","Nov","Dec"
    ]
)

min_year = unique_years.min()
for year, subset in seaonal_plot_df.groupby("Year"):
    x = subset["Month_num"].iloc[-1] +.1
    y = subset["TotalC"].iloc[-1]
    color=year_palette[year-min_year]
    ax.text(x,y,str(year),
            ha="left", va="center", fontsize=9,weight="bold", color=color)

plt.show()
