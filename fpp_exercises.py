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
print(repr(pd.Timestamp("2020-01")))
print(repr(pd.Period("2020-01")))

print(repr(pd.Timestamp("2020")))
print(repr(pd.Timestamp("2020-01-01 12:34")))
print(repr(pd.Period("2020-01-01")))
print(repr(pd.Period("2020-01-01",freq="M")))

## Converting between lists of strings and time objects
ts_few = pd.to_datetime(["2020-01-01","2020-01-02","2020-01-03"])
ts_range = pd.date_range("2020-01-01","2020-01-02",freq= "8h")
print(repr(ts_few)) # note to self the to_datetime object is DatetimeIndex like and its repr is the same as to_str in terms of behavior. Who knows how under hood :P
print(ts_range)

## converting between datetime, period, and timestamp objects
print(ts_range.to_period()) # it infers for you
print(ts_range.to_period(freq="D")) # or you can specify
print(ts_range.to_period(freq="W")) # or you can specify
print(ts_range.to_period().to_timestamp()) 
