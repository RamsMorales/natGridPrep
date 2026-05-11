import os
import warnings
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf
import tsfeatures as tsf
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
random.seed(1)
from plotnine import *

aus_tourism = pd.read_csv('data/aus_tourism.csv', parse_dates=['ds'])
## computing the mean naively
mean_tourism = aus_tourism.groupby(
    "unique_id",
    as_index=False
)["y"].mean()

# print(mean_tourism.sort_values(by="y").head(10))
if __name__ == "__main__":
    ## using tsfeatures
    summary_stats = tsf.tsfeatures(aus_tourism,freq=4,features=[tsf.acf_features],scale=False)
    # print(summary_stats.head(10).iloc[:,:5])

    stl_feat = tsf.tsfeatures(aus_tourism, freq=4,features=[tsf.stl_features])
    # print(stl_feat.head(10).iloc[:,:5])

    df = (
        stl_feat["unique_id"].str.split("-", expand=True).rename(
            columns={0:"region",1:"state",2:"purpose"}).join(stl_feat)
    )
    fig, axs = plt.subplots(3,3,figsize=(8,8))
    axs = axs.flatten()
    for ax, (state,state_df) in zip(axs, df.groupby('state')):
        sns.scatterplot(x="trend",y="seasonal_strength",hue="purpose", edgecolor="none", data=
                        state_df,ax=ax)
        ax.get_legend().remove()
        ax.set(title=state,xlabel="",ylabel="", xlim=(0,1),ylim=(0,1))
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels,
                   title="Purpose", loc="center left",frameon=False,
                   bbox_to_anchor=(1.02,.5), borderaxespad=0)
        fig.supxlabel('Trend')
        fig.supylabel('Seasonal Strnegth')
    for ax in axs:
        if not ax.lines:
            ax.set_visible(False)
    plt.show()
