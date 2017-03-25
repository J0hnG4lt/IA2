import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from extractFeatures import readLanguages,getFeatureVectors
import json
from similarityMatrix import calculateCondProbMatrix,getUsageData

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


def plot_correlation(dataframe, filename, title='', corr_type=''):
    lang_names = dataframe.columns.tolist()
    tick_indices = np.arange(0.5, len(lang_names) + 0.5)
    plt.figure()
    plt.pcolor(dataframe.values, cmap='RdBu', vmin=-1, vmax=1)
    colorbar = plt.colorbar()
    colorbar.set_label(corr_type)
    plt.title(title)
    plt.xticks(tick_indices, lang_names, rotation='vertical')
    plt.yticks(tick_indices, lang_names)
    plt.savefig(filename)


if __name__ == '__main__':

    data = readLanguages("languagesUsersGithub.json")

    correlations = getUsageData(data,pruneLanguages=30)

    dataset = pd.DataFrame.from_dict(correlations,orient="index")


    pearson_corr = dataset.corr()
    plot_correlation(
        pearson_corr,
        'pearson_language_correlation.svg',
        title='Popular GitHub Language Correlations',
        corr_type='Pearson\'s Correlation')

    spearman_corr = dataset.corr(method='spearman')
    plot_correlation(
        spearman_corr,
        'spearman_language_correlation.svg',
        title='Popular GitHub Language Correlations',
        corr_type='Spearman\'s Rank Correlation')
