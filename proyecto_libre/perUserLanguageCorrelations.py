import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from extractFeatures import readLanguages,getFeatureVectors
import json
from similarityMatrix import calculateCondProbMatrix,getUsageData
import os
import matplotlib.pyplot as plt


def plot_correlation(dataframe, filename, title='', corr_type='',numberOfLanguages = 0):
    lang_names = dataframe.columns.tolist()
    indices = np.arange(0.5, len(lang_names) + 0.5)
    plt.figure()
    plt.pcolor(dataframe.values, cmap='RdBu', vmin=-1, vmax=1)
    colorbar = plt.colorbar()
    colorbar.set_label(corr_type)
    plt.gcf().subplots_adjust(bottom=0.20)
    plt.gcf().subplots_adjust(left=0.20)
    plt.title(title)
    plt.xticks(indices, lang_names, rotation='vertical')
    plt.yticks(indices, lang_names)
    plt.savefig("corridas/Correlaciones/" + str(numberOfLanguages) + " Lenguajes/" + filename)


if __name__ == '__main__':


    try:
        data = readLanguages("DATASET_FINAL/languagesUsersGithub.json")
    except FileNotFoundError:
        print("Couldn't find data at DATASET_FINAL/languagesUsersGithub.json")
        print("Exiting...")
        exit()
    nroLenguajes = int(input("Number of languages to use: "))

    correlations = getUsageData(data,pruneLanguages=nroLenguajes)

    dataset = pd.DataFrame.from_dict(correlations,orient="index")


    # Check if folders exist

    if not os.path.exists("corridas"):
        os.makedirs("corridas")

    if not os.path.exists("corridas/Correlaciones"):
        os.makedirs("corridas/Correlaciones")

    if not os.path.exists("corridas/Correlaciones/" + str(nroLenguajes) + " Lenguajes"):
        os.makedirs("corridas/Correlaciones/" + str(nroLenguajes) + " Lenguajes")



    pearson_corr = dataset.corr()
    plot_correlation(
        pearson_corr,
        'pearson_language_correlation.svg',
        title='Popular GitHub Language Correlations',
        corr_type='Pearson\'s Correlation',
        numberOfLanguages = nroLenguajes)

    spearman_corr = dataset.corr(method='spearman')
    plot_correlation(
        spearman_corr,
        'spearman_language_correlation.svg',
        title='Popular GitHub Language Correlations',
        corr_type='Spearman\'s Rank Correlation',
        numberOfLanguages= nroLenguajes)

    print("Saved results to corridas/Correlaciones/" + str(nroLenguajes)+ " Lenguajes/")
