import seaborn as sns
import pandas as pd

from pathlib import Path
from matplotlib import pyplot as plt
from pandas import DataFrame

from util import Validation


class PreProcessing(object):
    """
    Contains helper methods
    """

    @staticmethod
    def get_na_count(dataset: DataFrame) -> int:
        """
        Function counting missing values for each feature from dataset
        :param dataset: dataset
        :return: sum of missing values
        """
        # per ogni elemento (i,j) del dataset, isna() restituisce
        # TRUE/FALSE se il valore corrispondente è mancante/presente
        boolean_mask = dataset.isna()
        # contiamo il numero di TRUE per ogni attributo sul dataset
        return boolean_mask.sum(axis=0)

    @staticmethod
    def compute_pairplot(dataset: DataFrame, destination_file: str = None):
        """

        :param dataset:
        :param destination_file:
        :return:
        """
        sns.set(style='ticks', color_codes=True)
        sns.pairplot(dataset, hue='CLASS', height=2.5)

        if destination_file is None:
            plt.show()
        else:
            plt.savefig(destination_file)
