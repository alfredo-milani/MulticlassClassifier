import seaborn as sns
import pandas as pd

from matplotlib import pyplot as plt
from pandas import DataFrame


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
    def compute_pairplot(dataset_path: str):
        from util import Validation
        Validation.can_read(dataset_path)

        sns.set(style='ticks', color_codes=True)
        dataset = pd.read_csv(dataset_path)
        dataset.describe(include='all')
        sns.pairplot(dataset, hue='CLASS', height=2.5)
        plt.show()
