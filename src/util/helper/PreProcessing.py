import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from scipy import stats


class PreProcessing(object):
    """
    Contains helper methods
    """

    DEFAULT_ZSCORE_THRESHOLD = 3
    DEFAULT_MODIFIED_ZSCORE_THRESHOLD = 3.5

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
    def compute_pairplot(dataset: DataFrame, destination_file: str = None) -> None:
        """

        :param dataset: DataFrame containing current dataset
        :param destination_file: str with destination filename to be saved
        """
        sns.set(style='ticks', color_codes=True)
        sns.pairplot(dataset, hue='CLASS', height=2.5)

        if destination_file is None:
            plt.show()
        else:
            plt.savefig(destination_file)

    @staticmethod
    def zscore(dataset: DataFrame, threshold: int = DEFAULT_ZSCORE_THRESHOLD):
        """

        :param dataset:
        :param threshold:
        :return:
        """
        for feature in dataset.columns:
            # using z-score method to detect outliers
            zscore = stats.zscore(dataset[feature], nan_policy='raise')
            # outlier is replaced with feature mean
            dataset.loc[np.abs(zscore) > threshold, feature] = dataset[feature].mean()

    @staticmethod
    def modified_zscore(dataset: DataFrame, threshold: int = DEFAULT_MODIFIED_ZSCORE_THRESHOLD):
        """

        :param dataset:
        :param threshold:
        :return:
        """
        for feature in dataset.columns:
            # using a modified version of z-score method to detect outliers
            # this method uses median and MAD rather than the mean and standard deviation
            # the median and MAD are robust measures of central tendency and dispersion, respectively
            median = np.median(dataset[feature])
            median_abs_deviation = np.median([np.abs(x - median) for x in dataset[feature]])
            modified_zscore = [0.6745 * (x - median) / median_abs_deviation for x in dataset[feature]]
            # outlier is replaced with feature mean
            dataset.loc[np.abs(modified_zscore) > threshold, feature] = dataset[feature].mean()

    @staticmethod
    def iqr(dataset: DataFrame):
        """

        :param dataset:
        :return:
        """
        for feature in dataset.columns:
            # using inter-quartile range method to detect outliers
            quartile_1, quartile_3 = np.percentile(dataset[feature], [25, 75])
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr * 1.5)
            upper_bound = quartile_3 + (iqr * 1.5)
            # outlier is replaced with feature mean
            dataset.loc[
                (dataset[feature] > upper_bound) | (dataset[feature] < lower_bound),
                feature
            ] = dataset[feature].mean()
