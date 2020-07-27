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
    def zscore(dataset: ndarray, threshold: int = DEFAULT_ZSCORE_THRESHOLD):
        """

        :param dataset: column of a dataset (1-dimension array)
        :param threshold:
        :return: mask with True values iif abs(zscore) > threshold
        """
        # using z-score method to detect outliers
        zscore = stats.zscore(dataset, nan_policy='raise')
        # outlier is replaced with feature mean
        return np.array(np.abs(zscore) > threshold)

    @staticmethod
    def modified_zscore(dataset: ndarray, threshold: int = DEFAULT_MODIFIED_ZSCORE_THRESHOLD):
        """

        :param dataset: column of a dataset (1-dimension array)
        :param threshold:
        :return: mask with True values iif abs(modified_zscore) > threshold
        """
        # using a modified version of z-score method to detect outliers
        # this method uses median and MAD rather than the mean and standard deviation
        # the median and MAD are robust measures of central tendency and dispersion, respectively
        median = np.median(dataset)
        median_abs_deviation = np.median([np.abs(x - median) for x in dataset])
        modified_zscore = [0.6745 * (x - median) / median_abs_deviation for x in dataset]
        # outlier is replaced with feature mean
        return np.array(np.abs(modified_zscore) > threshold)

    @staticmethod
    def iqr(dataset: ndarray):
        """

        :param dataset: column of a dataset (1-dimension array)
        :return: mask with True values iif (dataset > upper_bound) | (dataset < lower_bound)
        """
        # using inter-quartile range method to detect outliers
        quartile_1, quartile_3 = np.percentile(dataset, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        # outlier is replaced with feature mean
        return np.array((dataset > upper_bound) | (dataset < lower_bound))
