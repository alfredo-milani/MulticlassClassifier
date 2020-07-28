import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import sklearn
import imblearn
import scipy
import sklearn.preprocessing as prep
import sklearn.model_selection as ms
import sklearn.metrics as metrics
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from scipy import stats
from sklearn.neural_network import MLPClassifier

from classifier import AbstractClassifier
from model import Conf, Set
from util import LogManager, Validation
from util.helper import PreProcessing, Tuning, Evaluation


class Evaluator(AbstractClassifier):
    """

    """

    __LOG: logging.Logger = None

    REQUIRED_PYTHON: tuple = (3, 7)

    # current classifiers used
    _MULTILAYER_PERCEPTRON = 'Multi-Layer Perceptron'
    _SUPPORT_VECTOR_MACHINE = 'Support Vector Machine'
    _RANDOM_FOREST = 'Random Forest'
    _KNEAREST_NEIGHBORS = 'K-Nearest Neighbors'
    _STOCHASTIC_GRADIENT_DESCENT = 'Stochastic Gradient Descent'
    _ADA_BOOST = 'Ada Boost'
    _NAIVE_BAYES = 'Naive Bayes'
    _KMEANS = 'K-Means'

    def __init__(self, conf: Conf):
        super().__init__()

        # validate python version
        Validation.python_version(
            Evaluator.REQUIRED_PYTHON,
            f"Unsupported Python version.\n"
            f"Required Python {Evaluator.REQUIRED_PYTHON[0]}.{Evaluator.REQUIRED_PYTHON[1]} or higher."
        )

        self.__LOG = LogManager.get_instance().logger(LogManager.Logger.EVAL)
        self.__conf = conf

        # using full dataset as training set
        self.__training = Set(pd.read_csv(self.conf.dataset))
        self.__test = Set(pd.read_csv(self.conf.dataset_test))

        # current classifiers used
        self.__classifiers = {
            # Evaluator._MULTILAYER_PERCEPTRON: None,
            Evaluator._KMEANS: None
        }

    def prepare(self) -> None:
        super().prepare()

        # print libs' version
        self.__LOG.debug(f"[LIB VERSION] {np.__name__} : {np.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {pd.__name__} : {pd.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {matplotlib.__name__} : {matplotlib.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {sklearn.__name__} : {sklearn.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {imblearn.__name__} : {imblearn.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {scipy.__name__} : {scipy.__version__}")

    def split(self) -> None:
        pass

    def manage_bad_values(self) -> None:
        ###########################
        ### manage missing data ###
        ###########################
        self.__LOG.info(f"[MISSING DATA] Managing missing data")

        self.__LOG.debug(
            f"[MISSING DATA] Training set x before processing (shape: {self.training.set_x.shape}):\n"
            f"{PreProcessing.get_na_count(self.training.set_x)}"
        )
        self.__LOG.debug(
            f"[MISSING DATA] Test set x before processing (shape: {self.test.set_x.shape}):\n"
            f"{PreProcessing.get_na_count(self.test.set_x)}"
        )

        # dictionary containing median for each feature
        feature_median_dict: dict = {}
        for feature in self.training.set_x.columns:
            # using feature median from training set to manage missing values for training and test set
            # it is not used mean as it is affected by outliers
            feature_median_dict[feature] = self.training.set_x[feature].median()
            self.training.set_x[feature].fillna(feature_median_dict[feature], inplace=True)
            self.test.set_x[feature].fillna(feature_median_dict[feature], inplace=True)

        self.__LOG.debug(
            f"[MISSING DATA] Training set x after processing (shape: {self.training.set_x.shape}):\n"
            f"{PreProcessing.get_na_count(self.training.set_x)}"
        )
        self.__LOG.debug(
            f"[MISSING DATA] Test set x after processing (shape: {self.test.set_x.shape}):\n"
            f"{PreProcessing.get_na_count(self.test.set_x)}"
        )

        #######################
        ### manage outliers ###
        #######################
        zscore = 'z-score'
        modified_zscore = 'modified z-score'
        iqr = 'inter-quantile range'

        self.__LOG.info(f"[OUTLIER] Managing outlier using {modified_zscore} method")

        self.__LOG.debug(
            f"[DESCRIPTION] Training set x description before manage outlier:\n"
            f"{self.training.set_x.describe(include='all')}"
        )

        # TODO - vedere se, usando RobustScaler e modifica di outlier con mediana, i risultati sono simili
        for feature in self.training.set_x.columns:
            # TODO - devono essere gestiti anche gli outliers del test set ?
            # outliers = PreProcessing.zscore(self.training.set_x[feature])
            outliers = PreProcessing.modified_zscore(self.training.set_x[feature])
            # outliers = PreProcessing.iqr(self.training.set_x[feature])

            # outliers approximation
            self.training.set_x.loc[outliers, feature] = feature_median_dict[feature]

        self.__LOG.debug(
            f"[DESCRIPTION] Training set x description after manage outlier:\n"
            f"{self.training.set_x.describe(include='all')}"
        )

    def normalize(self) -> None:
        ##################################
        ### data scaling/normalization ###
        ##################################
        scaler = prep.MinMaxScaler(feature_range=(0, 1))
        # using following transformation:
        #  X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        #  X_scaled = X_std * (max - min) + min
        self.__LOG.info(f"[SCALING] Data scaling using {type(scaler).__qualname__}")
        scaler.fit(self.training.set_x)
        self.training.set_x = scaler.transform(self.training.set_x)
        self.test.set_x = scaler.transform(self.test.set_x)

    def feature_selection(self) -> None:
        ##########################
        ### features selection ###
        ##########################
        # TODO - provare un approccio di tipo wrapper, il Recursive Feature Elimination o il SelectFromModel di sklearn
        # do not consider the 5 features that have less dependence on the target feature
        # (i.e. the class to which they belong)
        selector = SelectKBest(mutual_info_classif, k=15)
        self.__LOG.info(f"[FEATURE SELECTION] Feature selection using {type(selector).__qualname__}")
        selector.fit(self.training.set_x, self.training.set_y)
        self.training.set_x = selector.transform(self.training.set_x)
        self.__LOG.debug(f"[FEATURE SELECTION] Feature index after SelectKBest: {selector.get_support(indices=True)}")
        self.test.set_x = selector.transform(self.test.set_x)
        self.__LOG.debug(
            f"[FEATURE SELECTION] Train shape after feature selection: {self.training.set_x.shape} | {self.training.set_y.shape}")
        self.__LOG.debug(
            f"[FEATURE SELECTION] Test shape after feature selection: {self.test.set_x.shape} | {self.test.set_y.shape}")

    def sample(self) -> None:
        pass

    def tune(self) -> None:
        ###############################
        ### hyper-parameters tuning ###
        ###############################
        self.__LOG.info(f"[TUNING] Hyper-parameters tuning of: {', '.join(k for k in self.classifiers.keys())}")

        for name, classifier in self.classifiers.items():
            self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {name}")
            if name == Evaluator._MULTILAYER_PERCEPTRON:
                self.__classifiers[name] = Tuning.multilayer_perceptron_param_selection(self.training.set_x, self.training.set_y)
            elif name == Evaluator._KMEANS:
                self.__classifiers[name] = Tuning.kmeans_param_selection(self.training.set_x, self.training.set_y)

    def train(self) -> None:
        ##############################
        ### train best classifiers ###
        ##############################
        classifier = MLPClassifier(max_iter=10000, activation='relu', hidden_layer_sizes=(100, 50),
                                   learning_rate='adaptive', learning_rate_init=0.01, solver='sgd')
        classifier.fit(self.training.set_x, self.training.set_y)

    def evaluate(self) -> None:
        ###############################
        ### classifiers' evaluation ###
        ###############################
        self.__LOG.info(f"[EVAL] Computing evaluation for: {', '.join(k for k in self.classifiers.keys())}")

        for name, classifier in self.classifiers.items():
            accuracy, precision, recall, f1_score, confusion_matrix = Evaluation.evaluate(
                self.classifiers[name],
                self.test.set_x,
                self.test.set_y
            )
            self.__LOG.info(
                f"[EVAL] Evaluation of {name}:\n"
                f"\t- Accuracy: {accuracy}\n"
                f"\t- Precision: {precision}\n"
                f"\t- Recall: {recall}\n"
                f"\t- F1-score: {f1_score}\n"
                f"\t- Confusion matrix: \n{confusion_matrix}"
            )

    def on_success(self) -> None:
        super().on_success()

    def on_error(self, exception: Exception = None) -> None:
        super().on_error(exception)

    @property
    def conf(self) -> Conf:
        return self.__conf

    @property
    def training(self) -> Set:
        return self.__training

    @property
    def test(self) -> Set:
        return self.__test

    @property
    def classifiers(self) -> dict:
        return self.__classifiers
