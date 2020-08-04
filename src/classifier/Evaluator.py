import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import sklearn
import imblearn
import scipy
import sklearn.preprocessing as prep
from joblib import load
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler

from classifier import AbstractClassifier
from model import Conf, Set
from util import LogManager, Validation, Common
from util.helper import PreProcessing, Evaluation, Tuning


class Evaluator(AbstractClassifier):
    """
    Class for evaluation of secret test set for MOBD project
    """

    __LOG: logging.Logger = None

    REQUIRED_PYTHON: tuple = (3, 7)

    # current classifiers used
    _MULTILAYER_PERCEPTRON = 'Multi-Layer Perceptron'
    _SUPPORT_VECTOR_MACHINE = 'Support Vector Machine'
    _DECISION_TREE = 'Decision Tree'
    _RANDOM_FOREST = 'Random Forest'
    _KNEAREST_NEIGHBORS = 'K-Nearest Neighbors'
    _STOCHASTIC_GRADIENT_DESCENT = 'Stochastic Gradient Descent'
    _ADA_BOOST = 'Ada Boost'
    _NAIVE_BAYES = 'Naive Bayes'
    _KMEANS = 'K-Means'

    _CLASSIFIER_REL_PATH = './res/classifier/'

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
        # NOTE: if csv test set file has been saved using command pd.to_csv('/path', index=True), it is necessary
        #   using pd.read_csv(self.conf.dataset_test, index_col=0)), so just uncomment following line
        #   and comment previous one
        # self.__test = Set(pd.read_csv(self.conf.dataset_test, index_col=0))

        # current classifiers used
        self.__classifiers = {
             Evaluator._MULTILAYER_PERCEPTRON: None,
             Evaluator._SUPPORT_VECTOR_MACHINE: None,
             Evaluator._DECISION_TREE: None,
             Evaluator._RANDOM_FOREST: None,
             Evaluator._KNEAREST_NEIGHBORS: None,
             #Evaluator._STOCHASTIC_GRADIENT_DESCENT: None,
             Evaluator._ADA_BOOST: None,
             Evaluator._NAIVE_BAYES: None,
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

        # mode
        self.__LOG.info(f"[MODE] Classifier evaluation on test set ({Evaluator.__qualname__})")

        # dataset description
        self.__LOG.debug(f"[DESCRIPTION] Training set description:\n{self.training.set_.describe(include='all')}")
        self.__LOG.debug(f"[DESCRIPTION] Test set description:\n{self.test.set_.describe(include='all')}")

    def split(self) -> None:
        # split features and label
        self.training.set_y = self.training.set_.pop('CLASS')
        self.training.set_x = self.training.set_
        self.training.set_ = None
        self.test.set_y = self.test.set_.pop('CLASS')
        self.test.set_x = self.test.set_
        self.test.set_ = None

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

        # mask = np.array([False, True, True, True, True, True, True, True, False, False,
        #                  True, True, True, True, True, True, False, True, False, True])
        # self.training.set_x = self.training.set_x[:, mask]
        # self.test.set_x = self.test.set_x[:, mask]

    def sample(self) -> None:
        ###############################
        ### data over/undersampling ###
        ###############################

        # oversampling with SMOTE
        sampler = SMOTE(random_state=self.conf.rng_seed)
        # sampler = RandomUnderSampler()
        # sampler = RandomOverSampler()
        # sampler = ADASYN(sampling_strategy='auto', random_state=self.conf.rng_seed)
        self.__LOG.info(f"[SAMPLING] Data sampling using {type(sampler).__qualname__}")
        self.training.set_x, self.training.set_y = sampler.fit_resample(self.training.set_x, self.training.set_y)
        self.__LOG.debug(
            f"[SAMPLING] Train shape after feature selection: {self.training.set_x.shape} | {self.training.set_y.shape}")
        self.__LOG.debug(
            f"[SAMPLING] Test shape after feature selection: {self.test.set_x.shape} | {self.test.set_y.shape}")

    def train(self) -> None:
        # TODO - provare a modificare parametri di tuning (aumenta layer in MLP)
        ###############################
        ### hyper-parameters tuning ###
        ###############################
        self.__LOG.info(f"[TUNING] Hyper-parameters tuning of: {', '.join(self.classifiers.keys())}")

        for name, classifier in self.classifiers.items():
            # if trained classifiers had been dumped, load from *.joblib file
            if self.conf.classifier_dump:
                filename = '_'.join(name.split()) + '.joblib'
                classifier_path = Path(Common.get_root_path(), Evaluator._CLASSIFIER_REL_PATH, filename)
                Validation.can_read(classifier_path, f"Classifier {classifier_path} *must* exists and be readable.")
                self.__LOG.debug(f"[TUNING] Loading {classifier_path} for {name} classifier")
                self.classifiers[name] = load(classifier_path)
            # otherwise, retrain all classifiers
            else:
                self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {name}")
                if name == Evaluator._MULTILAYER_PERCEPTRON:
                    # perform grid search and fit on best evaluator
                    self.classifiers[name] = Tuning.multilayer_perceptron_param_selection(
                        self.training.set_x, self.training.set_y, thread=self.conf.threads)
                elif name == Evaluator._SUPPORT_VECTOR_MACHINE:
                    # perform grid search and fit on best evaluator
                    self.classifiers[name] = Tuning.support_vector_machine_param_selection(
                        self.training.set_x, self.training.set_y, thread=self.conf.threads)
                elif name == Evaluator._DECISION_TREE:
                    # perform grid search and fit on best evaluator
                    self.classifiers[name] = Tuning.decision_tree_param_selection(
                        self.training.set_x, self.training.set_y, thread=self.conf.threads)
                elif name == Evaluator._RANDOM_FOREST:
                    # perform grid search and fit on best evaluator
                    self.classifiers[name] = Tuning.random_forest_param_selection(
                        self.training.set_x, self.training.set_y, thread=self.conf.threads)
                elif name == Evaluator._KNEAREST_NEIGHBORS:
                    # perform grid search and fit on best evaluator
                    self.classifiers[name] = Tuning.knearest_neighbors_param_selection(
                        self.training.set_x, self.training.set_y, thread=self.conf.threads)
                elif name == Evaluator._STOCHASTIC_GRADIENT_DESCENT:
                    # perform grid search and fit on best evaluator
                    self.classifiers[name] = Tuning.stochastic_gradient_descent_param_selection(
                        self.training.set_x, self.training.set_y, thread=self.conf.threads)
                elif name == Evaluator._ADA_BOOST:
                    # perform grid search and fit on best evaluator
                    self.classifiers[name] = Tuning.ada_boosting_param_selection(
                        self.training.set_x, self.training.set_y, thread=self.conf.threads)
                elif name == Evaluator._NAIVE_BAYES:
                    # perform grid search and fit on best evaluator
                    self.classifiers[name] = Tuning.naive_bayes_param_selection(
                        self.training.set_x, self.training.set_y, thread=self.conf.threads)
                elif name == Evaluator._KMEANS:
                    # perform grid search and fit on best evaluator
                    self.classifiers[name] = Tuning.kmeans_param_selection(
                        self.training.set_x, self.training.set_y, thread=self.conf.threads)

    def evaluate(self) -> None:
        ###############################
        ### classifiers' evaluation ###
        ###############################
        self.__LOG.info(f"[EVAL] Computing evaluation on test set for: {', '.join(self.classifiers.keys())}")

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

        self.__LOG.error(f"Something went wrong while training '{__name__}'.", exc_info=True)

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
