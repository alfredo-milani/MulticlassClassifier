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
        Validation.can_read(
            conf.dataset_train,
            f"Training set file *must* exists and be readable. "
            f"Current file: '{conf.dataset_train}'.\n"
            f"Training set path (fully qualified) can be specified in conf.ini file or using Conf object."
        )
        Validation.can_read(
            conf.dataset_test,
            f"Test set file *must* exists and be readable. "
            f"Current file: '{conf.dataset_test}'.\n"
            f"Test set path (fully qualified) can be specified in conf.ini file or using Conf object."
        )

        self.__LOG = LogManager.get_instance().logger(LogManager.Logger.EVAL)
        self.__conf = conf

        # using full dataset as training set
        self.__training = Set(pd.read_csv(self.conf.dataset_train))

        # load test set if it has same format as training_set.csv provided
        # as example file see ./res/dataset/test_set_no_index.csv
        self.__test = Set(pd.read_csv(self.conf.dataset_test))
        # load test set if it has header (F1-20 and CLASS row) and index, so a test test saved using
        #   command pd.to_csv('/path', index=True)
        # as example file see ./res/dataset/test_set_index.csv
        # self.__test = Set(pd.read_csv(self.conf.dataset_test, index_col=0))
        # load test set if it does not have header row (does not have F1-20 and CLASS row) and
        #   it is was not saved using command pd.to_csv('/path', index=True), so it has not index
        # as example file see ./res/dataset/test_set_no_index_features.csv
        # self.__test = Set(pd.read_csv(self.conf.dataset_test, header=None,
        #                               names=[f"F{i}" for i in range(1, 21)] + ["CLASS"]))

        # current classifiers used
        self.__classifiers = {
             Evaluator._MULTILAYER_PERCEPTRON: None,
             Evaluator._SUPPORT_VECTOR_MACHINE: None,
             Evaluator._DECISION_TREE: None,
             Evaluator._RANDOM_FOREST: None,
             Evaluator._KNEAREST_NEIGHBORS: None,
             # Evaluator._STOCHASTIC_GRADIENT_DESCENT: None,
             Evaluator._ADA_BOOST: None,
             Evaluator._NAIVE_BAYES: None,
             # Evaluator._KMEANS: None
        }

    def prepare(self) -> None:
        """
        Print library version, classifier type, training set description
        """
        super().prepare()

        # print libs' version
        self.__LOG.debug(f"[LIB VERSION] {np.__name__} : {np.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {pd.__name__} : {pd.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {matplotlib.__name__} : {matplotlib.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {sklearn.__name__} : {sklearn.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {imblearn.__name__} : {imblearn.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {scipy.__name__} : {scipy.__version__}")

        # mode
        self.__LOG.info(f"[MODE] Classifiers evaluation on test set ({Evaluator.__qualname__})")

        # dataset description
        self.__LOG.debug(f"[DESCRIPTION] Training set description:\n{self.training.set_.describe(include='all')}")
        self.__LOG.debug(f"[DESCRIPTION] Test set description:\n{self.test.set_.describe(include='all')}")

    def split(self) -> None:
        """
        Split training/testing set features and labels
        """
        self.__LOG.info(f"[DATA SPLIT] Separating training/test set in features and labels")

        # split features and label
        self.training.set_y = self.training.set_.pop('CLASS')
        self.training.set_x = self.training.set_
        self.training.set_ = None
        self.test.set_y = self.test.set_.pop('CLASS')
        self.test.set_x = self.test.set_
        self.test.set_ = None

    def manage_bad_values(self) -> None:
        """
        Replace missing data with median (as it is not affected by outliers) and
          outliers detected using modified z-score
        """
        # manage missing data
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

        # manage outliers on training se only if there are no classifiers' dump
        if not self.conf.classifier_dump:
            # manage outliers
            # outliers_manager = 'z-score'
            # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
            # http://colingorrie.github.io/outlier-detection.html#fn:2
            outliers_manager = 'modified z-score'
            # outliers_manager = 'inter-quartile range'

            self.__LOG.info(f"[OUTLIERS] Managing outliers using {outliers_manager} method")

            self.__LOG.debug(
                f"[OUTLIERS] Training set x description before manage outlier:\n"
                f"{self.training.set_x.describe(include='all')}"
            )

            outliers_count = 0
            for feature in self.training.set_x.columns:
                # outliers_mask = PreProcessing.zscore(self.training.set_x[feature])
                outliers_mask = PreProcessing.modified_zscore(self.training.set_x[feature])
                # outliers_mask = PreProcessing.iqr(self.training.set_x[feature])

                # outliers approximation
                self.training.set_x.loc[outliers_mask, feature] = feature_median_dict[feature]

                # update outliers_count
                outliers_count += sum(outliers_mask)

            samples_training_set_x = self.training.set_x.shape[0] * self.training.set_x.shape[1]
            self.__LOG.debug(f"[OUTLIERS] Outliers detected on all features (F1-F20): "
                             f"{outliers_count} out of {samples_training_set_x} "
                             f"samples ({round(outliers_count / samples_training_set_x * 100, 2)} %)")

            self.__LOG.debug(
                f"[OUTLIERS] Training set x description after manage outlier:\n"
                f"{self.training.set_x.describe(include='all')}"
            )

    def normalize(self) -> None:
        """
        Data scaling/normalization
        """
        scaler = prep.MinMaxScaler(feature_range=(0, 1))
        # using following transformation:
        #  X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        #  X_scaled = X_std * (max - min) + min
        self.__LOG.info(f"[SCALING] Data scaling using {type(scaler).__qualname__}")

        # if there are classifiers' dump, scale using data set used previously for training
        if self.conf.classifier_dump:
            # TODO - vedere se normalizzare solo la parte usata nella fase di training
            self.__LOG.debug(f"[SCALING] Data scaling fitted on same training set of training phase.")
            # using same training set from previous training phase
            scaler.fit(self.training.set_x.sample(
                frac=1 - self.conf.dataset_test_ratio,
                random_state=self.conf.rng_seed
            ))
            self.test.set_x = scaler.transform(self.test.set_x)
        # if there are not classifiers' dump, scale using complete training set
        else:
            scaler.fit(self.training.set_x)
            self.training.set_x = scaler.transform(self.training.set_x)
            self.test.set_x = scaler.transform(self.test.set_x)

    def feature_selection(self) -> None:
        """
        Features selection
        """
        self.__LOG.info(f"[FEATURE SELECTION] Feature selection")

        # if trained classifiers has been dumped, use mask obtained from previous training
        if self.conf.classifier_dump:
            features_mask = np.array([False, True, True, True, True, True, True, True, False, False,
                                      True, True, True, True, True, True, False, True, False, True])
            self.__LOG.debug(f"[FEATURE SELECTION] Feature selection using boolean mask obtained "
                             f"from previous training phase: {features_mask}")
            self.test.set_x = self.test.set_x[:, features_mask]
        # otherwise, select best features
        else:
            # TODO - provare un approccio di tipo wrapper, il Recursive Feature Elimination o il SelectFromModel di sklearn
            # do not consider the 5 features that have less dependence on the target feature
            # (i.e. the class to which they belong)
            selector = SelectKBest(mutual_info_classif, k=15)
            self.__LOG.debug(f"[FEATURE SELECTION] Feature selection using {type(selector).__qualname__}")
            selector.fit(self.training.set_x, self.training.set_y)
            self.training.set_x = selector.transform(self.training.set_x)
            self.__LOG.debug(
                f"[FEATURE SELECTION] Feature index after SelectKBest: {selector.get_support(indices=True)}")
            self.test.set_x = selector.transform(self.test.set_x)
            self.__LOG.debug(
                f"[FEATURE SELECTION] Train shape after feature selection: {self.training.set_x.shape} | {self.training.set_y.shape}")
            self.__LOG.debug(
                f"[FEATURE SELECTION] Test shape after feature selection: {self.test.set_x.shape} | {self.test.set_y.shape}")

    def sample(self) -> None:
        """
        Data over/undersampling
        """
        # if there are not dumps for classifiers, sample training set
        if not self.conf.classifier_dump:
            # TODO - vedere se usando altri sampler i classificatori performano meglio
            # oversampling with SMOTE
            sampler = SMOTE(random_state=self.conf.rng_seed)
            # sampler = RandomUnderSampler(random_state=self.conf.rng_seed)
            # sampler = RandomOverSampler(random_state=self.conf.rng_seed)
            # sampler = ADASYN(sampling_strategy="auto", random_state=self.conf.rng_seed)
            self.__LOG.info(f"[SAMPLING] Data sampling using {type(sampler).__qualname__}")
            self.training.set_x, self.training.set_y = sampler.fit_resample(self.training.set_x, self.training.set_y)
            self.__LOG.debug(
                f"[SAMPLING] Train shape after feature selection: {self.training.set_x.shape} | {self.training.set_y.shape}")
            self.__LOG.debug(
                f"[SAMPLING] Test shape after feature selection: {self.test.set_x.shape} | {self.test.set_y.shape}")

    def train(self) -> None:
        """
        Perform Cross-Validation using GridSearchCV to find best hyper-parameter and refit classifiers on
          complete training set
        """
        self.__LOG.info(f"[TUNING] Hyper-parameters tuning of: {', '.join(self.classifiers.keys())}")

        for name, classifier in self.classifiers.items():
            # if trained classifiers had been dumped, load from *.joblib file
            if self.conf.classifier_dump:
                filename = '_'.join(name.split()) + '.joblib'
                classifier_path = Path(Common.get_root_path(), Evaluator._CLASSIFIER_REL_PATH, filename)
                try:
                    Validation.can_read(classifier_path)
                    self.__LOG.debug(f"[TUNING] Loading {classifier_path} for {name} classifier")
                    self.classifiers[name] = load(classifier_path)
                    self.__LOG.info(f"[TUNING] Best {name} classifier: {self.classifiers[name]}")
                except PermissionError:
                    self.__LOG.warning(f"[TUNING] Error loading file '{classifier_path}'.\n"
                                       f"The file *must* exists and be readable.\n"
                                       f"Classifier '{name}' will be skipped.")
                    continue
                except KeyError:
                    self.__LOG.warning(f"[TUNING] The file '{classifier_path}' appears to be corrupted.\n"
                                       f"Classifier '{name}' will be skipped.")
                    continue
            # otherwise, retrain all classifiers
            else:
                self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {name}")
                if name == Evaluator._MULTILAYER_PERCEPTRON:
                    # perform grid search and fit on best evaluator
                    self.classifiers[name] = Tuning.multilayer_perceptron_param_selection(
                        self.training.set_x, self.training.set_y, jobs=self.conf.jobs)
                elif name == Evaluator._SUPPORT_VECTOR_MACHINE:
                    # perform grid search and fit on best evaluator
                    self.classifiers[name] = Tuning.support_vector_machine_param_selection(
                        self.training.set_x, self.training.set_y, jobs=self.conf.jobs)
                elif name == Evaluator._DECISION_TREE:
                    # perform grid search and fit on best evaluator
                    self.classifiers[name] = Tuning.decision_tree_param_selection(
                        self.training.set_x, self.training.set_y, jobs=self.conf.jobs)
                elif name == Evaluator._RANDOM_FOREST:
                    # perform grid search and fit on best evaluator
                    self.classifiers[name] = Tuning.random_forest_param_selection(
                        self.training.set_x, self.training.set_y, jobs=self.conf.jobs)
                elif name == Evaluator._KNEAREST_NEIGHBORS:
                    # perform grid search and fit on best evaluator
                    self.classifiers[name] = Tuning.knearest_neighbors_param_selection(
                        self.training.set_x, self.training.set_y, jobs=self.conf.jobs)
                elif name == Evaluator._STOCHASTIC_GRADIENT_DESCENT:
                    # perform grid search and fit on best evaluator
                    self.classifiers[name] = Tuning.stochastic_gradient_descent_param_selection(
                        self.training.set_x, self.training.set_y, jobs=self.conf.jobs)
                elif name == Evaluator._ADA_BOOST:
                    # perform grid search and fit on best evaluator
                    self.classifiers[name] = Tuning.ada_boosting_param_selection(
                        self.training.set_x, self.training.set_y, jobs=self.conf.jobs)
                elif name == Evaluator._NAIVE_BAYES:
                    # perform grid search and fit on best evaluator
                    self.classifiers[name] = Tuning.naive_bayes_param_selection(
                        self.training.set_x, self.training.set_y, jobs=self.conf.jobs)
                elif name == Evaluator._KMEANS:
                    # perform grid search and fit on best evaluator
                    self.classifiers[name] = Tuning.kmeans_param_selection(
                        self.training.set_x, self.training.set_y, jobs=self.conf.jobs)

    def evaluate(self) -> None:
        """
        Evaluate all specified classifiers
        """
        self.__LOG.info(f"[EVAL] Computing evaluation on test set for: {', '.join(self.classifiers.keys())}")

        for name, classifier in self.classifiers.items():
            if not classifier:
                continue

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

        self.__LOG.info(f"Successfully evaluated all (valid) classifiers specified")

    def on_error(self, exception: Exception = None) -> None:
        super().on_error(exception)

        self.__LOG.error(f"Something went wrong during evaluation ({__name__}).", exc_info=True)

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
