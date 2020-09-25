import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import sklearn
import imblearn
import scipy
import sklearn.preprocessing as prep
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from joblib import load
from sklearn import set_config
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, SelectPercentile
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler, SVMSMOTE
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
             # Evaluator._DECISION_TREE: None,
             Evaluator._RANDOM_FOREST: None,
             # Evaluator._KNEAREST_NEIGHBORS: None,
             # Evaluator._STOCHASTIC_GRADIENT_DESCENT: None,
             Evaluator._ADA_BOOST: None,
             # Evaluator._NAIVE_BAYES: None,
             # Evaluator._KMEANS: None
        }

    def prepare(self) -> None:
        """
        Print library version, classifier type, training set description
        """
        super().prepare()

        # print libs' version
        self.log.debug(f"[LIB VERSION] {np.__name__} : {np.__version__}")
        self.log.debug(f"[LIB VERSION] {pd.__name__} : {pd.__version__}")
        self.log.debug(f"[LIB VERSION] {matplotlib.__name__} : {matplotlib.__version__}")
        self.log.debug(f"[LIB VERSION] {sklearn.__name__} : {sklearn.__version__}")
        self.log.debug(f"[LIB VERSION] {imblearn.__name__} : {imblearn.__version__}")
        self.log.debug(f"[LIB VERSION] {scipy.__name__} : {scipy.__version__}")

        # mode
        self.log.info(f"[MODE] Classifiers evaluation on test set ({Evaluator.__qualname__})")

        # dataset description
        self.log.debug(f"[DESCRIPTION] Training set description:\n{self.training.set_.describe(include='all')}")
        self.log.debug(f"[DESCRIPTION] Test set description:\n{self.test.set_.describe(include='all')}")

        # print all parameters for classifiers
        set_config(print_changed_only=False)

    def clean_data(self) -> None:
        """
        Replace missing data with median (as it is not affected by outliers) and
          outliers detected using modified z-score
        """
        self.log.info(f"[DATA CLEANING] Splitting features/labels for train / test")
        # split features and label
        self.training.y = self.training.set_.pop('CLASS')
        self.training.X = self.training.set_
        self.training.set_ = None
        self.test.y = self.test.set_.pop('CLASS')
        self.test.X = self.test.set_
        self.test.set_ = None

        ########################################
        # DATA NORMALIZATION AND DATA SCALING
        scaler = prep.MinMaxScaler(feature_range=(0, 1))
        # scaler = prep.MaxAbsScaler()
        # scaler = prep.QuantileTransformer(output_distribution='uniform')
        # scaler = prep.PowerTransformer(method='yeo-johnson')
        # scaler = prep.PowerTransformer(method='box - cox')

        self.log.info(f"[DATA CLEANING] Data scaling using {type(scaler).__qualname__}")
        scaler.fit(self.training.X)
        # scaler = Normalizer().fit(self.training.set_x)  # fit does nothing.

        self.training.X = scaler.transform(self.training.X)
        self.test.X = scaler.transform(self.test.X)

        ############################
        # MANAGING MISSING VALUES
        self.log.info(f"[DATA CLEANING] Managing missing values")

        # define imputer
        # missing_values_imputer = KNNImputer(n_neighbors=7, weights='distance')
        missing_values_imputer = IterativeImputer(max_iter=10, imputation_order='ascending',
                                                  random_state=self.conf.rng_seed, initial_strategy='constant')
        # fit on the dataset
        missing_values_imputer.fit(self.training.X)
        # transform the dataset
        self.training.X = missing_values_imputer.transform(self.training.X)
        self.test.X = missing_values_imputer.transform(self.test.X)

        ####################
        # MANAGING OUTLIERS
        self.log.info(f"[DATA CLEANING] Managing outliers")

        # outliers management depends on the nature of the individual features.
        outliers_count = 0
        for feature in range(self.training.X.shape[1]):
            # outliers detection
            # outliers_mask = PreProcessing.zscore(self.training.X[:, feature])
            outliers_mask = PreProcessing.modified_zscore(self.training.X[:, feature])
            # outliers_mask = PreProcessing.iqr(self.training.X[:, feature])

            # outliers approximation
            # self.training.X[outliers_mask, feature] = np.median(self.training.X[:, feature])
            self.training.X[outliers_mask, feature] = np.nan

            # update outliers_count
            outliers_count += sum(outliers_mask)

        samples_training_set_x = self.training.X.shape[0] * self.training.X.shape[1]
        self.log.debug(f"[DATA CLEANING] Outliers detected on all features (F1-F20): "
                       f"{outliers_count} out of {samples_training_set_x} "
                       f"samples ({(outliers_count / samples_training_set_x * 100):.2f} %)")

        # replace outliers using imputer
        # outliers_imputer = KNNImputer(n_neighbors=7, weights='distance')
        outliers_imputer = IterativeImputer(max_iter=1500, imputation_order='ascending',
                                            random_state=self.conf.rng_seed, initial_strategy='constant')
        # fit on the dataset
        outliers_imputer.fit(self.training.X)
        # transform the dataset
        self.training.X = outliers_imputer.transform(self.training.X)

    def feature_selection(self) -> None:
        """
        Features selection
        """
        #######################
        # FEATURE SELECTION
        # selector = SelectPercentile(score_func=mutual_info_classif, percentile=100)
        # selector = RFE(estimator=LogisticRegression(max_iter=1500), n_features_to_select=15)
        selector = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=15)
        self.log.info(f"[FEATURE SELECTION] Feature selection using {type(selector).__qualname__}")
        selector.fit(self.training.X, self.training.y)
        self.training.X = selector.transform(self.training.X)
        self.log.debug(f"[FEATURE SELECTION] Feature index after {type(selector).__qualname__}: {selector.get_support(indices=True)}")
        self.test.X = selector.transform(self.test.X)
        self.log.debug(
            f"[FEATURE SELECTION] Train shape after feature selection: {self.training.X.shape} | {self.training.y.shape}")
        self.log.debug(
            f"[FEATURE SELECTION] Test shape after feature selection: {self.test.X.shape} | {self.test.y.shape}")

    def sample(self) -> None:
        """
        Data over/undersampling
        """
        self.log.debug(
            f"[SAMPLING] Train shape: {self.training.X.shape} | {self.training.y.shape}")
        # oversampling with SMOTE
        oversampling = SVMSMOTE(svm_estimator=SVC(), k_neighbors=5, m_neighbors=5,
                                n_jobs=self.conf.jobs, random_state=self.conf.rng_seed)
        # undersampling = RandomUnderSampler(random_state=self.conf.rng_seed)
        # steps = [('oversampling', oversampling), ('undersampling', undersampling)]
        # sampler = Pipeline(steps=steps)
        sampler = oversampling
        self.log.info(f"[SAMPLING] Data sampling using {type(sampler).__qualname__}")
        self.training.X, self.training.y = sampler.fit_resample(self.training.X, self.training.y)
        self.log.debug(
            f"[SAMPLING] Train shape after data sampling: {self.training.X.shape} | {self.training.y.shape}")

    def train(self) -> None:
        """
        Perform Cross-Validation using GridSearchCV to find best hyper-parameter and refit classifiers on
          complete training set
        """
        self.log.info(f"[TUNING] Training of: {', '.join(self.classifiers.keys())}")

        for name, classifier in self.classifiers.items():
            if name == Evaluator._MULTILAYER_PERCEPTRON:
                # training classifier on training set
                self.classifiers[name] = MLPClassifier(
                    activation='relu', alpha=0.05, hidden_layer_sizes=(240, 120),
                    learning_rate='adaptive', learning_rate_init=0.01, solver='sgd',
                    max_iter=8000
                ).fit(self.training.X, self.training.y)
            elif name == Evaluator._SUPPORT_VECTOR_MACHINE:
                # training classifier on training set
                self.classifiers[name] = SVC(
                    C=10, decision_function_shape='ovo', gamma=10, kernel='rbf'
                ).fit(self.training.X, self.training.y)
            elif name == Evaluator._RANDOM_FOREST:
                # training classifier on training set
                self.classifiers[name] = RandomForestClassifier(
                    criterion='entropy', max_depth=80, max_features='log2',
                    min_samples_leaf=2, n_estimators=600, warm_start=True
                ).fit(self.training.X, self.training.y)
            elif name == Evaluator._ADA_BOOST:
                # training classifier on training set
                dtc = DecisionTreeClassifier(
                    criterion='gini',
                    splitter='best',
                    max_depth=90,
                    max_features=3,
                    min_samples_leaf=4,
                    class_weight='balanced'
                )
                self.classifiers[name] = AdaBoostClassifier(
                    base_estimator=dtc, n_estimators=700
                ).fit(self.training.X, self.training.y)

    def evaluate(self) -> None:
        """
        Evaluate all specified classifiers
        """
        # filter invalid classifiers
        self.__classifiers = {k: v for k, v in self.classifiers.items() if v is not None}
        self.log.info(f"[EVAL] Computing evaluation on test set for: {', '.join(self.classifiers.keys())}")

        for name, classifier in self.classifiers.items():
            accuracy, precision, recall, f1_score, confusion_matrix = Evaluation.evaluate(
                self.classifiers[name],
                self.test.X,
                self.test.y
            )
            self.log.info(
                f"[EVAL] Evaluation of {name}:\n"
                f"\t- Accuracy: {accuracy}\n"
                f"\t- Precision: {precision}\n"
                f"\t- Recall: {recall}\n"
                f"\t- F1-macro: {f1_score}"
            )

    def on_success(self) -> None:
        super().on_success()

        self.log.info(f"Successfully evaluated all (valid) classifiers specified")

    def on_error(self, exception: Exception = None) -> None:
        super().on_error(exception)

        self.log.error(f"Something went wrong during evaluation ({__name__}).", exc_info=True)
    
    @property
    def log(self) -> logging.Logger:
        return self.__LOG

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
