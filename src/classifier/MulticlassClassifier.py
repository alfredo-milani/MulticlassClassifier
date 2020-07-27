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

from classifier import AbstractClassifier
from model import Conf, Set
from util import LogManager, Validation
from util.helper import PreProcessing, Tuning


class MulticlassClassifier(AbstractClassifier):
    """
    Multi-class classifier
    """

    __LOG: logging.Logger = None

    REQUIRED_PYTHON: tuple = (3, 7)

    def __init__(self, conf: Conf):
        super().__init__()

        # validate python version
        Validation.python_version(
            MulticlassClassifier.REQUIRED_PYTHON,
            f"Unsupported Python version.\n"
            f"Required Python {MulticlassClassifier.REQUIRED_PYTHON[0]}.{MulticlassClassifier.REQUIRED_PYTHON[1]} or higher."
        )

        self.__LOG = LogManager.get_instance().logger(LogManager.Logger.MCSVM)
        self.__conf = conf

        self.__data = Set(pd.read_csv(self.conf.dataset))
        self.__training = Set()
        self.__test = Set()

    def prepare(self) -> None:
        super().prepare()

        # print libs' version
        self.__LOG.debug(f"[LIB VERSION] {np.__name__} : {np.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {pd.__name__} : {pd.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {matplotlib.__name__} : {matplotlib.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {sklearn.__name__} : {sklearn.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {imblearn.__name__} : {imblearn.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {scipy.__name__} : {scipy.__version__}")

        # dataset description
        self.__LOG.debug(f"[DESCRIPTION] Dataset description:\n{self.data.set_.describe(include='all')}")

        # count data with respect to class
        counts = self.data.set_['CLASS'].value_counts()
        class1 = counts[0]
        class2 = counts[1]
        class3 = counts[2]
        class4 = counts[3]
        self.__LOG.debug(
            f"[DESCRIPTION] Class percentage :\n"
            f"\tC1: {round(class1 / (class1 + class2 + class3 + class4) * 100, 2)} %\n"
            f"\tC2: {round(class2 / (class1 + class2 + class3 + class4) * 100, 2)} %\n"
            f"\tC3: {round(class3 / (class1 + class2 + class3 + class4) * 100, 2)} %\n"
            f"\tC4: {round(class4 / (class1 + class2 + class3 + class4) * 100, 2)} %"
        )

        #########################
        ### compute pair plot ###
        #########################
        if self.conf.pair_plot_compute:
            self.__LOG.info(f"[DESCRIPTION] Saving pair plot of '{self.conf.dataset}' in '{self.conf.tmp}'")
            Validation.can_write(
                self.conf.tmp,
                f"Directory '{self.conf.tmp}' *must* exists and be writable."
            )
            destination_file = Path(self.conf.tmp, Path(self.conf.dataset).stem).with_suffix('.png')
            PreProcessing.compute_pairplot(self.data.set_, str(destination_file.resolve()))

        #########################
        ### splitting dataset ###
        #########################
        self.__LOG.info(f"[DATA SPLIT] Splitting dataset into training and test set with ratio: {self.conf.dataset_test_ratio}")
        # self.data.set_x = self.data.set_.iloc[:, 0:20].values
        # self.data.set_y = self.data.set_.iloc[:, 20].values
        # split training/test set
        # self.training.set_x, self.test.set_x, self.training.set_y, self.test.set_y = \
        #     ms.train_test_split(
        #         self.data.set_x,
        #         self.data.set_y,
        #         test_size=self.conf.dataset_test_ratio,
        #         random_state=self.conf.rng_seed
        #     )
        self.training.set_ = self.data.set_.sample(
            frac=1 - self.conf.dataset_test_ratio,
            random_state=self.conf.rng_seed
        )
        self.test.set_ = self.data.set_.drop(self.training.set_.index)
        self.data.set_ = None

        self.training.set_y = self.training.set_.pop('CLASS')
        self.training.set_x = self.training.set_
        self.training.set_ = None
        self.test.set_y = self.test.set_.pop('CLASS')
        self.test.set_x = self.test.set_
        self.test.set_ = None

        ###########################
        ### manage missing data ###
        ###########################
        # TODO - vedere se performa meglio sostituendo dati mancanti con algo di clustering
        self.__LOG.info(f"[MISSING DATA] Managing missing data")

        self.__LOG.debug(
            f"[MISSING DATA] Training set x before processing (shape: {self.training.set_x.shape}):\n"
            f"{PreProcessing.get_na_count(self.training.set_x)}"
        )
        self.__LOG.debug(
            f"[MISSING DATA] Test set x before processing (shape: {self.test.set_x.shape}):\n"
            f"{PreProcessing.get_na_count(self.test.set_x)}"
        )

        feature_mean_dict: dict = {}
        for feature in self.training.set_x.columns:
            # using feature mean from training set to manage missing values for
            #  training and test set
            feature_mean_dict[feature] = self.training.set_x[feature].mean()
            self.training.set_x[feature].fillna(feature_mean_dict[feature], inplace=True)
            self.test.set_x[feature].fillna(feature_mean_dict[feature], inplace=True)

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
        # TODO - vedere se performa meglio sostituendo outlier con algo di clustering
        zscore = 'z-score'
        modified_zscore = 'modified z-score'
        iqr = 'inter-quantile range'

        self.__LOG.info(f"[OUTLIER] Managing outlier using {modified_zscore} method")

        self.__LOG.debug(
            f"[DESCRIPTION] Training set x description before manage outlier:\n"
            f"{self.training.set_x.describe(include='all')}"
        )

        # TODO - devono essere sostituiti anche gli outliers del test set ?
        for feature in self.training.set_x.columns:
            # zscore_mask = PreProcessing.zscore(self.training.set_x[feature])
            zscore_mask = PreProcessing.modified_zscore(self.training.set_x[feature])
            # zscore_mask = PreProcessing.iqr(self.training.set_x[feature])
            self.training.set_x.loc[zscore_mask, feature] = feature_mean_dict[feature]

        self.__LOG.debug(
            f"[DESCRIPTION] Training set x description after manage outlier:\n"
            f"{self.training.set_x.describe(include='all')}"
        )

    def refactor(self) -> None:
        ##################################
        ### data scaling/normalization ###
        ##################################
        scaler = prep.MinMaxScaler(feature_range=(0, 1))
        self.__LOG.info(f"[SCALING] Data scaling using {type(scaler).__qualname__}")
        # using following transformation:
        #  X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        #  X_scaled = X_std * (max - min) + min
        scaler.fit(self.training.set_x)
        self.training.set_x = scaler.transform(self.training.set_x)
        self.test.set_x = scaler.transform(self.test.set_x)

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
        self.__LOG.debug(f"[FEATURE SELECTION] Train shape after feature selection: {self.training.set_x.shape} | {self.training.set_y.shape}")
        self.__LOG.debug(f"[FEATURE SELECTION] Test shape after feature selection: {self.test.set_x.shape} | {self.test.set_y.shape}")

        ###############################
        ### data over/undersampling ###
        ###############################
        # oversampling with SMOTE
        sampler = SMOTE(random_state=self.conf.rng_seed)
        # sampler = RandomUnderSampler()
        # sampler = RandomOverSampler()
        # sampler = ADASYN(sampling_strategy='auto', random_state=self.conf.rng_seed)
        self.__LOG.info(f"[SAMPLING] Data sampling using {type(sampler).__qualname__}")
        self.training.set_x, self.training.set_y = sampler.fit_sample(self.training.set_x, self.training.set_y)
        self.__LOG.debug(f"[SAMPLING] Train shape after feature selection: {self.training.set_x.shape} | {self.training.set_y.shape}")
        self.__LOG.debug(f"[SAMPLING] Test shape after feature selection: {self.test.set_x.shape} | {self.test.set_y.shape}")

    def tune(self) -> None:
        # TODO - testare altre macchine o provare a cambiare parametri
        ###############################
        ### hyper-parameters tuning ###
        ###############################

        # current classifiers used
        multilayer_perceptron = 'Multi-Layer Perceptron'
        support_vector_machine = 'Support Vector Machine'
        random_forest = 'Random Forest'
        knearest_neighbors = 'K-Nearest Neighbors'
        stochastic_gradient_descent = 'Stochastic Gradient Descent'
        ada_boost = 'Ada Boost'
        naive_bayes = 'Naive Bayes'
        kmeans = 'K-Means'

        self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {multilayer_perceptron}")
        mlp_classifier = Tuning.multilayer_perceptron_param_selection(
            self.training.set_x,
            self.training.set_y
        )
        f1_score = metrics.f1_score(self.test.set_y, mlp_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1 score for {multilayer_perceptron}: {f1_score}")

        self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {support_vector_machine}")
        svm_classifier = Tuning.support_vector_machine_param_selection(
            self.training.set_x,
            self.training.set_y
        )
        f1_score = metrics.f1_score(self.test.set_y, svm_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1 score for {support_vector_machine}: {f1_score}")

        self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {random_forest}")
        random_forest_classifier = Tuning.random_forest_param_selection(
            self.training.set_x,
            self.training.set_y
        )
        f1_score = metrics.f1_score(self.test.set_y, random_forest_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1 score for {random_forest}: {f1_score}")

        self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {knearest_neighbors}")
        knn_classifier = Tuning.knearest_neighbors_param_selection(
            self.training.set_x,
            self.training.set_y
        )
        f1_score = metrics.f1_score(self.test.set_y, knn_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1 score for {knearest_neighbors}: {f1_score}")

        self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {stochastic_gradient_descent}")
        sgd_classifier = Tuning.stochastic_gradient_descent_param_selection(
            self.training.set_x,
            self.training.set_y
        )
        f1_score = metrics.f1_score(self.test.set_y, sgd_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1 score for {stochastic_gradient_descent}: {f1_score}")

        self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {ada_boost}")
        ada_boost_classifier = Tuning.ada_boosting_param_selection(
            self.training.set_x,
            self.training.set_y
        )
        f1_score = metrics.f1_score(self.test.set_y, ada_boost_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1 score for {ada_boost}: {f1_score}")

        self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {naive_bayes}")
        naive_bayes_classifier = Tuning.naive_bayes_param_selection(
            self.training.set_x,
            self.training.set_y
        )
        f1_score = metrics.f1_score(self.test.set_y, naive_bayes_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1 score for {naive_bayes}: {f1_score}")

        self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {kmeans}")
        kmeans_classifier = Tuning.kmeans_param_selection(
            self.training.set_x,
            self.training.set_y
        )
        f1_score = metrics.f1_score(self.test.set_y, kmeans_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1 score for {kmeans}: {f1_score}")

    def train(self) -> None:
        pass

    def evaluate(self) -> None:
        pass

    def on_success(self) -> None:
        super().on_success()

    def on_error(self, exception: Exception = None) -> None:
        super().on_error(exception)

        self.__LOG.error(f"Something went wrong while training '{__name__}'.", exc_info=True)

    @property
    def conf(self) -> Conf:
        return self.__conf

    @property
    def data(self) -> Set:
        return self.__data

    @property
    def training(self) -> Set:
        return self.__training

    @property
    def test(self) -> Set:
        return self.__test
