import logging
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import sklearn
import sklearn.preprocessing as prep
import sklearn.model_selection as ms
import sklearn.metrics as metrics
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

    RNG_SEED: int = 0
    TEST_RATIO: float = 0.2

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

        # print classifier type
        self.__LOG.info(f"[CLASSIFIER] {MulticlassClassifier.__qualname__}")

        # print libs' version
        self.__LOG.debug(f"[LIB VERSION] {np.__name__} : {np.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {pd.__name__} : {pd.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {matplotlib.__name__} : {matplotlib.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {sklearn.__name__} : {sklearn.__version__}")

        # processing missing data
        self.__LOG.info(f"[DATA PREP] Managing missing data")
        self.data.set_.describe(include='all')
        self.__LOG.debug(f"[DATA PREP] Missing data before processing:\n{PreProcessing.get_na_count(self.data.set_)}")
        for i in range(1, 21):
            mean = self.data.set_["F" + str(i)].mean()
            self.data.set_["F" + str(i)] = self.data.set_["F" + str(i)].fillna(mean)
        self.__LOG.debug(f"[MISSING DATA] Missing data after processing:\n{PreProcessing.get_na_count(self.data.set_)}")

        # manage outlier
        z_scores = stats.zscore(self.data.set_)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        self.data.set_ = self.data.set_[filtered_entries]
        # TODO - vedere se performa meglio la gestione degli outlier con algo di clustering

        counts = self.data.set_['CLASS'].value_counts()
        class1 = counts[0]
        class2 = counts[1]
        class3 = counts[2]
        class4 = counts[3]
        self.__LOG.debug(
            f"[DATA PREP] Class percentage:\n"
            f"\tC1: {round(class1 / (class1 + class2 + class3 + class4) * 100, 2)} %\n"
            f"\tC2: {round(class2 / (class1 + class2 + class3 + class4) * 100, 2)} %\n"
            f"\tC3: {round(class3 / (class1 + class2 + class3 + class4) * 100, 2)} %\n"
            f"\tC4: {round(class4 / (class1 + class2 + class3 + class4) * 100, 2)} %"
        )

        self.__LOG.info(f"Splitting dataset into training and test set with ratio: {MulticlassClassifier.TEST_RATIO}")
        self.data.set_x = self.data.set_.iloc[:, 0:20].values
        self.data.set_y = self.data.set_.iloc[:, 20].values
        # split training/test set
        self.training.set_x, self.test.set_x, self.training.set_y, self.test.set_y = \
            ms.train_test_split(
                self.data.set_x,
                self.data.set_y,
                test_size=MulticlassClassifier.TEST_RATIO,
                random_state=MulticlassClassifier.RNG_SEED
            )

    def refactor(self) -> None:
        # data scaling
        scaler = prep.MinMaxScaler()
        self.__LOG.info(f"[SCALING] Using {type(scaler).__qualname__}")

        # using mean and variance from training set to not overfit
        scaler.fit(self.training.set_x)
        self.training.set_x = scaler.transform(self.training.set_x)
        self.test.set_x = scaler.transform(self.test.set_x)

        # feature selection
        # do not consider the 5 features that have less dependence on the target feature
        # (i.e. the class to which they belong)
        selector = SelectKBest(mutual_info_classif, k=15)
        selector.fit(self.training.set_x, self.training.set_y)
        self.training.set_x = selector.transform(self.training.set_x)
        self.__LOG.debug(f"Feature index after SelectKBest: {selector.get_support(indices=True)}")
        self.test.set_x = selector.transform(self.test.set_x)
        self.__LOG.debug(f"Train shape after feature selection: {self.training.set_x.shape} | {self.training.set_y.shape}")
        self.__LOG.debug(f"Test shape after feature selection: {self.test.set_x.shape} | {self.test.set_y.shape}")

        # oversampling with SMOTE
        sampler = SMOTE()
        # sampler = RandomUnderSampler()
        # sampler = RandomOverSampler()
        # sampler = ADASYN(sampling_strategy="not majority")
        self.training.set_x, self.training.set_y = sampler.fit_sample(self.training.set_x, self.training.set_y)
        self.__LOG.debug(f"Train shape after feature selection: {self.training.set_x.shape} | {self.training.set_y.shape}")
        self.__LOG.debug(f"Test shape after feature selection: {self.test.set_x.shape} | {self.test.set_y.shape}")

    def tune(self) -> None:
        # TODO - testare altre macchine

        # hyper-parameters tuning
        self.__LOG.info(f"[TUNING] Hyper-parameters tuning using: MLP, SVM, RF, KNN, SGD")

        self.__LOG.debug(f"[TUNING] Multi-Layer Perceptron")
        mlp_classifier = Tuning.mlp_param_selection(
            self.training.set_x,
            self.training.set_y,
            n_folds=10,
            metric='f1_macro'
        )
        f1_score = metrics.f1_score(self.test.set_y, mlp_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1(MLP): {f1_score}")

        self.__LOG.debug(f"[TUNING] Support Vector Machine")
        svm_classifier = Tuning.svm_param_selection(
            self.training.set_x,
            self.training.set_y,
            n_folds=10,
            metric='f1_macro'
        )
        f1_score = metrics.f1_score(self.test.set_y, svm_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1(SVM): {f1_score}")

        self.__LOG.debug(f"[TUNING] Random Forest")
        random_forest_classifier = Tuning.random_forest_param_selection(
            self.training.set_x,
            self.training.set_y,
            n_folds=10,
            metric='f1_macro'
        )
        f1_score = metrics.f1_score(self.test.set_y, random_forest_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1(RF): {f1_score}")

        self.__LOG.debug(f"[TUNING] K-Nearest Neighbors")
        knn_classifier = Tuning.knn_param_selection(
            self.training.set_x,
            self.training.set_y,
            n_folds=10,
            metric='f1_macro'
        )
        f1_score = metrics.f1_score(self.test.set_y, knn_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1(KNN): {f1_score}")

        self.__LOG.debug(f"[TUNING] Stochastic Gradient Descent")
        sgd_classifier = Tuning.sgd_param_selection(
            self.training.set_x,
            self.training.set_y,
            n_folds=10,
            metric='f1_macro'
        )
        f1_score = metrics.f1_score(self.test.set_y, sgd_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1(SGD): {f1_score}")

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
