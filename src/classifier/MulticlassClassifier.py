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
from sklearn.naive_bayes import GaussianNB

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

        # print libs' version
        self.__LOG.debug(f"[LIB VERSION] {np.__name__} : {np.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {pd.__name__} : {pd.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {matplotlib.__name__} : {matplotlib.__version__}")
        self.__LOG.debug(f"[LIB VERSION] {sklearn.__name__} : {sklearn.__version__}")

        # dataset description
        self.__LOG.debug(f"[DESCRIBE] Dataset description:\n{self.data.set_.describe(include='all')}")

        ###########################
        ### manage missing data ###
        ###########################
        # TODO - vedere se performa meglio sostituendo dati mancanti con algo di clustering
        self.__LOG.debug(f"[MISSING DATA] Before processing:\n{PreProcessing.get_na_count(self.data.set_)}")
        for i in range(1, 21):
            mean = self.data.set_["F" + str(i)].mean()
            self.data.set_["F" + str(i)] = self.data.set_["F" + str(i)].fillna(mean)
        self.__LOG.debug(f"[MISSING DATA] After processing:\n{PreProcessing.get_na_count(self.data.set_)}")

        #######################
        ### manage outliers ###
        #######################
        # TODO - vedere se performa meglio la gestione degli outlier con algo di clustering
        z_scores = stats.zscore(self.data.set_)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        self.data.set_ = self.data.set_[filtered_entries]

        counts = self.data.set_['CLASS'].value_counts()
        class1 = counts[0]
        class2 = counts[1]
        class3 = counts[2]
        class4 = counts[3]
        self.__LOG.debug(
            f"[DESCRIBE] Class percentage :\n"
            f"\tC1: {round(class1 / (class1 + class2 + class3 + class4) * 100, 2)} %\n"
            f"\tC2: {round(class2 / (class1 + class2 + class3 + class4) * 100, 2)} %\n"
            f"\tC3: {round(class3 / (class1 + class2 + class3 + class4) * 100, 2)} %\n"
            f"\tC4: {round(class4 / (class1 + class2 + class3 + class4) * 100, 2)} %"
        )

        #########################
        ### splitting dataset ###
        #########################
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
        ##################################
        ### data scaling/normalization ###
        ##################################
        scaler = prep.MinMaxScaler()
        self.__LOG.info(f"[SCALING] Using {type(scaler).__qualname__}")
        # using mean and variance from training set to not overfit
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
        self.__LOG.info(f"[FEATURE SELECTION] Using {type(selector).__qualname__}")
        selector.fit(self.training.set_x, self.training.set_y)
        self.training.set_x = selector.transform(self.training.set_x)
        self.__LOG.debug(f"Feature index after SelectKBest: {selector.get_support(indices=True)}")
        self.test.set_x = selector.transform(self.test.set_x)
        self.__LOG.debug(f"Train shape after feature selection: {self.training.set_x.shape} | {self.training.set_y.shape}")
        self.__LOG.debug(f"Test shape after feature selection: {self.test.set_x.shape} | {self.test.set_y.shape}")

        ###############################
        ### data over/undersampling ###
        ###############################
        # oversampling with SMOTE
        sampler = SMOTE()
        # sampler = RandomUnderSampler()
        # sampler = RandomOverSampler()
        # sampler = ADASYN(sampling_strategy="not majority")
        self.__LOG.info(f"[SAMPLING] Using {type(sampler).__qualname__}")
        self.training.set_x, self.training.set_y = sampler.fit_sample(self.training.set_x, self.training.set_y)
        self.__LOG.debug(f"Train shape after feature selection: {self.training.set_x.shape} | {self.training.set_y.shape}")
        self.__LOG.debug(f"Test shape after feature selection: {self.test.set_x.shape} | {self.test.set_y.shape}")

    def tune(self) -> None:
        # TODO - testare altre macchine
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
            self.training.set_y,
            n_folds=10,
            metric='f1_macro'
        )
        f1_score = metrics.f1_score(self.test.set_y, mlp_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1 score for {multilayer_perceptron}: {f1_score}")

        self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {support_vector_machine}")
        svm_classifier = Tuning.support_vector_machine_param_selection(
            self.training.set_x,
            self.training.set_y,
            n_folds=10,
            metric='f1_macro'
        )
        f1_score = metrics.f1_score(self.test.set_y, svm_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1 score for {support_vector_machine}: {f1_score}")

        self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {random_forest}")
        random_forest_classifier = Tuning.random_forest_param_selection(
            self.training.set_x,
            self.training.set_y,
            n_folds=10,
            metric='f1_macro'
        )
        f1_score = metrics.f1_score(self.test.set_y, random_forest_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1 score for {random_forest}: {f1_score}")

        self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {knearest_neighbors}")
        knn_classifier = Tuning.knearest_neighbors_param_selection(
            self.training.set_x,
            self.training.set_y,
            n_folds=10,
            metric='f1_macro'
        )
        f1_score = metrics.f1_score(self.test.set_y, knn_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1 score for {knearest_neighbors}: {f1_score}")

        self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {stochastic_gradient_descent}")
        sgd_classifier = Tuning.stochastic_gradient_descent_param_selection(
            self.training.set_x,
            self.training.set_y,
            n_folds=10,
            metric='f1_macro'
        )
        f1_score = metrics.f1_score(self.test.set_y, sgd_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1 score for {stochastic_gradient_descent}: {f1_score}")

        self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {ada_boost}")
        ada_boost_classifier = Tuning.ada_boosting_param_selection(
            self.training.set_x,
            self.training.set_y,
            n_folds=10,
            metric='f1_macro'
        )
        f1_score = metrics.f1_score(self.test.set_y, ada_boost_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1 score for {ada_boost}: {f1_score}")

        self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {naive_bayes}")
        naive_bayes_classifier = Tuning.naive_bayes_param_selection(
            self.training.set_x,
            self.training.set_y,
            n_folds=10,
            metric='f1_macro'
        )
        f1_score = metrics.f1_score(self.test.set_y, naive_bayes_classifier.predict(self.test.set_x), average='macro')
        self.__LOG.debug(f"[TUNING] F1 score for {naive_bayes}: {f1_score}")

        # self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {kmeans}")
        # kmeans_classifier = Tuning.kmeans_param_selection(
        #     self.training.set_x,
        #     self.training.set_y,
        #     n_folds=10,
        #     metric='f1_macro'
        # )
        # f1_score = metrics.f1_score(self.test.set_y, kmeans_classifier.predict(self.test.set_x), average='macro')
        # self.__LOG.debug(f"[TUNING] F1 score for {kmeans}: {f1_score}")

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
