import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import sklearn
import imblearn
import scipy
import sklearn.preprocessing as prep
from joblib import dump
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE

from classifier import AbstractClassifier
from model import Conf, Set
from util import LogManager, Validation, Common
from util.helper import PreProcessing, Tuning, Evaluation
from sklearn.preprocessing import Normalizer

class MulticlassClassifier(AbstractClassifier):
    """
    Class for parameter selection for various models of machine learning
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
            MulticlassClassifier.REQUIRED_PYTHON,
            f"Unsupported Python version.\n"
            f"Required Python {MulticlassClassifier.REQUIRED_PYTHON[0]}.{MulticlassClassifier.REQUIRED_PYTHON[1]} or higher."
        )
        Validation.can_read(
            conf.dataset_train,
            f"Training set file *must* exists and be readable. "
            f"Current file: '{conf.dataset_train}'.\n"
            f"Training set path (fully qualified) can be specified in conf.ini file or using Conf object."
        )

        self.__LOG = LogManager.get_instance().logger(LogManager.Logger.MCC)
        self.__conf = conf

        self.__data = Set(pd.read_csv(self.conf.dataset_train))
        self.__training = Set()
        self.__test = Set()

        # current classifiers used
        self.__classifiers = {
            MulticlassClassifier._MULTILAYER_PERCEPTRON: None,
            MulticlassClassifier._SUPPORT_VECTOR_MACHINE: None,
            MulticlassClassifier._DECISION_TREE: None,
            MulticlassClassifier._RANDOM_FOREST: None,
            MulticlassClassifier._KNEAREST_NEIGHBORS: None,
            # MulticlassClassifier._STOCHASTIC_GRADIENT_DESCENT: None,
            MulticlassClassifier._ADA_BOOST: None,
            MulticlassClassifier._NAIVE_BAYES: None,
            # MulticlassClassifier._KMEANS: None
        }

    def prepare(self) -> None:
        """
        Print library version, classifier type, training set description, class percentage and
          compute pair-plot if requested
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
        self.__LOG.info(f"[MODE] Finding best classifier on data set ({MulticlassClassifier.__qualname__})")

        # dataset description
        self.__LOG.debug(f"[DESCRIPTION] Dataset description:\n{self.data.set_.describe(include='all')}")

        # count data with respect to class
        counts = self.data.set_['CLASS'].value_counts()
        class1 = counts[0]
        class2 = counts[1]
        class3 = counts[2]
        class4 = counts[3]
        self.__LOG.debug(
            f"[DESCRIPTION] Class percentage in dataset :\n"
            f"\tC1: {round(class1 / (class1 + class2 + class3 + class4) * 100, 2)} %\n"
            f"\tC2: {round(class2 / (class1 + class2 + class3 + class4) * 100, 2)} %\n"
            f"\tC3: {round(class3 / (class1 + class2 + class3 + class4) * 100, 2)} %\n"
            f"\tC4: {round(class4 / (class1 + class2 + class3 + class4) * 100, 2)} %"
        )

        # compute pair plot
        if self.conf.pair_plot_compute:
            if self.conf.pair_plot_save:
                self.__LOG.info(
                    f"[DESCRIPTION] Computing and saving pair plot of '{self.conf.dataset_train}' in '{self.conf.tmp}'")
                Validation.can_write(
                    self.conf.tmp,
                    f"Directory '{self.conf.tmp}' *must* exists and be writable."
                )
                destination_file = Path(self.conf.tmp, Path(self.conf.dataset_train).stem).with_suffix('.png')
                PreProcessing.compute_pairplot(self.data.set_, str(destination_file.resolve()))
            else:
                self.__LOG.info(f"[DESCRIPTION] Computing pair plot of '{self.conf.dataset_train}'")
                PreProcessing.compute_pairplot(self.data.set_)

    def split(self) -> None:
        """
        Split training/testing set features and labels
        """
        self.__LOG.info(
            f"[DATA SPLIT] Splitting dataset into training and test set with ratio: {self.conf.dataset_test_ratio}")

        # split training/test set
        self.training.set_ = self.data.set_.sample(
            frac=1 - self.conf.dataset_test_ratio,
            random_state=self.conf.rng_seed
        )
        self.test.set_ = self.data.set_.drop(self.training.set_.index)
        self.data.set_ = None
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
            # TODO - VEDERE SE CON LA MEDIA AL POSTO DELLA MEDIA SI RISOLVE IL PROBLEMA (cambiare nome variabile)
            feature_median_dict[feature] = self.training.set_x[feature].mean()
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
        #scaler = prep.MaxAbsScaler()
        #scaler = prep.QuantileTransformer(output_distribution='uniform')
        #scaler = prep.PowerTransformer(method='yeo-johnson')
        #scaler = prep.PowerTransformer(method='box - cox')

        self.__LOG.info(f"[SCALING] Data scaling using {type(scaler).__qualname__}")
        scaler.fit(self.training.set_x)
        #scaler = Normalizer().fit(self.training.set_x)  # fit does nothing.

        self.training.set_x = scaler.transform(self.training.set_x)
        self.test.set_x = scaler.transform(self.test.set_x)

    def feature_selection(self) -> None:
        """
        Features selection
        """
        # do not consider the 5 features that have less dependence on the target feature
        #   (i.e. the class to which they belong)
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
        """
        Data over/undersampling
        """
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

        # dump classifier, if requested
        dump_directory_path = Path(Common.get_root_path(), MulticlassClassifier._CLASSIFIER_REL_PATH)
        if self.conf.classifier_dump:
            try:
                Validation.is_dir(dump_directory_path)
            except NotADirectoryError:
                self.__LOG.debug(f"[TUNING] Creating folder '{dump_directory_path}'")
                dump_directory_path.mkdir(parents=True, exist_ok=True)

        for name, classifier in self.classifiers.items():
            self.__LOG.debug(f"[TUNING] Hyper-parameter tuning using {name}")
            if name == MulticlassClassifier._MULTILAYER_PERCEPTRON:
                # perform grid search and fit on best evaluator
                self.classifiers[name] = Tuning.multilayer_perceptron_param_selection(
                    self.training.set_x, self.training.set_y, jobs=self.conf.jobs)
            elif name == MulticlassClassifier._SUPPORT_VECTOR_MACHINE:
                # perform grid search and fit on best evaluator
                self.classifiers[name] = Tuning.support_vector_machine_param_selection(
                    self.training.set_x, self.training.set_y, jobs=self.conf.jobs)
            elif name == MulticlassClassifier._DECISION_TREE:
                # perform grid search and fit on best evaluator
                self.classifiers[name] = Tuning.decision_tree_param_selection(
                    self.training.set_x, self.training.set_y, jobs=self.conf.jobs)
            elif name == MulticlassClassifier._RANDOM_FOREST:
                # perform grid search and fit on best evaluator
                self.classifiers[name] = Tuning.random_forest_param_selection(
                    self.training.set_x, self.training.set_y, jobs=self.conf.jobs)
            elif name == MulticlassClassifier._KNEAREST_NEIGHBORS:
                # perform grid search and fit on best evaluator
                self.classifiers[name] = Tuning.knearest_neighbors_param_selection(
                    self.training.set_x, self.training.set_y, jobs=self.conf.jobs)
            elif name == MulticlassClassifier._STOCHASTIC_GRADIENT_DESCENT:
                # perform grid search and fit on best evaluator
                self.classifiers[name] = Tuning.stochastic_gradient_descent_param_selection(
                    self.training.set_x, self.training.set_y, jobs=self.conf.jobs)
            elif name == MulticlassClassifier._ADA_BOOST:
                # perform grid search and fit on best evaluator
                self.classifiers[name] = Tuning.ada_boosting_param_selection(
                    self.training.set_x, self.training.set_y, jobs=self.conf.jobs)
            elif name == MulticlassClassifier._NAIVE_BAYES:
                # perform grid search and fit on best evaluator
                self.classifiers[name] = Tuning.naive_bayes_param_selection(
                    self.training.set_x, self.training.set_y, jobs=self.conf.jobs)
            elif name == MulticlassClassifier._KMEANS:
                # perform grid search and fit on best evaluator
                self.classifiers[name] = Tuning.kmeans_param_selection(
                    self.training.set_x, self.training.set_y, jobs=self.conf.jobs)

            # dump classifier, if requested
            if self.conf.classifier_dump:
                filename = '_'.join(name.split()) + '.joblib'
                filename_path = dump_directory_path.joinpath(filename)
                self.__LOG.debug(f"[TUNING] Dump of {name} in {filename_path}")
                dump(self.classifiers[name], filename_path)

    def evaluate(self) -> None:
        """
        Evaluate all specified classifiers
        """
        self.__LOG.info(f"[EVAL] Computing evaluation for: {', '.join(self.classifiers.keys())}")

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
            self.__LOG.debug(f"[EVAL] Best parameters for {name}: {self.classifiers[name]}")

    def on_success(self) -> None:
        super().on_success()

        self.__LOG.info(f"Successfully trained all specified classifiers")

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

    @property
    def classifiers(self) -> dict:
        return self.__classifiers
