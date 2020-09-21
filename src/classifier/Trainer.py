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
import sklearn.model_selection as ms
from matplotlib import pyplot
from sklearn import set_config
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE
from matplotlib import pyplot

from classifier import AbstractClassifier
from model import Conf, Set
from util import LogManager, Validation, Common
from util.helper import PreProcessing, Tuning, Evaluation
from sklearn.preprocessing import Normalizer


class Trainer(AbstractClassifier):
    """
    Class for parameter selection for various models of machine learning
    """

    __LOG: logging.Logger = None

    REQUIRED_PYTHON: tuple = (3, 7)

    # classifiers
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
            Trainer.REQUIRED_PYTHON,
            f"Unsupported Python version.\n"
            f"Required Python {Trainer.REQUIRED_PYTHON[0]}.{Trainer.REQUIRED_PYTHON[1]} or higher."
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
        self.data.X = self.data.w_set.iloc[:, :-1]
        self.data.y = self.data.w_set.iloc[:, -1:]
        self.__training = Set()
        self.__test = Set()

        # current classifiers used
        self.__classifiers = {
            Trainer._MULTILAYER_PERCEPTRON: None,
            Trainer._SUPPORT_VECTOR_MACHINE: None,
            # MulticlassClassifier._DECISION_TREE: None,
            # MulticlassClassifier._RANDOM_FOREST: None,
            # MulticlassClassifier._KNEAREST_NEIGHBORS: None,
            # MulticlassClassifier._STOCHASTIC_GRADIENT_DESCENT: None,
            Trainer._ADA_BOOST: None,
            # MulticlassClassifier._NAIVE_BAYES: None,
            # MulticlassClassifier._KMEANS: None
        }

    def prepare(self) -> None:
        """
        Print library version, classifier type, training set description, class percentage and
          compute pair-plot if requested
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
        self.log.info(f"[MODE] Finding best classifier on data set ({Trainer.__qualname__})")

        # check on directory for charts images
        if self.conf.charts_compute and self.conf.charts_save:
            Validation.can_write(
                self.conf.tmp,
                f"Directory '{self.conf.tmp}' *must* exists and be writable in order to save charts."
            )

        # print all parameters for classifiers
        set_config(print_changed_only=False)

    def data_cleaning(self) -> None:
        """
        Replace missing data with median (as it is not affected by outliers) and
          outliers detected using modified z-score
        """

        ########################
        # DATASET DESCRIPTION
        self.log.debug(f"[DATA CLEANING] Dataset description:\n{self.data.w_set.describe(include='all')}")

        #############################
        # MANAGING USELESS COLUMNS
        self.log.info(f"[DATA CLEANING] Checking useless columns in dataset")
        # summarize the number of unique values in each column
        unique_values = self.data.w_set.nunique()
        self.log.debug(f"[DATA CLEANING] Unique values in each column: \n{unique_values}")
        # record columns to delete
        useless_columns = [i for i, v in enumerate(unique_values) if v == 1]
        self.log.debug(f"[DATA CLEANING] Non unique values to delete: {useless_columns}")
        # drop useless columns
        self.log.debug(f"[DATA CLEANING] Dataset shape: {self.data.w_set.shape}")
        self.data.w_set.drop(useless_columns, axis=1, inplace=True)
        self.log.debug(f"[DATA CLEANING] Dataset shape after dropping columns with only one value: {self.data.w_set.shape}")

        # computing percentage for unique values
        percentage_unique_values_columns = ''
        for feature in self.data.X.columns:
            num = self.data.X[feature].nunique()
            percentage = float(num) / self.data.X.shape[0] * 100
            percentage_unique_values_columns += f"{feature}, {num}, {percentage:.2f} %\n"
        self.log.info(f"[DATA CLEANING] Percentage for unique value in each column: \n{percentage_unique_values_columns}")

        ##################################
        # MANAGING DUPLICATE ROW VALUES
        # calculate duplicates
        duplicates_in_rows = self.data.w_set.duplicated()
        self.log.info(f"[DATA CLEANING] Checking duplicates in rows")
        self.log.debug(f"[DATA CLEANING] Duplicates: \n{self.data.w_set[duplicates_in_rows]}")
        self.log.debug(f"[DATA CLEANING] Dataset shape: {self.data.w_set.shape}")
        self.data.X.drop_duplicates(inplace=True)
        self.log.debug(f"[DATA CLEANING] Dataset shape after dropping row duplicates: {self.data.w_set.shape}")

        #############################################
        # SAMPLES PERCENTAGE WITH RESPECT TO CLASS
        counts = self.data.w_set['CLASS'].value_counts()
        class1 = counts[0]
        class2 = counts[1]
        class3 = counts[2]
        class4 = counts[3]
        self.log.debug(
            f"[DESCRIPTION] Class percentage in dataset :\n"
            f"\tC1: {(class1 / (class1 + class2 + class3 + class4) * 100):.2f} %\n"
            f"\tC2: {(class2 / (class1 + class2 + class3 + class4) * 100):.2f} %\n"
            f"\tC3: {(class3 / (class1 + class2 + class3 + class4) * 100):.2f} %\n"
            f"\tC4: {(class4 / (class1 + class2 + class3 + class4) * 100):.2f} %"
        )

        ############################################
        # SPLITTING DATASET INTO TRAIN / TEST SET
        # TODO

        self.log.info(f"[DATA CLEANING] Managing missing values")

        self.log.debug(
            f"[DATA CLEANING] Training set x before processing (shape: {self.training.X.shape}):\n"
            f"{PreProcessing.get_na_count(self.training.X)}"
        )
        self.log.debug(
            f"[DATA CLEANING] Test set x before processing (shape: {self.test.X.shape}):\n"
            f"{PreProcessing.get_na_count(self.test.X)}"
        )

        # dictionary containing median for each feature
        feature_median_dict: dict = {}
        for feature in self.training.X.columns:
            # using feature median from training set to manage missing values for training and test set
            # it is not used mean as it is affected by outliers and in this phase outliers are still in the training set
            feature_median_dict[feature] = self.training.X[feature].median()
            self.training.X[feature].fillna(feature_median_dict[feature], inplace=True)
            self.test.X[feature].fillna(feature_median_dict[feature], inplace=True)

        self.log.debug(
            f"[MISSING DATA] Training set x after processing (shape: {self.training.X.shape}):\n"
            f"{PreProcessing.get_na_count(self.training.X)}"
        )
        self.log.debug(
            f"[MISSING DATA] Test set x after processing (shape: {self.test.X.shape}):\n"
            f"{PreProcessing.get_na_count(self.test.X)}"
        )

        # manage outliers
        # outliers_manager = 'z-score'
        # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
        # http://colingorrie.github.io/outlier-detection.html#fn:2
        outliers_manager = 'modified z-score'
        # outliers_manager = 'inter-quartile range'

        self.log.info(f"[OUTLIERS] Managing outliers using {outliers_manager} method")

        self.log.debug(
            f"[OUTLIERS] Training set x description before manage outlier:\n"
            f"{self.training.X.describe(include='all')}"
        )

        # outliers management depends on the nature of the individual features.
        outliers_count = 0
        for feature in self.training.X.columns:
            # TODO - VEDI SE FARE CAPPING DEGLI OUTLIERS O SOSTITUIRLI CON LA MEDIANA
            # outliers_mask = PreProcessing.zscore(self.training.set_x[feature])
            outliers_mask = PreProcessing.modified_zscore(self.training.X[feature])
            # outliers_mask = PreProcessing.iqr(self.training.set_x[feature])

            # outliers approximation
            self.training.X.loc[outliers_mask, feature] = feature_median_dict[feature]

            # update outliers_count
            outliers_count += sum(outliers_mask)

        samples_training_set_x = self.training.X.shape[0] * self.training.X.shape[1]
        self.log.debug(f"[OUTLIERS] Outliers detected on all features (F1-F20): "
                         f"{outliers_count} out of {samples_training_set_x} "
                         f"samples ({round(outliers_count / samples_training_set_x * 100, 2)} %)")

        self.log.debug(
            f"[OUTLIERS] Training set x description after manage outlier:\n"
            f"{self.training.X.describe(include='all')}"
        )

    def normalize(self) -> None:
        """
        Normalization and data scaling
        """
        ###################################
        # DATA NORMALIZATION AND SCALING
        scaler = prep.MinMaxScaler(feature_range=(0, 1))
        # scaler = prep.MaxAbsScaler()
        # scaler = prep.QuantileTransformer(output_distribution='uniform')
        # scaler = prep.PowerTransformer(method='yeo-johnson')
        # scaler = prep.PowerTransformer(method='box - cox')

        self.log.info(f"[SCALING] Data scaling using {type(scaler).__qualname__}")
        scaler.fit(self.training.X)
        # scaler = Normalizer().fit(self.training.set_x)  # fit does nothing.

        self.training.X = scaler.transform(self.training.X)
        self.test.X = scaler.transform(self.test.X)

    def feature_selection(self) -> None:
        """
        Features selection
        """
        ########################
        # COMPUTING PAIR-PLOT
        if self.conf.charts_compute:
            if self.conf.charts_save:
                self.log.info(
                    f"[DESCRIPTION] Computing and saving pair plot of '{self.conf.dataset_train}' in '{self.conf.tmp}'")
                destination_file = Path(self.conf.tmp, Path(self.conf.dataset_train).stem).with_suffix('.png')
                PreProcessing.compute_pairplot(self.data.w_set, str(destination_file.resolve()))
            else:
                self.log.info(f"[DESCRIPTION] Computing pair plot of '{self.conf.dataset_train}'")
                PreProcessing.compute_pairplot(self.data.w_set)

        ##############################
        # VARIANCE THRESHOLD METHOD
        thresholds = np.arange(0.0, 5.0, 0.05)
        # apply transform with each threshold
        results = list()
        from sklearn.feature_selection import VarianceThreshold
        for t in thresholds:
            # define the transform
            transform = VarianceThreshold(threshold=t)
            # transform the input data
            X_sel = transform.fit_transform(self.data.X)
            # determine the number of input features
            n_features = X_sel.shape[1]
            print('>Threshold=%.2f, Features=%d' % (t, n_features))
            # store the result
            results.append(n_features)
        if self.conf.charts_compute:
            # plot the threshold vs the number of selected features
            pyplot.plot(thresholds, results)
            pyplot.show()

        ### TODO - PROVARE UN METODO DI TIPO WRAPPER PER OTTENERE L'INSIEME OTTIMO DELLE FEATURES
        ###  see https://www.datacamp.com/community/tutorials/feature-selection-python

        ### TODO - FAI FEATURE ENGINEERING per vedere le migliori features

        ##################
        # K-BEST METHOD
        # do not consider the 5 features that have less dependence on the target feature
        #   (i.e. the class to which they belong)
        selector = SelectKBest(k=15)
        self.log.info(f"[FEATURE SELECTION] Feature selection using {type(selector).__qualname__}")
        selector.fit(self.training.X, self.training.y)
        self.training.X = selector.transform(self.training.X)
        self.log.debug(f"[FEATURE SELECTION] Feature index after SelectKBest: {selector.get_support(indices=True)}")
        self.test.X = selector.transform(self.test.X)
        self.log.debug(
            f"[FEATURE SELECTION] Train shape after feature selection: {self.training.X.shape} | {self.training.y.shape}")
        self.log.debug(
            f"[FEATURE SELECTION] Test shape after feature selection: {self.test.X.shape} | {self.test.y.shape}")

    def sample(self) -> None:
        """
        Data over/undersampling
        """
        ### TODO - TESTARE MEGLIO METODI PER UNDER/OVER SAMPLING
        ###  see https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
        ###  see https://medium.com/quantyca/oversampling-and-undersampling-adasyn-vs-enn-60828a58db39
        # oversampling with SMOTE
        sampler = SMOTE(random_state=self.conf.rng_seed)
        # sampler = RandomUnderSampler(random_state=self.conf.rng_seed)
        # sampler = RandomOverSampler(random_state=self.conf.rng_seed)
        # sampler = ADASYN(sampling_strategy="auto", random_state=self.conf.rng_seed)
        self.log.info(f"[SAMPLING] Data sampling using {type(sampler).__qualname__}")
        self.training.X, self.training.y = sampler.fit_resample(self.training.X, self.training.y)
        self.log.debug(
            f"[SAMPLING] Train shape after feature selection: {self.training.X.shape} | {self.training.y.shape}")
        self.log.debug(
            f"[SAMPLING] Test shape after feature selection: {self.test.X.shape} | {self.test.y.shape}")

    def train(self) -> None:
        """
        Perform Cross-Validation using GridSearchCV to find best hyper-parameter and refit classifiers on
          complete training set
        """
        ### TODO
        ###  - VEDERE SE USARE PIPELINE PER PRE-PROCESSARE TRAINING/VALIDATION SET DURANTE CROSS-VALIDATION
        ###  - PROVARE REPEATED KFOLD E STRATIFIED KFOLD (usando stratified kfold provare con e senza SMOTE)
        ###    - https://towardsdatascience.com/how-to-train-test-split-kfold-vs-stratifiedkfold-281767b93869
        ###    - PROVA STRATIFIED CON 10 FOLDS, VEDI SE RIPROVARE CON LO STRATIFIED SEMPLICE (NON REPEATED) E USANDO FOLD 5 E 10 E CON E SENZA SHUFFLE
        ### TODO
        ###  - SCEGLI TRA ADABOOST E DECISION TREE ED ELIMINA UNO DEI DUE PER LA GRID SEARCH
        ###  - CAMBIA PARAMETRI SVM PER GRID SEARCH (ELIMINA POLY E LINEAR)
        self.log.info(f"[TUNING] Hyper-parameters tuning of: {', '.join(self.classifiers.keys())}")

        dump_directory_path = Path(Common.get_root_path(), Trainer._CLASSIFIER_REL_PATH)
        # create directory for dump classifier, if requested
        if self.conf.classifier_dump:
            try:
                Validation.is_dir(dump_directory_path)
            except NotADirectoryError:
                self.log.debug(f"[TUNING] Creating folder '{dump_directory_path}'")
                dump_directory_path.mkdir(parents=True, exist_ok=True)

        for name, classifier in self.classifiers.items():
            self.log.debug(f"[TUNING] Hyper-parameter tuning using {name}")
            if name == Trainer._MULTILAYER_PERCEPTRON:
                # perform grid search and fit on best evaluator
                self.classifiers[name] = Tuning.multilayer_perceptron_param_selection(
                    self.training.X, self.training.y,
                    cv=ms.RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=self.conf.rng_seed),
                    jobs=self.conf.jobs,
                    random_state=self.conf.rng_seed)
            elif name == Trainer._SUPPORT_VECTOR_MACHINE:
                # perform grid search and fit on best evaluator
                self.classifiers[name] = Tuning.support_vector_machine_param_selection(
                    self.training.X, self.training.y,
                    cv=ms.RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=self.conf.rng_seed),
                    jobs=self.conf.jobs,
                    random_state=self.conf.rng_seed)
            elif name == Trainer._DECISION_TREE:
                # perform grid search and fit on best evaluator
                self.classifiers[name] = Tuning.decision_tree_param_selection(self.training.X,
                                                                              self.training.y,
                                                                              jobs=self.conf.jobs)
            elif name == Trainer._RANDOM_FOREST:
                # perform grid search and fit on best evaluator
                self.classifiers[name] = Tuning.random_forest_param_selection(
                    self.training.X, self.training.y,
                    cv=ms.RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=self.conf.rng_seed),
                    jobs=self.conf.jobs,
                    random_state=self.conf.rng_seed)
            elif name == Trainer._KNEAREST_NEIGHBORS:
                # perform grid search and fit on best evaluator
                self.classifiers[name] = Tuning.knearest_neighbors_param_selection(
                    self.training.X, self.training.y, jobs=self.conf.jobs)
            elif name == Trainer._STOCHASTIC_GRADIENT_DESCENT:
                # perform grid search and fit on best evaluator
                self.classifiers[name] = Tuning.stochastic_gradient_descent_param_selection(
                    self.training.X, self.training.y, jobs=self.conf.jobs)
            elif name == Trainer._ADA_BOOST:
                # perform grid search and fit on best evaluator
                self.classifiers[name] = Tuning.ada_boosting_param_selection(
                    self.training.X, self.training.y,
                    cv=ms.RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=self.conf.rng_seed),
                    jobs=self.conf.jobs,
                    random_state=self.conf.rng_seed)
            elif name == Trainer._NAIVE_BAYES:
                # perform grid search and fit on best evaluator
                self.classifiers[name] = Tuning.naive_bayes_param_selection(
                    self.training.X, self.training.y, jobs=self.conf.jobs)
            elif name == Trainer._KMEANS:
                # perform grid search and fit on best evaluator
                self.classifiers[name] = Tuning.kmeans_param_selection(
                    self.training.X, self.training.y, jobs=self.conf.jobs)

            # dump classifier, if requested
            if self.conf.classifier_dump:
                filename = '_'.join(name.split()) + '.joblib'
                filename_path = dump_directory_path.joinpath(filename)
                self.log.debug(f"[TUNING] Dump of {name} in {filename_path}")
                dump(self.classifiers[name], filename_path)

    def evaluate(self) -> None:
        """
        Evaluate all specified classifiers
        """
        # TESTING
        from sklearn.neural_network import MLPClassifier
        # self.__classifiers['MLP'] = MLPClassifier(
        #     hidden_layer_sizes=(300,),
        #     max_iter=500,
        #     alpha=0.1,
        #     activation='relu',
        #     solver='adam',
        #     random_state=42,
        #     beta_1=0.8,
        #     beta_2=0.888,
        #     shuffle=True,
        # )

        # hidden_layer_sizes=(240, 120)
        self.__classifiers['MLP'] = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(240, 120), learning_rate='adaptive',
              learning_rate_init=0.01, max_fun=15000, max_iter=10000,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=43531, shuffle=True, solver='sgd',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=True)

        self.__classifiers['MLP'].fit(self.training.X, self.training.y)

        # filter invalid classifiers
        self.__classifiers = {k: v for k, v in self.classifiers.items() if v is not None}
        self.log.info(f"[EVAL] Computing evaluation for: {', '.join(self.classifiers.keys())}")

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
                f"\t- F1-score: {f1_score}\n"
                f"\t- Confusion matrix: \n{confusion_matrix}"
            )
            self.log.debug(f"[EVAL] Best parameters for {name}: {self.classifiers[name]}")

    def on_success(self) -> None:
        super().on_success()

        self.log.info(f"Successfully trained all specified classifiers")

    def on_error(self, exception: Exception = None) -> None:
        super().on_error(exception)

        self.log.error(f"Something went wrong while training '{__name__}'.", exc_info=True)

    @property
    def log(self) -> logging.Logger:
        return self.__LOG

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
