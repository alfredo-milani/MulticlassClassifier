import logging
from collections import Counter
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
from joblib import dump
import sklearn.model_selection as ms
from matplotlib import pyplot
from sklearn import set_config
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, SelectPercentile, f_classif, \
    VarianceThreshold, RFECV, RFE
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, BorderlineSMOTE, SVMSMOTE
from matplotlib import pyplot
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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

        self.__data = Set(pd.read_csv(self.conf.dataset_train, na_values=np.nan))
        self.__training = Set()
        self.__test = Set()

        # current classifiers used
        self.__classifiers = {
            Trainer._MULTILAYER_PERCEPTRON: None,
            # Trainer._SUPPORT_VECTOR_MACHINE: None,
            # Trainer._DECISION_TREE: None,
            Trainer._RANDOM_FOREST: None,
            # Trainer._KNEAREST_NEIGHBORS: None,
            # Trainer._STOCHASTIC_GRADIENT_DESCENT: None,
            Trainer._ADA_BOOST: None,
            # Trainer._NAIVE_BAYES: None,
            # Trainer._KMEANS: None
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

    def clean_data(self) -> None:
        """
        Replace missing data with median (as it is not affected by outliers) and outliers detected
        """

        ########################
        # DATASET DESCRIPTION
        self.log.debug(f"[DATA CLEANING] Dataset description:\n{self.data.set_.describe(include='all')}")

        #############################################
        # SAMPLES PERCENTAGE WITH RESPECT TO CLASS
        counts = self.data.set_['CLASS'].value_counts()
        class1 = counts[0]
        class2 = counts[1]
        class3 = counts[2]
        class4 = counts[3]
        self.log.debug(
            f"[DESCRIPTION] Class percentage in training set :\n"
            f"\tC1: {class1} ({(class1 / (class1 + class2 + class3 + class4) * 100):.2f} %)\n"
            f"\tC2: {class2} ({(class2 / (class1 + class2 + class3 + class4) * 100):.2f} %)\n"
            f"\tC3: {class3} ({(class3 / (class1 + class2 + class3 + class4) * 100):.2f} %)\n"
            f"\tC4: {class4} ({(class4 / (class1 + class2 + class3 + class4) * 100):.2f} %)"
        )

        ############################################
        # SPLITTING DATASET INTO TRAIN / TEST SET
        self.log.info(
            f"[DATA CLEANING] Splitting dataset into train / test set with ratio {self.conf.dataset_test_ratio}")
        self.training.set_ = self.data.set_.sample(
            frac=1 - self.conf.dataset_test_ratio,
            random_state=self.conf.rng_seed
        )
        self.test.set_ = self.data.set_.drop(self.training.set_.index)
        self.data.set_ = None

        # percentage of training set with respect to class
        counts = self.training.set_['CLASS'].value_counts()
        class1 = counts[0]
        class2 = counts[1]
        class3 = counts[2]
        class4 = counts[3]
        self.log.debug(
            f"[DESCRIPTION] Class percentage in training set :\n"
            f"\tC1: {class1} ({(class1 / (class1 + class2 + class3 + class4) * 100):.2f} %)\n"
            f"\tC2: {class2} ({(class2 / (class1 + class2 + class3 + class4) * 100):.2f} %)\n"
            f"\tC3: {class3} ({(class3 / (class1 + class2 + class3 + class4) * 100):.2f} %)\n"
            f"\tC4: {class4} ({(class4 / (class1 + class2 + class3 + class4) * 100):.2f} %)"
        )

        # split features and label
        self.training.y = self.training.set_.pop('CLASS')
        self.training.X = self.training.set_
        self.training.set_ = None
        self.test.y = self.test.set_.pop('CLASS')
        self.test.X = self.test.set_
        self.test.set_ = None

        # # ##############################
        # # VARIANCE THRESHOLD METHOD
        # thresholds = np.arange(0.0, 6.0, 0.05)
        # # apply transform with each threshold
        # results = list()
        # for t in thresholds:
        #     # define the transform
        #     transform = VarianceThreshold(threshold=t)
        #     # transform the input data
        #     X_sel = transform.fit_transform(self.training.X)
        #     # determine the number of input features
        #     n_features = X_sel.shape[1]
        #     print('>Threshold=%.2f, Features=%d' % (t, n_features))
        #     # store the result
        #     results.append(n_features)
        # if self.conf.charts_compute:
        #     # plot the threshold vs the number of selected features
        #     pyplot.plot(thresholds, results)
        #     pyplot.show()

        # summarize the number of rows with missing values for each column
        self.log.info(f"[DATA CLEANING] Percentage missing value in training set:")
        for feature in self.training.X.columns:
            # count number of rows with missing values
            n_miss = self.training.X[feature].isnull().sum()
            perc = n_miss / self.training.X.shape[0] * 100
            print(f'> {feature} - missing: {n_miss} ({perc:.1f}%)')
        # summarize the number of rows with missing values for each column
        self.log.info(f"[DATA CLEANING] Percentage missing value in testing set:")
        for feature in self.test.X.columns:
            # count number of rows with missing values
            n_miss = self.test.X[feature].isnull().sum()
            perc = n_miss / self.test.X.shape[0] * 100
            print(f'> {feature} - missing: {n_miss} ({perc:.1f}%)')

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

        # TESTING DONE - IMPUTER FOR MISSING VALUES
        # # evaluate each strategy on the dataset
        # results = list()
        # # SimpleImputer
        # strategies = ['mean', 'median', 'most_frequent', 'constant']
        # # KNNImputer
        # strategies = [str(i) for i in [1, 3, 5, 7, 9, 15, 18, 21]]
        # # IterativeImputer
        # strategies = ['ascending', 'descending', 'roman', 'arabic', 'random']
        # strategies_it = [int(i) for i in range(6, 21)]
        # for si in strategies_it:
        #     for s in strategies:
        #         # create the modeling pipeline
        #         # pipeline = Pipeline(steps=[('i', SimpleImputer(strategy=s)), ('m', MLPClassifier(max_iter=1000))])
        #         # pipeline = Pipeline(steps=[('i', KNNImputer(n_neighbors=int(s), weights='distance')),
        #         #                            ('m', MLPClassifier(max_iter=1000))])
        #         pipeline = Pipeline(steps=[('i', IterativeImputer(max_iter=si, imputation_order=s,
        #                                                           random_state=self.conf.rng_seed, initial_strategy='constant')),
        #                                    ('m', MLPClassifier(max_iter=1000))])
        #         # evaluate the model
        #         cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=self.conf.rng_seed)
        #         scores = cross_val_score(pipeline, self.training.X, self.training.y, scoring='f1_macro',
        #                                  cv=cv, n_jobs=-1, error_score='raise')
        #         # store results
        #         results.append(scores)
        #         print(f'>{s}, max_iter={si} - {np.mean(scores):.3f} ({np.std(scores):.3f})')
        # # plot model performance for comparison
        # pyplot.boxplot(results, labels=strategies, showmeans=True)
        # pyplot.show()

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
        # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
        # http://colingorrie.github.io/outlier-detection.html#fn:2

        # capping data
        # for feature in range(self.training.X.shape[1]):
        #     if (self.training.X[:, feature].dtype == 'float64') or (self.training.X[:, feature].dtype == 'int64'):
        #         percentiles = np.quantile(self.training.X[:, feature], [0.01, 0.99])
        #         print(f"F[{feature}] - P[0]: {np.sum(self.training.X[:, feature] <= percentiles[0])}")
        #         print(f"F[{feature}] - P[1]: {np.sum(self.training.X[:, feature] >= percentiles[1])}")
        #         self.training.X[self.training.X[:, feature] <= percentiles[0], feature] = percentiles[0]
        #         self.training.X[self.training.X[:, feature] >= percentiles[1], feature] = percentiles[1]

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

        # TESTING - OUTLIERS (non so se farlo: vedere se è meglio con sostituzione singoli outliers con imputer o facendo capping)
        # for c in class_err:
        #     # outliers_detector = LocalOutlierFactor(n_neighbors=n, metric=m)
        #     # outliers_detector = IsolationForest(n_estimators=e, contamination=c, max_features=f, bootstrap=True,
        #     #                                     n_jobs=self.conf.jobs, random_state=self.conf.rng_seed,
        #     #                                     warm_start=True)
        #     # outliers_detector = EllipticEnvelope(support_fraction=f, contamination=c, random_state=self.conf.rng_seed)
        #     outliers_detector = OneClassSVM(nu=c)
        #     yhat = outliers_detector.fit_predict(self.training.X)
        #     # select all rows that are not outliers
        #     mask = yhat != -1
        #     self.training.X, self.training.y = self.training.X[mask, :], self.training.y[mask]
        #
        #     # evaluate the model
        #     cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=self.conf.rng_seed)
        #     scores = cross_val_score(MLPClassifier(max_iter=1500), self.training.X, self.training.y, scoring='f1_macro',
        #                              cv=cv, n_jobs=-1, error_score='raise')
        #     # store results
        #     results.append(scores)
        #     print(f'>nu={c} - '
        #           f'{np.mean(scores):.3f} ({np.std(scores):.3f}) - #outliers={np.sum(~mask)}')

        # removing outliers
        # self.training.y = self.training.y[~np.isnan(self.training.X).any(axis=1)]
        # self.training.X = self.training.X[~np.isnan(self.training.X).any(axis=1)]
        # replace outliers using imputer
        # imputer = IterativeImputer(max_iter=10, imputation_order='ascending',
        #                            random_state=self.conf.rng_seed, initial_strategy='constant')
        # # fit on the dataset
        # imputer.fit(self.training.X)
        # # transform the dataset
        # self.training.X = imputer.transform(self.training.X)

        # TESTING DONE - AUTOMATIC OUTLIERS DETECTION
        # evaluate each strategy on the dataset
        # results = list()
        # LocalOutlierFactor
        # neighbors = [5, 10, 15, 20, 25, 30]
        # metric = ['dice', 'correlation', 'l2', 'euclidean', 'minkowski']
        # IsolationForest
        # estimator = [30, 60, 90]
        # contamination = [0.05, 0.1, 0.2, 0.3]
        # max_features = [10.0, 15.0, 20.0]
        # EllipticEnvelope
        # support_fraction = [30, 60, 90]
        # contamination = [0.05, 0.1, 0.2, 0.3]
        # OneClassSVM
        # class_err = [0.05, 0.1, 0.2, 0.3]
        # for c in class_err:
        #     # outliers_detector = LocalOutlierFactor(n_neighbors=n, metric=m)
        #     # outliers_detector = IsolationForest(n_estimators=e, contamination=c, max_features=f, bootstrap=True,
        #     #                                     n_jobs=self.conf.jobs, random_state=self.conf.rng_seed,
        #     #                                     warm_start=True)
        #     # outliers_detector = EllipticEnvelope(support_fraction=f, contamination=c, random_state=self.conf.rng_seed)
        #     outliers_detector = OneClassSVM(nu=c)
        #     yhat = outliers_detector.fit_predict(self.training.X)
        #     # select all rows that are not outliers
        #     mask = yhat != -1
        #     self.training.X, self.training.y = self.training.X[mask, :], self.training.y[mask]
        #
        #     # evaluate the model
        #     cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=self.conf.rng_seed)
        #     scores = cross_val_score(MLPClassifier(max_iter=1500), self.training.X, self.training.y, scoring='f1_macro',
        #                              cv=cv, n_jobs=-1, error_score='raise')
        #     # store results
        #     results.append(scores)
        #     print(f'>nu={c} - '
        #           f'{np.mean(scores):.3f} ({np.std(scores):.3f}) - #outliers={np.sum(~mask)}')

    def feature_selection(self) -> None:
        """
        Features selection
        https://www.datacamp.com/community/tutorials/feature-selection-python
        """
        ########################
        # COMPUTING PAIR-PLOT
        if self.conf.charts_compute:
            if self.conf.charts_save:
                self.log.info(
                    f"[DESCRIPTION] Computing and saving pair plot of '{self.conf.dataset_train}' in '{self.conf.tmp}'")
                destination_file = Path(self.conf.tmp, Path(self.conf.dataset_train).stem).with_suffix('.png')
                PreProcessing.compute_pairplot(self.data.set_, str(destination_file.resolve()))
            else:
                self.log.info(f"[DESCRIPTION] Computing pair plot of '{self.conf.dataset_train}'")
                PreProcessing.compute_pairplot(self.data.set_)

        # TESTING DONE - FEATURES DIMENSIONALITY REDUCTION
        # # PCA, Isomap
        # # features = ['mle', 5, 10, 15]
        # # LDA
        # features = [1, 2, 3]
        # # Isomap
        # features = [10, 15]
        # # Isomap, LocallyLinearEmbedding
        # neighbors = [5, 7, 10, 15, 20]
        # for f in features:
        #     for n in neighbors:
        #         # define the pipeline
        #         model = MLPClassifier(max_iter=3000)
        #         # dr = PCA(n_components=f)
        #         # dr = LinearDiscriminantAnalysis(n_components=f)
        #         # dr = Isomap(n_neighbors=n, n_components=f, n_jobs=self.conf.jobs)
        #         dr = LocallyLinearEmbedding(n_components=f, n_neighbors=n, random_state=self.conf.rng_seed, n_jobs=self.conf.jobs)
        #         steps = [('dr', dr), ('m', model)]
        #         pipeline = Pipeline(steps=steps)
        #         # evaluate model
        #         cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=self.conf.rng_seed)
        #         scores = cross_val_score(pipeline, self.training.X, self.training.y, scoring='f1_macro',
        #                                  cv=cv, n_jobs=self.conf.jobs, error_score='raise')
        #         print(f'> comp={f}, neigh={n} - {np.mean(scores):.3f} ({np.std(scores):.3f})')

        # TESTING DONE - FEATURE SELECTION WRAPPER (RFE)
        # features_sel = [15, 19, 20]
        # for features in features_sel:
        #     models = dict()
        #     # lr
        #     from sklearn.linear_model import LogisticRegression
        #     rfe = RFE(estimator=LogisticRegression(max_iter=1500), n_features_to_select=features)
        #     model = MLPClassifier(max_iter=1500)
        #     # models['lr'] = Pipeline(steps=[('s', rfe), ('m', model)])
        #     oversampling = SVMSMOTE(svm_estimator=SVC(), k_neighbors=5, m_neighbors=5,
        #                             n_jobs=self.conf.jobs, random_state=self.conf.rng_seed)
        #     models['lr'] = Pipeline(steps=[('o', oversampling), ('s', rfe), ('m', model)])
        #     # perceptron
        #     rfe = RFE(estimator=Perceptron(max_iter=1500), n_features_to_select=features)
        #     model = MLPClassifier(max_iter=1500)
        #     # models['lr'] = Pipeline(steps=[('s', rfe), ('m', model)])
        #     oversampling = SVMSMOTE(svm_estimator=SVC(), k_neighbors=5, m_neighbors=5,
        #                             n_jobs=self.conf.jobs, random_state=self.conf.rng_seed)
        #     models['lr'] = Pipeline(steps=[('o', oversampling), ('s', rfe), ('m', model)])
        #     # cart
        #     rfe = RFE(estimator=DecisionTreeClassifier(criterion='entropy', max_depth=90,
        #                                                max_features=None, splitter='best', min_samples_leaf=1, min_samples_split=2),
        #               n_features_to_select=features)
        #     model = MLPClassifier(max_iter=1500)
        #     # models['lr'] = Pipeline(steps=[('s', rfe), ('m', model)])
        #     oversampling = SVMSMOTE(svm_estimator=SVC(), k_neighbors=5, m_neighbors=5,
        #                             n_jobs=self.conf.jobs, random_state=self.conf.rng_seed)
        #     models['lr'] = Pipeline(steps=[('o', oversampling), ('s', rfe), ('m', model)])
        #     # rf
        #     rfe = RFE(estimator=RandomForestClassifier(criterion='entropy', max_depth=90,
        #                                                max_features='log2', n_estimators=400, min_samples_leaf=2, min_samples_split=2),
        #               n_features_to_select=features)
        #     model = MLPClassifier(max_iter=1500)
        #     # models['lr'] = Pipeline(steps=[('s', rfe), ('m', model)])
        #     oversampling = SVMSMOTE(svm_estimator=SVC(), k_neighbors=5, m_neighbors=5,
        #                             n_jobs=self.conf.jobs, random_state=self.conf.rng_seed)
        #     models['lr'] = Pipeline(steps=[('o', oversampling), ('s', rfe), ('m', model)])
        #     # gbm
        #     rfe = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=features)
        #     model = MLPClassifier(max_iter=1500)
        #     # models['lr'] = Pipeline(steps=[('s', rfe), ('m', model)])
        #     oversampling = SVMSMOTE(svm_estimator=SVC(), k_neighbors=5, m_neighbors=5,
        #                             n_jobs=self.conf.jobs, random_state=self.conf.rng_seed)
        #     models['lr'] = Pipeline(steps=[('o', oversampling), ('s', rfe), ('m', model)])
        #
        #     ###########
        #     results, names = list(), list()
        #     for name, model in models.items():
        #         cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=self.conf.rng_seed)
        #         scores = cross_val_score(model, self.training.X, self.training.y, scoring='f1_macro',
        #                                  cv=cv, n_jobs=self.conf.jobs, error_score='raise')
        #         results.append(scores)
        #         names.append(name)
        #         print(f'> #feat={features}, name={name} - {np.mean(scores):.3f} ({np.std(scores):.3f})')
        #     # plot model performance for comparison
        #     # pyplot.boxplot(results, labels=names, showmeans=True)
        #     # pyplot.show()

        # TESTING DONE - FEATURE SELECTION FILTER
        # # SelectKBest
        # # number of features to evaluate
        # num_features = [10, 15, 16, 17, 18, 19, 20]
        # func = [f_classif, mutual_info_classif]
        # # SelectPercentile
        # percentiles = [50, 75, 80, 85, 90, 95, 100]  # features: 3, 4, 5, 8, 9, 10, 15, 20
        # # enumerate each number of features
        # results = list()
        # for f in func:
        #     for k in num_features:
        #         # create pipeline
        #         model = MLPClassifier(max_iter=1500)
        #         fs = SelectKBest(score_func=f, k=k)
        #         # fs = SelectPercentile(score_func=f, percentile=p)
        #         pipeline = Pipeline(steps=[('fs', fs), ('lr', model)])
        #         # evaluate pipeline
        #         cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=self.conf.rng_seed)
        #         scores = cross_val_score(pipeline, self.training.X, self.training.y, scoring='f1_macro',
        #                                  cv=cv, n_jobs=self.conf.jobs, error_score='raise')
        #         results.append(scores)
        #         # summarize the results
        #         print(f'> func={f.__qualname__}, #features={k} - {np.mean(scores):.3f} ({np.std(scores):.3f})')
        #     # plot model performance for comparison
        #     # pyplot.boxplot(results, labels=num_features, showmeans=True)
        #     # pyplot.show()
        #     # results = list()

        #######################
        # FEATURE SELECTION
        # selector = SelectPercentile(score_func=mutual_info_classif, percentile=100)
        # selector = RFE(estimator=LogisticRegression(max_iter=1500), n_features_to_select=15)
        selector = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=15)
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
        # TESTING DONE - SAMPLING
        # # SMOTE, BorderlineSMOTE, SVMSMOTE
        # k_neighbors = [1, 2, 3, 5, 7, 8, 10]
        # # BorderlineSMOTE
        # kind = ['borderline-1', 'borderline-2']
        # # BorderlineSMOTE, SVMSMOTE
        # m_neighbors = [5, 10, 15, 20]
        # # SVMSMOTE
        # from sklearn.svm import SVC
        # estimators = [SVC(), SVC(C=10, decision_function_shape='ovo', gamma=10, kernel='rbf')]
        # for k in k_neighbors:
        #     # define pipeline
        #     model = MLPClassifier(max_iter=1500)
        #     # over = SMOTE(sampling_strategy={1: 1700, 2: 1700}, k_neighbors=k, n_jobs=self.conf.jobs,
        #     #              random_state=self.conf.rng_seed)
        #     # under = RandomUnderSampler(sampling_strategy={1: 1700, 2: 1700}, random_state=self.conf.rng_seed)
        #     # over = SMOTE(k_neighbors=k, n_jobs=self.conf.jobs, random_state=self.conf.rng_seed)
        #     # over = BorderlineSMOTE(k_neighbors=k, m_neighbors=m, kind=kk, n_jobs=self.conf.jobs, random_state=self.conf.rng_seed)
        #     # over = SVMSMOTE(svm_estimator=e, k_neighbors=k, m_neighbors=m, n_jobs=self.conf.jobs, random_state=self.conf.rng_seed)
        #     over = ADASYN(sampling_strategy='minority', n_neighbors=k, n_jobs=self.conf.jobs, random_state=self.conf.rng_seed)
        #     # under = RandomUnderSampler(random_state=self.conf.rng_seed)
        #     # steps = [('over', over), ('under', under), ('model', model)]
        #     steps = [('over', over), ('model', model)]
        #     pipeline = Pipeline(steps=steps)
        #     # evaluate pipeline
        #     cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=self.conf.rng_seed)
        #     scores = cross_val_score(pipeline, self.training.X, self.training.y, scoring='f1_macro',
        #                              cv=cv, n_jobs=self.conf.jobs, error_score='raise')
        #     print(f'> k_neigh={k} - {np.mean(scores):.3f} ({np.std(scores):.3f})')

        self.log.debug(
            f"[SAMPLING] Train shape: {self.training.X.shape} | {self.training.y.shape}")
        # oversampling with SMOTE and RandomUnderSampler
        oversampling = SVMSMOTE(svm_estimator=SVC(), k_neighbors=5, m_neighbors=5,
                                n_jobs=self.conf.jobs, random_state=self.conf.rng_seed)
        # undersampling = RandomUnderSampler(random_state=self.conf.rng_seed)
        # steps = [('oversampling', oversampling), ('undersampling', undersampling)]
        steps = [('oversampling', oversampling)]
        sampler = Pipeline(steps=steps)
        self.log.info(f"[SAMPLING] Data sampling using {type(sampler).__qualname__}")
        self.training.X, self.training.y = sampler.fit_resample(self.training.X, self.training.y)
        self.log.debug(
            f"[SAMPLING] Train shape after data sampling: {self.training.X.shape} | {self.training.y.shape}")

        if self.conf.charts_compute:
            # summarize the new class distribution
            counter = Counter(self.training.y)
            # scatter plot of examples by class label
            for label, _ in counter.items():
                row_ix = np.where(self.training.y == label)[0]
                pyplot.scatter(self.training.X[row_ix, 0], self.training.X[row_ix, 1], label=str(label))
            pyplot.legend()
            pyplot.show()

    def train(self) -> None:
        """
        Perform Cross-Validation using GridSearchCV to find best hyper-parameter and refit classifiers on
          complete training set
        """
        ### TODO
        ###  - VEDERE SE USARE PIPELINE PER PRE-PROCESSARE TRAINING/VALIDATION SET DURANTE CROSS-VALIDATION
        ###  - PROVARE REPEATED KFOLD E STRATIFIED KFOLD (usando stratified kfold provare con e senza SMOTE)
        ###    - https://towardsdatascience.com/how-to-train-test-split-kfold-vs-stratifiedkfold-281767b93869
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
        # from sklearn.neural_network import MLPClassifier
        #
        # # hidden_layer_sizes=(240, 120)
        # self.__classifiers['MLP'] = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
        #                                           beta_2=0.999, early_stopping=False, epsilon=1e-08,
        #                                           hidden_layer_sizes=(150, 100), learning_rate='adaptive',
        #                                           learning_rate_init=0.01, max_fun=15000, max_iter=10000,
        #                                           momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
        #                                           power_t=0.5, random_state=self.conf.rng_seed, shuffle=True,
        #                                           solver='sgd',
        #                                           tol=0.0001, validation_fraction=0.1, verbose=False,
        #                                           warm_start=True)
        #
        # self.__classifiers['MLP'].fit(self.training.X, self.training.y)

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
                f"\t- F1-macro: {f1_score}\n"
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
