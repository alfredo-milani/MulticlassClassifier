import sklearn.model_selection as ms
import sklearn.svm as svm
from pandas import DataFrame
from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from model import Conf


class Tuning(object):
    """
    Contains helper methods for hyperparameter tuning
    """

    DEFAULT_CV = 10
    DEFAULT_METRIC: str = 'f1_macro'
    DEFAULT_THREAD: int = -1
    DEFAULT_RANDOM_STATE = 0

    @staticmethod
    def support_vector_machine_param_selection(x: DataFrame, y: DataFrame = None,
                                               cv=DEFAULT_CV, metric: str = DEFAULT_METRIC,
                                               jobs: int = DEFAULT_THREAD,
                                               random_state: int = DEFAULT_RANDOM_STATE):
        """

        :param x:
        :param y:
        :param cv:
        :param metric:
        :param jobs:
        :param random_state:
        :return:
        """

        param_grid = [
            {
                'kernel': ['linear'],
                'C': [0.1, 1, 10],
                'decision_function_shape': ['ovo', 'ovr']
            },
            {
                'kernel': ['rbf'],
                'gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1e+1, 1e+2, 1e+3, 1e+4],
                'C': [0.1, 1, 10, 50, 100],
                'decision_function_shape': ['ovo', 'ovr']
            },
            {
                'kernel': ['poly'],
                'degree': [2, 3, 4],
                'gamma': ['scale'],
                'C': [0.1, 1, 10],
                'decision_function_shape': ['ovo', 'ovr']
            }
        ]

        grid_search = ms.GridSearchCV(
            svm.SVC(random_state=random_state),
            param_grid,
            scoring=metric,
            cv=cv,
            refit=True,
            n_jobs=jobs
        )
        grid_search.fit(x, y)

        print("Best parameters:")
        print()
        print(grid_search.best_params_)
        print()
        print("Grid scores:")
        print()
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
            print("%0.4f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        return grid_search.best_estimator_

    @staticmethod
    def decision_tree_param_selection(x: DataFrame, y: DataFrame = None,
                                      cv=DEFAULT_CV, metric: str = DEFAULT_METRIC,
                                      jobs: int = DEFAULT_THREAD,
                                      random_state: int = DEFAULT_RANDOM_STATE):
        """

        :param x:
        :param y:
        :param cv:
        :param metric:
        :param jobs:
        :param random_state:
        :return:
        """

        param_grid = {
            'criterion': ['entropy', 'gini'],
            'splitter': ['best', 'random'],
            'max_depth': [80, 90],
            'max_features': ['log2', 'sqrt', None],
            'min_samples_leaf': [2, 5, 10]
        }

        grid_search = ms.GridSearchCV(
            DecisionTreeClassifier(random_state=random_state),
            param_grid,
            scoring=metric,
            cv=cv,
            refit=True,
            n_jobs=jobs
        )
        grid_search.fit(x, y)

        print("Best parameters:")
        print()
        print(grid_search.best_params_)
        print()
        print("Grid scores:")
        print()
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
            print("%0.4f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        return grid_search.best_estimator_

    @staticmethod
    def random_forest_param_selection(x: DataFrame, y: DataFrame = None,
                                      cv=DEFAULT_CV, metric: str = DEFAULT_METRIC,
                                      jobs: int = DEFAULT_THREAD,
                                      random_state: int = DEFAULT_RANDOM_STATE):
        """

        :param x:
        :param y:
        :param cv:
        :param metric:
        :param jobs:
        :param random_state:
        :return:
        """

        param_grid = {
            'criterion': ['entropy', 'gini'],
            'max_depth': [80, 90],
            'max_features': ['log2', 'sqrt', None],
            'min_samples_leaf': [2, 5, 10],
            'n_estimators': [100, 200, 300, 400, 500]
        }

        grid_search = ms.GridSearchCV(
            RandomForestClassifier(random_state=random_state),
            param_grid,
            scoring=metric,
            cv=cv,
            refit=True,
            n_jobs=jobs
        )
        grid_search.fit(x, y)

        print("Best parameters:")
        print()
        print(grid_search.best_params_)
        print()
        print("Grid scores:")
        print()
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
            print("%0.4f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        return grid_search.best_estimator_

    @staticmethod
    def multilayer_perceptron_param_selection(x: DataFrame, y: DataFrame = None,
                                              cv=DEFAULT_CV, metric: str = DEFAULT_METRIC,
                                              jobs: int = DEFAULT_THREAD,
                                              random_state: int = DEFAULT_RANDOM_STATE):
        """

        :param x:
        :param y:
        :param cv:
        :param metric:
        :param jobs:
        :param random_state:
        :return:
        """

        param_grid = {
            # 'hidden_layer_sizes': [(100, 50, 25), (100, 50), (100,), (75,), (45,)],
            # 'hidden_layer_sizes': [(150, 100), (120, 60), (60, 30), (75,), (45,)],
            'hidden_layer_sizes': [(200, 150), (240, 120), (150, 100), (120, 60)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'learning_rate_init': [1e-1, 1e-2, 1e-3, 1e-4],
            'learning_rate': ['constant', 'adaptive']
        }

        grid_search = ms.GridSearchCV(
            MLPClassifier(max_iter=10000, random_state=random_state),
            param_grid=param_grid,
            scoring=metric,
            cv=cv,
            refit=True,
            n_jobs=jobs
        )
        grid_search.fit(x, y)

        print("Best parameters:")
        print()
        print(grid_search.best_params_)
        print()
        print("Grid scores:")
        print()
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
            print("%0.4f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        return grid_search.best_estimator_

    @staticmethod
    def knearest_neighbors_param_selection(x: DataFrame, y: DataFrame = None,
                                           cv=DEFAULT_CV, metric: str = DEFAULT_METRIC,
                                           jobs: int = DEFAULT_THREAD,
                                           random_state: int = DEFAULT_RANDOM_STATE):
        """

        :param x:
        :param y:
        :param cv:
        :param metric:
        :param jobs:
        :param random_state:
        :return:
        """

        param_grid = {
            'n_neighbors': [3, 5, 7, 11],
            'metric': ['minkowski', 'euclidean', 'chebyshev'],
            'p': [3, 4, 5]
        }

        grid_search = ms.GridSearchCV(
            KNeighborsClassifier(),
            param_grid=param_grid,
            scoring=metric,
            cv=cv,
            refit=True,
            n_jobs=jobs
        )
        grid_search.fit(x, y)

        print("Best parameters:")
        print()
        print(grid_search.best_params_)
        print()
        print("Grid scores:")
        print()
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
            print("%0.4f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        return grid_search.best_estimator_

    @staticmethod
    def stochastic_gradient_descent_param_selection(x: DataFrame, y: DataFrame = None,
                                                    cv=DEFAULT_CV, metric: str = DEFAULT_METRIC,
                                                    jobs: int = DEFAULT_THREAD,
                                                    random_state: int = DEFAULT_RANDOM_STATE):
        """

        :param x:
        :param y:
        :param cv:
        :param metric:
        :param jobs:
        :param random_state:
        :return:
        """

        param_grid = {
            'loss': ['hinge', 'log', 'squared_hinge', 'modified_huber'],
            'max_iter': [1000],
            'l1_ratio': [0.08, 0.09, 0.1, 0.12, 0.13, 0.14, 0.15, 0.2],
            'penality': ['elasticnet', 'l2', 'l1']
        }

        grid_search = ms.GridSearchCV(
            SGDClassifier(max_iter=6000, random_state=random_state),
            param_grid=param_grid,
            scoring=metric,
            cv=cv,
            refit=True,
            n_jobs=jobs
        )
        grid_search.fit(x, y)
        print("Best parameters:")
        print()
        print(grid_search.best_params_)
        print()
        print("Grid scores:")
        print()
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
            print("%0.4f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        return grid_search.best_estimator_

    @staticmethod
    def naive_bayes_param_selection(x: DataFrame, y: DataFrame = None,
                                    cv=DEFAULT_CV, metric: str = DEFAULT_METRIC,
                                    jobs: int = DEFAULT_THREAD,
                                    random_state: int = DEFAULT_RANDOM_STATE):
        """

        :param x:
        :param y:
        :param cv:
        :param metric:
        :param jobs:
        :param random_state:
        :return:
        """

        param_grid = {
            'priors': [None, [0.25, 0.25, 0.25, 0.25]],
            'var_smoothing': [10e-9, 10e-6, 10e-3, 10e-1]
        }

        grid_search = ms.GridSearchCV(
            GaussianNB(),
            param_grid=param_grid,
            scoring=metric,
            cv=cv,
            refit=True,
            n_jobs=jobs
        )
        grid_search.fit(x, y)

        print("Best parameters:")
        print()
        print(grid_search.best_params_)
        print()
        print("Grid scores:")
        print()
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
            print("%0.4f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        return grid_search.best_estimator_

    @staticmethod
    def ada_boosting_param_selection(x: DataFrame, y: DataFrame = None,
                                     cv=DEFAULT_CV, metric: str = DEFAULT_METRIC,
                                     jobs: int = DEFAULT_THREAD,
                                     random_state: int = DEFAULT_RANDOM_STATE):
        """

        :param x:
        :param y:
        :param cv:
        :param metric:
        :param jobs:
        :param random_state:
        :return:
        """

        param_grid = {
            'base_estimator__criterion': ['gini', 'entropy'],
            'base_estimator__splitter': ['best', 'random'],
            'n_estimators': [100, 200, 300]
        }

        dtc = DecisionTreeClassifier(
            max_depth=90,
            max_features=3,
            min_samples_leaf=4,
            class_weight='balanced'
        )
        grid_search = ms.GridSearchCV(
            AdaBoostClassifier(base_estimator=dtc, random_state=random_state),
            param_grid=param_grid,
            scoring=metric,
            cv=cv,
            refit=True,
            n_jobs=jobs
        )
        grid_search.fit(x, y)

        print("Best parameters:")
        print()
        print(grid_search.best_params_)
        print()
        print("Grid scores:")
        print()
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
            print("%0.4f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        return grid_search.best_estimator_

    @staticmethod
    def kmeans_param_selection(x: DataFrame, y: DataFrame = None,
                               cv=DEFAULT_CV, metric: str = DEFAULT_METRIC,
                               jobs: int = DEFAULT_THREAD,
                               random_state: int = DEFAULT_RANDOM_STATE):
        """

        :param x:
        :param y:
        :param cv:
        :param metric:
        :param jobs:
        :param random_state:
        :return:
        """

        clf = KMeans(
            n_clusters=4,
            max_iter=10000,
            algorithm='auto',
            random_state=random_state
        )
        clf.fit(x, y)

        return clf
