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

    @staticmethod
    def support_vector_machine_param_selection(x: DataFrame, y: DataFrame = None,
                                               n_folds: int = 10, metric: str = 'f1_macro'):
        """

        :param x:
        :param y:
        :param n_folds:
        :param metric:
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
            }
        ]

        grid_search = ms.GridSearchCV(
            svm.SVC(),
            param_grid,
            scoring=metric,
            cv=n_folds,
            refit=True,
            n_jobs=-1
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
    def random_forest_param_selection(x: DataFrame, y: DataFrame = None,
                                      n_folds: int = 10, metric: str = 'f1_macro'):
        """

        :param x:
        :param y:
        :param n_folds:
        :param metric:
        :return:
        """
        param_grid = {
            'max_depth': [80, 90],
            'max_features': [2, 3],
            'min_samples_leaf': [3, 4],
            'n_estimators': [100, 200, 300]
        }

        grid_search = ms.GridSearchCV(
            RandomForestClassifier(),
            param_grid,
            scoring=metric,
            cv=n_folds,
            refit=True,
            n_jobs=-1
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
                                              n_folds: int = 10, metric: str = 'f1_macro'):
        """
        Multi-layer perceptron param selection
        :param x:
        :param y:
        :param n_folds:
        :param metric:
        :return:
        """
        param_grid = {
            'hidden_layer_sizes': [(100, 50, 25), (100, 50), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'learning_rate_init': [0.1, 0.01, 10 ** -3, 10 ** -4],
            'learning_rate': ['constant', 'adaptive']
        }

        grid_search = ms.GridSearchCV(
            MLPClassifier(max_iter=10000),
            param_grid=param_grid,
            scoring=metric,
            cv=n_folds,
            refit=True,
            n_jobs=-1
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
                                           n_folds: int = 10, metric: str = 'f1_macro'):
        """

        :param x:
        :param y:
        :param n_folds:
        :param metric:
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
            cv=n_folds,
            refit=True,
            n_jobs=-1
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
                                                    n_folds: int = 10, metric: str = 'f1_macro'):
        """

        :param x:
        :param y:
        :param n_folds:
        :param metric:
        :return:
        """
        param_grid = {
            'loss': ['hinge', 'log', 'squared_hinge', 'modified_humber'],
            'max_iter': [1000],
            'l1_ratio': [0.08, 0.09, 0.1, 0.12, 0.13, 0.14, 0.15, 0.2],
            'penality': ['l2', 'l1', 'elasticnet']
        }

        grid_search = ms.GridSearchCV(
            SGDClassifier(max_iter=6000),
            param_grid=param_grid,
            scoring=metric,
            cv=n_folds,
            refit=True,
            n_jobs=-1
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
                                    n_folds: int = 10, metric: str = 'f1_macro'):
        """

        :param x:
        :param y:
        :param n_folds:
        :param metric:
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
            cv=n_folds,
            refit=True,
            n_jobs=-1
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
                                     n_folds: int = 10, metric: str = 'f1_macro'):
        """

        :param x:
        :param y:
        :param n_folds:
        :param metric:
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
            AdaBoostClassifier(base_estimator=dtc),
            param_grid=param_grid,
            scoring=metric,
            cv=n_folds,
            refit=True,
            n_jobs=-1
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
                               n_folds: int = 10, metric: str = 'f1_macro'):
        """

        :param x:
        :param y:
        :param n_folds:
        :param metric:
        :return:
        """
        # TODO - vedere se è necessario normalizzare meglio i dati
        clf = KMeans(
            n_clusters=4,
            max_iter=6000,
            algorithm='auto',
            random_state=Conf.get_instance().rng_seed
        )
        clf.fit(x, y)

        return clf
