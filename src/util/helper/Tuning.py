import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier


class Tuning(object):
    """
    Contains helper methods for hyperparameter tuning
    """

    @staticmethod
    def k_fold_cross_validation_svm(x, y, train_x, k=5, C=1, kernel='linear', degree=3, gamma='auto'):
        """

        :param x:
        :param y:
        :param train_x:
        :param k:
        :param C:
        :param kernel:
        :param degree:
        :param gamma:
        :return:
        """
        avg_score = 0
        cv = ms.KFold(n_splits=k, random_state=0)
        # https://scikit-learn.org/stable/index.html
        classifier = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
        for train_index, test_index in cv.split(train_x):
            fold_train_x, fold_test_x = x[train_index], x[test_index]
            fold_train_y, fold_test_y = y[train_index], y[test_index]
            classifier.fit(fold_train_x, fold_train_y)
            fold_pred_y = classifier.predict(fold_test_x)
            # accuracy = percentuale esempi classificati correttamente
            score = metrics.accuracy_score(fold_test_y, fold_pred_y)
            print(score)
            avg_score += score
        avg_score = avg_score / k
        return avg_score

    @staticmethod
    def grid_search_linear_svm(x, y, k, train_x, c_list):
        """
        Grid search for hyperparameters tuning with SVM with linear kernel
        :param x:
        :param y:
        :param k:
        :param train_x:
        :param c_list:
        :return:
        """
        best_score = 0
        best_c = None
        for c in c_list:
            score = Tuning.k_fold_cross_validation_svm(x, y, train_x, k=k, C=c, kernel='linear')
            print('C =', c, 'accuracy =', score)
            if score > best_score:
                best_score = score
                best_c = c
        return best_score, best_c

    @staticmethod
    def grid_search_rbf_svm(x, y, k, train_x, c_list, gamma_list):
        """
        Grid search for hyperparameters tuning with SVM with gaussian kernel
        :param x:
        :param y:
        :param k:
        :param train_x:
        :param c_list:
        :param gamma_list:
        :return:
        """
        best_score = 0
        best_c = None
        best_gamma = None
        for c in c_list:
            for gamma in gamma_list:
                score = Tuning.k_fold_cross_validation_svm(x, y, train_x, k=k, C=c, kernel='rbf', gamma=gamma)
                print('C =', c, 'gamma = ', gamma, 'accuracy =', score)
            if score > best_score:
                best_score = score
                best_c = c
                best_gamma = gamma
        return best_score, best_c, best_gamma

    @staticmethod
    def svm_param_selection(x, y, n_folds, metric):
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

        clf = ms.GridSearchCV(
            svm.SVC(),
            param_grid,
            scoring=metric,
            cv=n_folds,
            refit=True,
            n_jobs=-1
        )
        clf.fit(x, y)

        print("Best parameters:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.4f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        return clf.best_estimator_

    @staticmethod
    def random_forest_param_selection(x, y, n_folds, metric):
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

        clf = ms.GridSearchCV(
            RandomForestClassifier(),
            param_grid,
            scoring=metric,
            cv=n_folds,
            refit=True,
            n_jobs=-1
        )
        clf.fit(x, y)

        print("Best parameters:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.4f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        return clf.best_estimator_

    @staticmethod
    def mlp_param_selection(x, y, n_folds, metric):
        """
        Multi-layer perceptron param selection
        :param x:
        :param y:
        :param n_folds:
        :param metric:
        :return:
        """
        param_grid = [{
            'hidden_layer_sizes': [(100, 50, 25), (100, 50), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'learning_rate_init': [0.1, 0.01, 10 ** -3, 10 ** -4],
            'learning_rate': ['constant', 'adaptive']
        }]

        clf = ms.GridSearchCV(
            MLPClassifier(max_iter=10000),
            param_grid=param_grid,
            scoring=metric,
            cv=n_folds,
            refit=True,
            n_jobs=-1
        )
        clf.fit(x, y)

        print("Best parameters:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.4f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        return clf.best_estimator_

    @staticmethod
    def knn_param_selection(x, y, n_folds, metric):
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

        clf = ms.GridSearchCV(
            KNeighborsClassifier(),
            param_grid=param_grid,
            scoring=metric,
            cv=n_folds,
            refit=True,
            n_jobs=-1
        )
        clf.fit(x, y)

        print("Best parameters:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.4f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        return clf.best_estimator_

    @staticmethod
    def sgd_param_selection(x, y, n_folds, metric):
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

        clf = ms.GridSearchCV(
            SGDClassifier(max_iter=6000),
            param_grid=param_grid,
            scoring=metric,
            cv=n_folds,
            refit=True,
            n_jobs=-1
        )
        clf.fit(x, y)

        print("Best parameters:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.4f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        return clf.best_estimator_
