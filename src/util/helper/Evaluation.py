import sklearn.metrics as metrics

from pandas import DataFrame


class Evaluation(object):
    """

    """

    @staticmethod
    def evaluate(classifier, x: DataFrame, y: DataFrame):
        """

        :param classifier:
        :param x:
        :param y:
        :return: accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        """
        return metrics.accuracy_score(y, classifier.predict(x)), \
               metrics.precision_score(y, classifier.predict(x), average='macro'), \
               metrics.recall_score(y, classifier.predict(x), average='macro'), \
               metrics.f1_score(y, classifier.predict(x), average='macro'), \
               metrics.confusion_matrix(y, classifier.predict(x))
