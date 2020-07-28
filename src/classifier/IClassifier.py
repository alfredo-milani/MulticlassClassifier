from abc import abstractmethod


class IClassifier(object):
    """

    """

    @abstractmethod
    def split(self) -> None:
        """

        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def manage_bad_values(self) -> None:
        """

        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def normalize(self) -> None:
        """
        Normalize, Scale features
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def feature_selection(self) -> None:
        """

        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self) -> None:
        """

        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def tune(self) -> None:
        """

        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        """
        Re-train Classifier after hyperparameter tuning
        TODO: vedere se inglobare il re-train (training dopo aver trovato iperparametri)
              in tune()
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> None:
        """

        :return:
        """
        raise NotImplementedError
