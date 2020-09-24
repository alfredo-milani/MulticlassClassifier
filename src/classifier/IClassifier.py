from abc import abstractmethod


class IClassifier(object):
    """
    Interface for classificator
    """

    @abstractmethod
    def clean_data(self) -> None:
        """
        Manage bad values
        """
        raise NotImplementedError

    @abstractmethod
    def feature_selection(self) -> None:
        """
        Features selection
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self) -> None:
        """
        Data sampling
        """
        raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        """
        Training phase
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> None:
        """
        Evaluation phase
        """
        raise NotImplementedError
