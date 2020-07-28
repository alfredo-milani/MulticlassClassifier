from abc import ABC

from classifier import IClassifier


class AbstractClassifier(ABC, IClassifier):
    """

    """

    def __init__(self):
        super().__init__()

    def process(self) -> None:
        """
        DO NOT EDIT OR OVERRIDE THIS TEMPLATE METHOD
        :return:
        """

        self.prepare()

        try:
            self.split()
            self.manage_bad_values()
            self.normalize()
            self.feature_selection()
            self.sample()
            self.tune()
            self.train()
            self.evaluate()
        except Exception as e:
            return self.on_error(e)

        return self.on_success()

    def prepare(self) -> None:
        pass

    def on_success(self) -> None:
        pass

    def on_error(self, exception: Exception = None) -> None:
        pass
