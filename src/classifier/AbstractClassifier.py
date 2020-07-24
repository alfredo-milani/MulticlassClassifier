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

        try:
            self.prepare()
            self.refactor()
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
