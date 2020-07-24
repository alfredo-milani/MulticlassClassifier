from pandas import DataFrame


class Set(object):
    """
    Container for classifier's dataset
    """

    def __init__(self, set_: DataFrame = None, set_x: DataFrame = None, set_y: DataFrame = None):
        super().__init__()

        self.set_ = set_
        self.set_x = set_x
        self.set_y = set_y

    @property
    def set_(self) -> DataFrame:
        return self.__set_

    @set_.setter
    def set_(self, set_: DataFrame):
        self.__set_ = set_

    @property
    def set_x(self) -> DataFrame:
        return self.__set_x

    @set_x.setter
    def set_x(self, set_x: DataFrame):
        self.__set_x = set_x

    @property
    def set_y(self) -> DataFrame:
        return self.__set_y

    @set_y.setter
    def set_y(self, set_y: DataFrame):
        self.__set_y = set_y
