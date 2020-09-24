from numpy import ndarray
from pandas import DataFrame


class Set(object):
    """
    Container for classifier's dataset
    """

    def __init__(self, set_=None, X=None, y=None):
        super().__init__()

        self.set_ = set_
        self.X = X
        self.y = y

    @property
    def set_(self):
        """
        Return whole dataset
        :return: dataset
        """
        return self.__set_

    @set_.setter
    def set_(self, set_):
        self.__set_ = set_

    @property
    def X(self):
        return self.__X

    @X.setter
    def X(self, X):
        self.__X = X

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, y):
        self.__y = y
