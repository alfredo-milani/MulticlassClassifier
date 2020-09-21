from numpy import ndarray
from pandas import DataFrame


class Set(object):
    """
    Container for classifier's dataset
    """

    def __init__(self, w_set=None, X=None, y=None):
        super().__init__()

        self.w_set = w_set
        self.X = X
        self.y = y

    @property
    def w_set(self):
        """
        Return whole dataset
        :return: dataset
        """
        return self.__w_set

    @w_set.setter
    def w_set(self, w_set):
        self.__w_set = w_set

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
