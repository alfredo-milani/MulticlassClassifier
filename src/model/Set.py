from numpy import ndarray
from pandas import DataFrame


class Set(object):
    """
    Container for classifier's dataset
    """

    def __init__(self, set_=None, set_x=None, set_y=None):
        super().__init__()

        self.set_ = set_
        self.set_x = set_x
        self.set_y = set_y

    @property
    def set_(self):
        return self.__set_

    @set_.setter
    def set_(self, set_):
        self.__set_ = set_

    @property
    def set_x(self):
        return self.__set_x

    @set_x.setter
    def set_x(self, set_x):
        self.__set_x = set_x

    @property
    def set_y(self):
        return self.__set_y

    @set_y.setter
    def set_y(self, set_y):
        self.__set_y = set_y
