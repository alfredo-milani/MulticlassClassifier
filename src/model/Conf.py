import ast
import configparser
import threading
import datetime

from datetime import date

import __version__
from util import Validation


class Conf(dict):
    """
    Contains configuration data
    """

    __INSTANCE = None
    __LOCK = threading.Lock()

    DATE_FORMAT = '%d/%m/%Y'

    # Intern
    K_VERSION = "version"
    V_DEFAULT_VERSION = __version__.__version__
    K_APP_NAME = "app_name"
    V_DEFAULT_APP_NAME = "MulticlassClassifier"

    # Section GENERAL
    S_GENERAL = "GENERAL"
    # Keys
    K_TMP = "tmp"
    V_DEFAULT_TMP = "/tmp"
    K_DEBUG = "debug"
    V_DEFAULT_DEBUG = False

    # Section TRAINING
    S_TRAINING = "TRAINING"
    # Keys
    K_DATASET = "dataset"
    V_DEFAULT_DATASET = None
    K_DATASET_TEST_RATIO = "dataset.test_ratio"
    V_DEFAULT_DATASET_TEST_RATIO = 0.2
    K_RNG_SEED = "rng.seed"
    V_DEFAULT_RNG_SEED = 0
    K_PAIR_PLOT_COMPUTE = "pair_plot.compute"
    V_DEFAULT_PAIR_PLOT_COMPUTE = False
    K_PAIR_PLOT_SAVE = "pair_plot.save"
    V_DEFAULT_PAIR_PLOT_SAVE = False
    K_THREADS = "threads"
    V_DEFAULT_THREADS = 1
    K_CLASSIFIER_DUMP = "classifier.dump"
    V_DEFAULT_CLASSIFIER_DUMP = False

    # Section MOBD
    S_MOBD = "MOBD"
    # Keys
    K_BENCHMARK_BEST_FOUND = "benchmark.best_found"
    V_DEFAULT_BENCHMARK_BEST_FOUND = (0.0, None)
    K_BENCHMARK_THRESHOLD = "benchmark.threshold"
    V_DEFAULT_BENCHMARK_THRESHOLD = (0.0, str(date.today()))
    K_DATASET_TEST = "dataset.test"
    V_DEFAULT_DATASET_TEST = ""

    def __init__(self):
        super().__init__()
        if Conf.__INSTANCE is not None:
            raise Conf.SingletonError(Conf.__qualname__)

        Conf.__INSTANCE = self
        self.__config_parser = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())

        # default values
        self.version = Conf.V_DEFAULT_VERSION
        self.app_name = Conf.V_DEFAULT_APP_NAME

        # section GENERAL
        self.tmp = Conf.V_DEFAULT_TMP
        self.debug = Conf.V_DEFAULT_DEBUG

        # section TRAINING
        self.dataset = Conf.V_DEFAULT_DATASET
        self.dataset_test_ratio = Conf.V_DEFAULT_DATASET_TEST_RATIO
        self.rng_seed = Conf.V_DEFAULT_RNG_SEED
        self.pair_plot_compute = Conf.V_DEFAULT_PAIR_PLOT_COMPUTE
        self.pair_plot_save = Conf.V_DEFAULT_PAIR_PLOT_SAVE
        self.threads = Conf.V_DEFAULT_THREADS
        self.classifier_dump = Conf.V_DEFAULT_CLASSIFIER_DUMP

        # section MOBD
        self.benchmark_best_found = Conf.V_DEFAULT_BENCHMARK_BEST_FOUND
        self.benchmark_threshold = Conf.V_DEFAULT_BENCHMARK_THRESHOLD
        self.dataset_test = Conf.V_DEFAULT_DATASET_TEST

    @classmethod
    def get_instance(cls) -> "Conf":
        if cls.__INSTANCE is None:
            with cls.__LOCK:
                if cls.__INSTANCE is None:
                    Conf()
        return cls.__INSTANCE

    def load_from(self, config_file: str) -> None:
        """

        :param config_file:
        :raise: SyntaxError if there is a syntax error in configuration file
        """
        Validation.is_file_readable(
            config_file,
            f"File '{config_file}' *must* exists and be readable"
        )
        self.__config_parser.read(config_file)

        # section GENERAL
        self.__put_str(Conf.K_TMP, Conf.S_GENERAL, Conf.K_TMP, Conf.V_DEFAULT_TMP)
        self.__put_bool(Conf.K_DEBUG, Conf.S_GENERAL, Conf.K_DEBUG, Conf.V_DEFAULT_DEBUG)

        # section TRAINING
        self.__put_str(Conf.K_DATASET, Conf.S_TRAINING, Conf.K_DATASET, Conf.V_DEFAULT_DATASET)
        self.__put_float(Conf.K_DATASET_TEST_RATIO, Conf.S_TRAINING, Conf.K_DATASET_TEST_RATIO, Conf.V_DEFAULT_DATASET_TEST_RATIO)
        self.__put_int(Conf.K_RNG_SEED, Conf.S_TRAINING, Conf.K_RNG_SEED, Conf.V_DEFAULT_RNG_SEED)
        self.__put_bool(Conf.K_PAIR_PLOT_COMPUTE, Conf.S_TRAINING, Conf.K_PAIR_PLOT_COMPUTE, Conf.V_DEFAULT_PAIR_PLOT_COMPUTE)
        self.__put_bool(Conf.K_PAIR_PLOT_SAVE, Conf.S_TRAINING, Conf.K_PAIR_PLOT_SAVE, Conf.V_DEFAULT_PAIR_PLOT_SAVE)
        self.__put_int(Conf.K_THREADS, Conf.S_TRAINING, Conf.K_THREADS, Conf.V_DEFAULT_THREADS)
        self.__put_bool(Conf.K_CLASSIFIER_DUMP, Conf.S_TRAINING, Conf.K_CLASSIFIER_DUMP, Conf.V_DEFAULT_CLASSIFIER_DUMP)

        # section MOBD
        self.__put_tuple(Conf.K_BENCHMARK_BEST_FOUND, Conf.S_MOBD, Conf.K_BENCHMARK_BEST_FOUND, Conf.V_DEFAULT_BENCHMARK_BEST_FOUND)
        self.__put_tuple(Conf.K_BENCHMARK_THRESHOLD, Conf.S_MOBD, Conf.K_BENCHMARK_THRESHOLD, Conf.V_DEFAULT_BENCHMARK_THRESHOLD)
        self.__put_str(Conf.K_DATASET_TEST, Conf.S_MOBD, Conf.K_DATASET_TEST, Conf.V_DEFAULT_DATASET_TEST)

    def __put_obj(self, key: str, section: str, section_key: str, default: object) -> None:
        try:
            self[key] = self.__config_parser.get(section, section_key)
        except (configparser.NoOptionError, configparser.NoSectionError):
            if default is None:
                raise Conf.NoValueError(key)
            self[key] = default

    def __put_str(self, key: str, section: str, section_key: str, default: str) -> None:
        """
        Create a new entry for the current instance (dict) with string object value
        :param key:
        :param section:
        :param section_key:
        :param default:
        :raise: Conf.NoValueError if None is specified as default value for current key
        """
        try:
            self[key] = str(self.__config_parser.get(section, section_key))
        except (configparser.NoOptionError, configparser.NoSectionError):
            if default is None:
                raise Conf.NoValueError(key)
            self[key] = str(default)

    def __put_int(self, key: str, section: str, section_key: str, default: int) -> None:
        try:
            self[key] = int(self.__config_parser.get(section, section_key))
        except (configparser.NoOptionError, configparser.NoSectionError):
            if default is None:
                raise Conf.NoValueError(key)
            self[key] = int(default)

    def __put_float(self, key: str, section: str, section_key: str, default: float) -> None:
        try:
            self[key] = float(self.__config_parser.get(section, section_key))
        except (configparser.NoOptionError, configparser.NoSectionError):
            if default is None:
                raise Conf.NoValueError(key)
            self[key] = float(default)

    def __put_tuple(self, key: str, section: str, section_key: str, default: tuple) -> None:
        """

        :param key:
        :param section:
        :param section_key:
        :param default:
        :raise: ValueError if ast.literal_eval(node_or_string) fails parsing
            input string (syntax error in key/value pairs type)
        """
        try:
            self[key] = ast.literal_eval(self.__config_parser.get(section, section_key))
        except (configparser.NoOptionError, configparser.NoSectionError):
            if default is None:
                raise Conf.NoValueError(key)
            self[key] = tuple(default)

    def __put_dict(self, key: str, section: str, section_key: str, default: dict) -> None:
        """

        :param key:
        :param section:
        :param section_key:
        :param default:
        :raise: SyntaxError if ast.literal_eval(node_or_string) fails parsing
            input string (syntax error in key/value pairs)
        """
        try:
            self[key] = ast.literal_eval(self.__config_parser.get(section, section_key))
        except (configparser.NoOptionError, configparser.NoSectionError):
            if default is None:
                raise Conf.NoValueError(key)
            self[key] = dict(default)

    def __put_bool(self, key: str, section: str, section_key: str, default: bool) -> None:
        try:
            self[key] = bool(self.__config_parser.get(section, section_key))
        except (configparser.NoOptionError, configparser.NoSectionError):
            if default is None:
                raise Conf.NoValueError(key)
            self[key] = bool(default)

    def __put_date(self, key: str, section: str, section_key: str, default: date) -> None:
        try:
            self[key] = datetime.datetime.strptime(
                self.__config_parser.get(section, section_key), Conf.DATE_FORMAT
            ).date()
        except (configparser.NoOptionError, configparser.NoSectionError):
            if default is None:
                raise Conf.NoValueError(key)
            self[key] = default

    @property
    def version(self):
        return self[Conf.K_VERSION]

    @version.setter
    def version(self, version: str):
        self[Conf.K_VERSION] = version

    @property
    def app_name(self) -> str:
        return self[Conf.K_APP_NAME]

    @app_name.setter
    def app_name(self, app_name: str):
        self[Conf.K_APP_NAME] = app_name

    @property
    def tmp(self) -> str:
        return self[Conf.K_TMP]

    @tmp.setter
    def tmp(self, tmp: str):
        self[Conf.K_TMP] = tmp

    @property
    def debug(self) -> bool:
        return self[Conf.K_DEBUG]

    @debug.setter
    def debug(self, debug: bool):
        self[Conf.K_DEBUG] = debug

    @property
    def dataset(self) -> str:
        return self[Conf.K_DATASET]

    @dataset.setter
    def dataset(self, dataset: str):
        self[Conf.K_DATASET] = dataset

    @property
    def dataset_test_ratio(self) -> float:
        return self[Conf.K_DATASET_TEST_RATIO]

    @dataset_test_ratio.setter
    def dataset_test_ratio(self, dataset_test_ratio: float):
        self[Conf.K_DATASET_TEST_RATIO] = dataset_test_ratio

    @property
    def rng_seed(self) -> int:
        return self[Conf.K_RNG_SEED]

    @rng_seed.setter
    def rng_seed(self, rng_seed: int):
        self[Conf.K_RNG_SEED] = rng_seed

    @property
    def pair_plot_compute(self) -> bool:
        return self[Conf.K_PAIR_PLOT_COMPUTE]

    @pair_plot_compute.setter
    def pair_plot_compute(self, pair_plot_compute: bool):
        self[Conf.K_PAIR_PLOT_COMPUTE] = pair_plot_compute

    @property
    def pair_plot_save(self) -> bool:
        return self[Conf.K_PAIR_PLOT_SAVE]

    @pair_plot_save.setter
    def pair_plot_save(self, pair_plot_save: bool):
        self[Conf.K_PAIR_PLOT_SAVE] = pair_plot_save

    @property
    def threads(self) -> int:
        return self[Conf.K_THREADS]

    @threads.setter
    def threads(self, threads: int):
        self[Conf.K_THREADS] = threads

    @property
    def classifier_dump(self) -> bool:
        return self[Conf.K_CLASSIFIER_DUMP]

    @classifier_dump.setter
    def classifier_dump(self, classifier_dump: bool):
        self[Conf.K_CLASSIFIER_DUMP] = classifier_dump

    @property
    def benchmark_best_found(self) -> tuple:
        return self[Conf.K_BENCHMARK_BEST_FOUND]

    @benchmark_best_found.setter
    def benchmark_best_found(self, benchmark_best_found: tuple):
        self[Conf.K_BENCHMARK_BEST_FOUND] = benchmark_best_found

    @property
    def benchmark_threshold(self) -> tuple:
        return self[Conf.K_BENCHMARK_THRESHOLD]

    @benchmark_threshold.setter
    def benchmark_threshold(self, benchmark_threshold: tuple):
        self[Conf.K_BENCHMARK_THRESHOLD] = benchmark_threshold

    @property
    def dataset_test(self) -> str:
        return self[Conf.K_DATASET_TEST]

    @dataset_test.setter
    def dataset_test(self, dataset_test: str):
        self[Conf.K_DATASET_TEST] = dataset_test

    def __str__(self):
        return f"### Configuration - from {self.__class__.__name__}: " \
               f"{str().join(f'{chr(10)}{chr(9)}{{ {k} : {v} }}' for k, v in self.items())}"

    def __add__(self, other):
        return str(self) + other

    def __radd__(self, other):
        return other + str(self)

    # Exception classes
    class Error(Exception):
        """
        Base class for Conf exceptions.
        """

        def __init__(self, msg=''):
            self.message = msg
            Exception.__init__(self, msg)

        def __repr__(self):
            return self.message

        __str__ = __repr__

    class SingletonError(Error):
        """
        Raised when a second instance of a Singleton is created
        """
        def __init__(self, instance):
            Conf.SingletonError.__init__(self, f"Singleton instance: a second instance of {instance} can not be created")
            self.instance = instance
            self.args = (instance,)

    class NoValueError(Error):
        """
        Raised when no value is provided for a specified key
        """

        def __init__(self, key):
            Conf.Error.__init__(self, f"Missing value for key '{key}'")
            self.key = key
            self.args = (key,)
