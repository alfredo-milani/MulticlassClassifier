import logging
import logging.config
import sys
import threading
from enum import Enum

from util import Validation


class LogManager(object):
    """
    Log manager class

    LEVELS: NOTSET < DEBUG < INFO < WARNING < ERROR < CRITICAL
    """

    __INSTANCE: "LogManager" = None
    __LOCK: threading.Lock = threading.Lock()

    __FORMATTER: logging.Formatter = None

    __HANDLER_CONSOLE: logging.Handler = None

    __LOGGER_ROOT: logging.Logger = None
    __LOGGER_MCSVM: logging.Logger = None

    def __init__(self):
        super().__init__()
        if LogManager.__INSTANCE is not None:
            raise LogManager.SingletonError(LogManager.__qualname__)

        LogManager.__INSTANCE = self

    @classmethod
    def get_instance(cls) -> "LogManager":
        if cls.__INSTANCE is None:
            with cls.__LOCK:
                if cls.__INSTANCE is None:
                    LogManager()
        return cls.__INSTANCE

    @classmethod
    def load(cls) -> None:
        with cls.__LOCK:
            # formatters
            cls.__FORMATTER = LogManager.__configure_formatter()

            # handlers
            cls.__HANDLER_CONSOLE = LogManager.__configure_handler_console()

            # loggers
            cls.__LOGGER_ROOT = LogManager.__configure_logger_root()
            cls.__LOGGER_MCSVM = LogManager.__configure_logger_observer()

    class Logger(Enum):
        ROOT = "root"
        MCSVM = "mcsvm"

    @classmethod
    def logger(cls, logger: Logger) -> logging.Logger:
        Validation.not_none(logger)

        if logger == cls.Logger.ROOT:
            return cls.__LOGGER_ROOT
        elif logger == cls.Logger.MCSVM:
            return cls.__LOGGER_MCSVM
        else:
            raise NotImplementedError

    @classmethod
    def __configure_formatter(cls) -> logging.Formatter:
        f = "[%(levelname)-7s] | %(asctime)s | [%(name)s] %(funcName)s (%(module)s:%(lineno)s) - %(message)s"
        return logging.Formatter(f)

    @classmethod
    def __configure_logger_root(cls) -> logging.Logger:
        root_logger = logging.getLogger(cls.Logger.ROOT.value)
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(cls.__HANDLER_CONSOLE)
        return root_logger

    @classmethod
    def __configure_logger_observer(cls) -> logging.Logger:
        observer_logger = logging.getLogger(cls.Logger.MCSVM.value)
        observer_logger.setLevel(logging.DEBUG)
        observer_logger.propagate = 0
        observer_logger.addHandler(cls.__HANDLER_CONSOLE)
        return observer_logger

    @classmethod
    def __configure_handler_console(cls) -> logging.Handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(cls.__FORMATTER)
        return console_handler

    @classmethod
    def shutdown(cls) -> None:
        logging.shutdown()

    @classmethod
    def enable_debug_level(cls):
        cls.__HANDLER_CONSOLE.setLevel(logging.DEBUG)

    # Exception classes
    class Error(Exception):
        """
        Base class for LogManager exceptions.
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
            LogManager.SingletonError.__init__(self, f"Singleton instance: a second instance of {instance} can not be created")
            self.instance = instance
            self.args = (instance,)
