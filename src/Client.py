import atexit
import logging
import sys

from pathlib import Path

from classifier import MulticlassClassifier
from model import Conf
from util import Validation, Common, LogManager


class Client(object):
    """
    Launcher
    """

    __DEFAULT_CONF_PATH: str = "./res/conf/conf.ini"

    __LOG: logging.Logger = None

    def __init__(self):
        super().__init__()

        # init logging
        self.__init_logging()

    def __init_conf(self):
        # construct configuration file
        if len(sys.argv) > 1:
            conf_path = sys.argv[1]
        else:
            conf_path = f"{Path(Common.get_root_path(), Client.__DEFAULT_CONF_PATH)}"

        Validation.is_file_readable(
            conf_path,
            f"Error on '{conf_path}': configuration file *must* exists and be readable"
        )

        self.__conf = Conf.get_instance()
        self.conf.load_from(conf_path)

        if self.conf.debug:
            self.__LOG_MANAGER.enable_debug_level()

    def __init_logging(self):
        self.__LOG_MANAGER = LogManager.get_instance()
        # create formatter, handler, logger for logging purpose
        self.__LOG_MANAGER.load()
        # get root logger
        self.__LOG = self.__LOG_MANAGER.logger(LogManager.Logger.ROOT)

    def stop(self) -> None:
        self.__LOG_MANAGER.shutdown()

    def start(self) -> None:
        try:
            # init configuration file
            self.__init_conf()

            # register cleaning routine at exit
            atexit.register(self.stop)

            self.__LOG.debug(f"v{self.conf.version}")
            self.__LOG.debug(f"\n{self.conf}")

            # print current benchmark value/deadline
            self.__LOG.info(f"[BENCHMARK] Current value: {self.conf.benchmark_value}")
            self.__LOG.info(f"[BENCHMARK] Current deadline: {self.conf.benchmark_deadline}")

            # execute Multi-class classification using SVM model
            MulticlassClassifier(self.conf).process()
            # TODO - client per calcolare il valore migliore sul test set
        except KeyboardInterrupt:
            self.__LOG.info(f"Execution interrupted by user.")
        except (Conf.NoValueError, SyntaxError) as e:
            self.__LOG.critical(f"Error in configuration file: {e}")
            sys.exit(Common.EXIT_FAILURE)
        except Exception as e:
            self.__LOG.critical(f"Unexpected exception.\n{e}")
            sys.exit(Common.EXIT_FAILURE)

    @property
    def conf(self) -> Conf:
        return self.__conf


if __name__ == "__main__":
    Client().start()
