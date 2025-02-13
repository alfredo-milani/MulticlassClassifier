import atexit
import logging
import sys

from pathlib import Path

from classifier import Trainer, Evaluator
from model import Conf
from util import Validation, Common, LogManager


class Client(object):
    """
    Launcher
    """

    __LOG: logging.Logger = None

    _DEFAULT_CONF_PATH: str = "./res/conf/conf.ini"

    def __init__(self):
        super().__init__()

        # init logging
        self.__init_logging()

    def __init_conf(self):
        # construct configuration file
        if len(sys.argv) > 1:
            conf_path = sys.argv[1]
        else:
            conf_path = f"{Path(Common.get_root_path(), Client._DEFAULT_CONF_PATH)}"

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

            self.log.info(f"v{self.conf.version}")
            self.log.debug(f"\n{self.conf}")

            # print current benchmark value/deadline
            self.log.info(
                f"[BENCHMARK] Best classifier found {self.conf.benchmark_best_found[1]}, with "
                f"F1-macro: {self.conf.benchmark_best_found[0]}."
            )
            self.log.info(
                f"[BENCHMARK] Threshold with F1-macro: {self.conf.benchmark_threshold[0]} "
                f"(deadline on {self.conf.benchmark_threshold[1]})."
            )

            if self.conf.dataset_test:
                # perform evaluation on test set using best classifier found
                Evaluator(self.conf).process()
            else:
                # execute multi-class classification to choose best classifier
                # Trainer(self.conf).process()

                # TESTING
                # TESTING - SAMPLING BEFORE FS
                import random
                rrange = [10000, 1000000]
                iterations = 100
                for r in random.sample(range(rrange[0], rrange[1]), iterations):
                    self.conf.rng_seed = r
                    self.log.critical(f"SEED: {self.conf.rng_seed}")
                    Trainer(self.conf).process()
        except KeyboardInterrupt:
            self.log.info(f"Execution interrupted by user.")
        except (Conf.NoValueError, SyntaxError, ValueError) as e:
            self.log.critical(f"Error in configuration file: {e}")
            sys.exit(Common.EXIT_FAILURE)
        except Exception as e:
            self.log.critical(f"Unexpected exception.\n{e}")
            sys.exit(Common.EXIT_FAILURE)
            
    @property
    def log(self) -> logging.Logger:
        return self.__LOG

    @property
    def conf(self) -> Conf:
        return self.__conf


if __name__ == "__main__":
    Client().start()
