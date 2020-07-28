class Common(object):
    """
    Common definitions
    """

    EXIT_SUCCESS = 0
    EXIT_FAILURE = 1

    @staticmethod
    def get_root_path() -> str:
        from pathlib import Path
        return Path(__file__).parent.parent.parent

    @staticmethod
    def trim(string: str):
        return string.replace(' ', '')

    @staticmethod
    def trim_all(string: str):
        return ''.join(string.split())
