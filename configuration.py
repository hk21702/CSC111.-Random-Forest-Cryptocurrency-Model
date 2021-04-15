"""Module for saving and reading constants related to cached settings
including sensitive settings such as API keys."""

from configparser import ConfigParser, DuplicateSectionError
from typing import Any

SETTINGS_FILE = 'cache/config.ini'


class Config(ConfigParser):
    """Class for handeling constants related to cached settings.

            Instance attributes:
            - config_name: Name of the config and its location.
    """
    config_name: str

    def __init__(self, config_name: str = SETTINGS_FILE) -> None:
        super().__init__()
        self.config_name = config_name
        self.read(self.config_name)

    def write_line(self, section: str, key: str, value: Any) -> None:
        """Writes a new configuration in the config parser.
        """
        try:
            self.add_section(section)
        except DuplicateSectionError:
            pass
        self.set(section, key, str(value))

    def update_cache(self) -> None:
        """Caches current configs into a local file.
        """
        with open(self.config_name, 'w') as configfile:
            self.write(configfile)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['typing', 'configparser'],
        'allowed-io': ['update_cache'],
        'max-line-length': 100,
        'disable': ['E1136']
    })
