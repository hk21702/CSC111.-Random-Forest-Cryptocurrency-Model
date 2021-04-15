""" Module for all initialization proceses related
to data ingest and initial pre-processing.
"""
import os


def create_project_dirs() -> None:
    """Create default project folders"""
    create_folder('cache/data')
    create_folder('cache/models')


def create_folder(directory: str) -> None:
    """Create folders"""
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['os'],
        'allowed-io': ['create_folder'],
        'max-line-length': 100,
        'disable': ['E1136']
    })
