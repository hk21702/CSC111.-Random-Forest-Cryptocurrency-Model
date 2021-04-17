""""Module which holds most of the major exceptions that may occur."""


class UnknownAVType(Exception):
    """Exception raised when trying to handle an unknown type of Alpha Vantage
    API call or invalid symbol."""


class RateLimited(Exception):
    """Exception raised when an API call fails because of a rate limit."""


class MissingData(Exception):
    """Exception raised when trying to make an input but there is insufficient data."""


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': [],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['E1136']
    })
