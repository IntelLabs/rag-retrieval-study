class InvalidArgument(Exception):
    """raise when user input arguments are invalid"""
    pass

class IncompleteSetup(Exception):
    """raise when user did not complete environmnet variable setup (see README and setup.sh)"""
    pass