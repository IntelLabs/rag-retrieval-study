# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

class InvalidArgument(Exception):
    """raise when user input arguments are invalid"""
    pass

class IncompleteSetup(Exception):
    """raise when user did not complete environmnet variable setup (see README and setup.sh)"""
    pass