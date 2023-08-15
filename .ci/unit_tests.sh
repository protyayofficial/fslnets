#!/usr/bin/env bash

set -e

pip install -e ".[test]"

# pytest can have trouble with
# native namespace packages
export PYTHONPATH=$(pwd)

pytest
