#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

docker build -t codellm:linux_amd64 --platform linux/amd64 $SCRIPT_DIR