#!/bin/bash

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"

if [[ "$(docker images -q codellm 2> /dev/null)" == "" ]]; then
  echo "Docker image does not exist. Building now..."
  docker build -t codellm $ROOT
else
  echo "Docker image exists. Skipping build."
fi

# run with shared volume
 docker run \
   -v $ROOT/models:/app/models \
   -v $ROOT/app.py:/app/app.py \
   -v $ROOT/repos:/app/repos \
   -p 8080:8080 \
   codellm
