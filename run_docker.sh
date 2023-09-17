#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ "$(docker images -q codellm 2> /dev/null)" == "" ]]; then
  echo "Docker image does not exist. Building now..."
  docker build -t codellm $SCRIPT_DIR
else
  echo "Docker image exists. Skipping build."
fi

# run without shared volume (you will lose all new embeddings)
# docker run -p 8080:8080 codellm

# run with shared volume
 docker run \
   -v $SCRIPT_DIR/llm.py:/app/llm.py \
   -v $SCRIPT_DIR/app.py:/app/app.py \
   -v $SCRIPT_DIR/data:/app/data \
   -p 8080:8080 \
   codellm