#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ "$(docker images -q codellm 2> /dev/null)" == "" ]]; then
  echo "Docker image does not exist. Building now..."
  docker build -t my-streamlit-app $SCRIPT_DIR
else
  echo "Docker image exists. Skipping build."
fi

#docker run -v $SCRIPT_DIR:/app -p 8080:8080 codellm
docker run \
  -v $SCRIPT_DIR/llm.py:/app/llm.py \
  -v $SCRIPT_DIR/app.py:/app/app.py \
  -v $SCRIPT_DIR/data:/app/data \
  -p 8080:8080 \
  codellm