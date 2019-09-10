#!/bin/bash

# load path is path to agent.ckpt-ITERATION
LOAD_PATH=$1
# split load path on '/'
IFS='/' read -ra ARR <<< "$LOAD_PATH"
# strip agent.ckpt-ITER part from string
filelen=${#ARR[(${#ARR[@]}-1)]}  # 0-indexing; take length of last element of ARR
LOAD_DIR=${LOAD_PATH:0:(${#LOAD_PATH} - $filelen - 1)}
# read agl arg from load path config_run.json
ALG=$(cat $LOAD_DIR/config_run.json | python -c "import sys, json; print(json.load(sys.stdin)['alg'])")

HOST="localhost"
OS=$(uname -s)
if [ "$OS" = "Darwin" ]; then
	HOST="docker.for.mac.host.internal"
fi

docker run -it \
  --net=host \
  -e CROWDAI_REDIS_HOST=$HOST \
  -e CROWDAI_IS_GRADING=True \
  -e CROWDAI_DEBUG_MODE=True \
  -e ALG=$ALG \
  -e LOAD_PATH=$LOAD_PATH \
  -v `pwd`:/www \
  $IMAGE_NAME \
  /www/run.sh
