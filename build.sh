#!/usr/bin/env bash

REF=$1

repo2docker \
  --no-run \
  --user-id 1001 \
  --user-name aicrowd \
  --image-name "${IMAGE_NAME}_${REF}" \
  --debug \
  --ref ${REF} \
  .
