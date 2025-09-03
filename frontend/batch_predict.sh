#!/bin/bash

IMAGE_DIR="./test"

MALE=1 #batch predict need to use same gender, but it could be improved later

for IMAGE in "$IMAGE_DIR"/*.{png,jpg,jpeg}; do
  if [[ -f "$IMAGE" ]]; then
    echo "Processing $IMAGE ..."
    python cli.py --image "$IMAGE" --male $MALE
    echo ""
  fi
done

echo "All done."