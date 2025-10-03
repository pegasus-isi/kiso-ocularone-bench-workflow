#!/bin/bash

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $(basename $0) <VIDEO-DIR>"
    exit 1
fi

VIDEO_DIR=$1
CONFIG=$2
RECIPE=$3

rm -f workflow.yml graph.png
SUBMIT_DIR=$(./workflow.py --video-dir ${VIDEO_DIR} 2>&1 | grep 'pegasus-run /' | cut -d' ' -f2)
echo "Submit Dir: ${SUBMIT_DIR}"

# Graph
pegasus-graphviz --label=xform \
                 --files \
                 --output graph.png \
                 workflow.yml &> /dev/null

# Replace docker_init with container_init and remove file transfer
# CONTAINER='ocularone-bench-workflow'
# find ${SUBMIT_DIR} -name '*.sh'  -exec sed -E -i '' -e "s@^docker_init (.*)@container_init ; cont_image='pegasus/\1:latest'@g" {} \;

# Run the workflow
pegasus-run ${SUBMIT_DIR}
