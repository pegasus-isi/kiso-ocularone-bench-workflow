#!/bin/sh

set -e

DIR=$(realpath $(dirname $0))

export PYTHONPATH=`pegasus-config --python`:`pegasus-config --python-externals`

#### Executing Workflow Generator ####
${DIR}/workflow.py -e condorpool -o $(pwd)/workflow.yml

#### Generating Pegasus Properties ####
echo "pegasus.transfer.arguments = -m 1" >> pegasus.properties

pegasus-plan --conf pegasus.properties \
    --dir submit \
    --sites condorpool \
    --output-sites local \
    --cleanup leaf \
    --force \
    "$@" workflow.yml
