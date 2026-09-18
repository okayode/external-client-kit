#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "WORKSPACE set to $DIR/.."

mkdir -p "$DIR/../transfer"

# Use the external client-kit application code.
unset PYTHONPATH

# Tell the client application where the permanent external kit is located.
export EXTERNAL_CLIENT_KIT_ROOT="$DIR/.."

# Use the CIFAR-10 data and split packaged in the external client kit.

echo "EXTERNAL_CLIENT_KIT_ROOT is $EXTERNAL_CLIENT_KIT_ROOT"
echo "PYTHONPATH is $PYTHONPATH"

start_fl() {
    if [ -f $DIR/../pid.fl ]; then
        echo "FL already running with pid $(cat $DIR/../pid.fl)"
        return 1
    fi

    echo "start fl because of no pid.fl"

    start_python &
    pid=$!
    echo $pid > $DIR/../pid.fl

    echo "new pid $pid"
}

start_python() {
    python3 -u -m nvflare.private.fed.app.client.client_train \
        -m $DIR/.. \
        -s fed_client.json \
        --set secure_train=true uid=external-site-1 org=external config_folder=config
}

start_fl
