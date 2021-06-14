# Environment variables preparation
if [ -z "$variables_requestedPortName" ]; then
    echo "[Env_var] No requested ports for this task. Nothing to be set."
else
    REQUESTED_PORT_NAME="PUBLIC_$variables_requestedPortName"
    if [[ ! -z $variables_requestedPortName ]]; then
        REQ="variables_$variables_requestedPortName"
        REQUESTED_PORT_VALUE=${!REQ}

        if [[ -z ${!REQUESTED_PORT_NAME} ]]; then
            echo "[Env_var] Variable $REQUESTED_PORT_NAME does not exist. Exporting ..."
            export PUBLIC_$REQUESTED_PORT_NAME=$REQUESTED_PORT_VALUE
        fi

        echo "[Env_var] $REQUESTED_PORT_NAME variable set to $REQUESTED_PORT_VALUE"
    fi
fi
