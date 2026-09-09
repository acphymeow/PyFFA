#!/usr/bin/env bash

# This file must be sourced so PYTHONPATH remains in the current shell:
#   source ./SetEnvCmd.sh
PYFFA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYFFA_SRC="${PYFFA_ROOT}/src"
PYFFA_EXAMPLES="${PYFFA_ROOT}/examples"

pyffa_prepend_path() {
    case ":${PYTHONPATH:-}:" in
        *":$1:"*) ;;
        *) PYTHONPATH="$1${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac
}

pyffa_prepend_path "${PYFFA_EXAMPLES}"
pyffa_prepend_path "${PYFFA_SRC}"
export PYTHONPATH
unset -f pyffa_prepend_path

echo "PyFFA paths enabled for this shell:"
echo "  ${PYFFA_SRC}"
echo "  ${PYFFA_EXAMPLES}"
echo "Run examples from the project root so ./Bmaps, ./Emaps, and output paths resolve correctly."
