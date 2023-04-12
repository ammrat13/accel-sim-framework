#!/bin/bash
set -e -u -o pipefail

# Check that all the required arguments were supplied
# If not, die
if [[ $# -ne 2 ]]; then
    echo "Usage: ./gen-gasan-trace-process-all.sh SRC_DIR DST_DIR"
    exit 1
fi

# Get the source and destination directories
# Check they exist
SRC_DIR=$1
DST_DIR=$2
if [[ ! -d "${SRC_DIR}" ]]; then
    echo "Source directory '${SRC_DIR}' does not exist"
    exit 1
fi
if [[ ! -d "${DST_DIR}" ]]; then
    echo "Destination directory '${DST_DIR}' does not exist"
    exit 1
fi
# Canonicalize the paths
SRC_DIR=$(realpath "${SRC_DIR}")
DST_DIR=$(realpath "${DST_DIR}")


# Get absolute paths
THIS_DIR=$(realpath $(dirname "${BASH_SOURCE}"))
PYS="${THIS_DIR}/gasan-trace-process.py"
OUT="${THIS_DIR}/gasan-trace-process-all.sh"

# Do it
# Find all the files that need to be processed in the source directory
# Write commands to process them and output them in the destination directory
find "${SRC_DIR}" -name "kernel-*.traceg" | awk "
    NR==1 {
        print \"set -e -u -x -o pipefail\"
    }
    {
        s1 = \$1;
        sub(\"${SRC_DIR}\", \"${DST_DIR}\");
        s2 = \$1;
        print \"python3 '${PYS}' '\" s1 \"' '\" s2 \"'\"
    }
" > "${OUT}"
chmod +x "${OUT}"
