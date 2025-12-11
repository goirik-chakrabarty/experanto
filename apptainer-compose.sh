#!/bin/bash
## Enable strict error handling
# set -e

# Get the absolute path of the current directory
BASE_DIR="$(pwd)"
IMG="experanto.sif"
DEF="apptainer.def"

# Define path to nexport (sibling directory)
NEXPORT_DIR="$(dirname "$BASE_DIR")/nexport"

# Build the image if it doesn't exist
if [ ! -f "$BASE_DIR/$IMG" ]; then
    echo "[INFO] Building Apptainer image..."
    # Set the Apptainer temporary directory to the external path
    export APPTAINER_TMPDIR=$LOCAL_TMPDIR
    # Run the build
    apptainer build --fakeroot "$BASE_DIR/$IMG" "$BASE_DIR/$DEF"
fi

echo "[INFO] Starting Container..."

# Run the image with GPU access and bind the current directory
# Added binding for nexport and updated PYTHONPATH
apptainer exec \
    --nv \
    --env "SSL_CERT_FILE=" \
    --env "PYTHONPATH=/nexport:\$PYTHONPATH" \
    --bind "$BASE_DIR":/project \
    --bind "$NEXPORT_DIR":/nexport \
    --bind /mnt/vast-react/projects/neural_foundation_model:/data \
    "$BASE_DIR/$IMG" \
    jupyter lab --allow-root --ip=0.0.0.0 --no-browser --port=8888 --NotebookApp.token='1234' --notebook-dir='/project'
