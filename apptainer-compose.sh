#!/bin/bash
# Enable strict error handling
set -e

# Get the absolute path of the current directory
BASE_DIR="$(pwd)"
IMG="experanto.sif"
DEF="apptainer.def"

# CHANGED: Create tmp directory OUTSIDE the project folder to avoid copy recursion
# This puts it in the parent directory (e.g., ../experanto_tmp)
TMP_DIR="$(dirname "$BASE_DIR")/experanto_tmp"

echo "[INFO] Setting up temporary directory at: $TMP_DIR"
mkdir -p "$TMP_DIR"

# Build the image if it doesn't exist
if [ ! -f "$BASE_DIR/$IMG" ]; then
    echo "[INFO] Building Apptainer image..."
    
    # Set the Apptainer temporary directory to the external path
    export APPTAINER_TMPDIR="$TMP_DIR"
    
    # Run the build
    apptainer build --fakeroot "$BASE_DIR/$IMG" "$BASE_DIR/$DEF"
    
    # Clean up tmp dir after successful build
    echo "[INFO] Cleaning up temporary directory..."
    rm -rf "$TMP_DIR"
fi

echo "[INFO] Starting Container..."

# Run the image with GPU access and bind the current directory
apptainer exec \
    --nv \
    --env "SSL_CERT_FILE=" \
    --bind "$BASE_DIR":/project \
    --bind /mnt/vast-react/projects/neural_foundation_model:/data \
    "$BASE_DIR/$IMG" \
    jupyter lab --allow-root --ip=0.0.0.0 --no-browser --port=8888 --NotebookApp.token='1234' --notebook-dir='/project'