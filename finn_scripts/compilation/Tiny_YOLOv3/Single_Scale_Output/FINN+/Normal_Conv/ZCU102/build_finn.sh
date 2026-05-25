#!/bin/bash

# Define input and output file paths
MODEL_PATH="./best.qonnx.onnx"
CLEANED_MODEL_PATH="./best.qonnx_cleanup.onnx"
BUILD_FILE="build.yaml"

# Step 1: Run the Python cleanup script
echo "Running cleanup on $MODEL_PATH..."
python3 model_cleanup.py --model-path "$MODEL_PATH" --exported-filename "$CLEANED_MODEL_PATH"

# Check if cleanup was successful
if [ $? -ne 0 ]; then
    echo "Error during cleanup. Exiting..."
    exit 1
fi

# Step 2: Run finn build
echo "Running finn build on $CLEANED_MODEL_PATH..."
finn build --skip-dep-update "$BUILD_FILE" "$CLEANED_MODEL_PATH"

# Check if finn build was successful
if [ $? -ne 0 ]; then
    echo "Error during finn build. Exiting..."
    exit 1
fi

echo "Process completed successfully!"
