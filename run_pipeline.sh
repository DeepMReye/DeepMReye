#!/bin/bash
# DeepMReye 2.0 Pipeline Orchestrator
#
# Thin convenience wrapper around the Python CLI. The real entry point is:
#     python -m deepmreye <command> [options]
# This script just activates .venv and forwards to it, so behaviour can never
# drift between the two. Prefer the Python CLI directly if you are not using .venv.

set -e

# Default settings
DATA_DIR="data"
LIMIT_COMPILE="5"

function print_header {
    echo ""
    echo "========================================================"
    echo " $1"
    echo "========================================================"
    echo ""
}

function show_help {
    echo "Usage: ./run_pipeline.sh [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  all             Run the entire pipeline (pauses for manual QA)"
    echo "  compile         Step 1: Fetch and compile samples from OpenNeuro"
    echo "  qa              Step 2: Launch Streamlit GUI for manual data QA"
    echo "  preprocess      Step 3: Download and preprocess full approved datasets"
    echo "  train           Step 4: Train the core JEPA model"
    echo ""
    echo "Options:"
    echo "  --data-dir      Set the data directory (default: data)"
    echo "  --limit         Number of datasets to sample in Step 1 (default: 5)"
}

COMMAND=$1
shift || true

# Parse optional arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data-dir) DATA_DIR="$2"; shift ;;
        --limit) LIMIT_COMPILE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; show_help; exit 1 ;;
    esac
    shift
done

if [[ -z "$COMMAND" ]]; then
    show_help
    exit 1
fi

# Ensure Python environment is active or exists
if [[ ! -d ".venv" ]]; then
    echo "Python environment (.venv) not found. Please install dependencies."
    exit 1
fi
PYTHON_CMD=".venv/bin/python3"

# Define the steps (all delegate to `python -m deepmreye`)
function step_compile {
    print_header "Step 1: Compiling OpenNeuro Samples"
    $PYTHON_CMD -m deepmreye compile --data-dir "$DATA_DIR" --limit "$LIMIT_COMPILE"
}

function step_qa {
    print_header "Step 2: Manual QA (browser labeling UI)"
    echo "Starting labeling UI for dataset approval..."
    echo "Open the printed URL in your browser to label datasets."
    echo "Press Ctrl+C to stop the UI once you are done."
    $PYTHON_CMD -m deepmreye qa --data-dir "$DATA_DIR"
}

function step_preprocess {
    print_header "Step 3: Full Download and Preprocessing"
    $PYTHON_CMD -m deepmreye preprocess --data-dir "$DATA_DIR"
}

function step_train {
    print_header "Step 4: Training JEPA Model"
    $PYTHON_CMD -m deepmreye train --data-dir "$DATA_DIR"
}

# Execute based on command
case $COMMAND in
    compile)
        step_compile
        ;;
    qa)
        step_qa
        ;;
    preprocess)
        step_preprocess
        ;;
    train)
        step_train
        ;;
    all)
        step_compile
        step_qa
        
        echo ""
        read -p "Have you finished QA labeling in the browser? Press Enter to continue with Preprocessing, or Ctrl+C to abort..."
        
        step_preprocess
        step_train
        ;;
    *)
        echo "Invalid command: $COMMAND"
        show_help
        exit 1
        ;;
esac

echo ""
echo "Pipeline execution finished successfully!"
