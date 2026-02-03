#!/bin/bash
# Eagle-HalluShift: Quick Run Script
# Usage: ./run.sh [phase0|phase1|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Check CUDA
check_cuda() {
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "CUDA: Available"
        return 0
    else
        echo "CUDA: Not available"
        return 1
    fi
}

# Phase 0: Validation Study
run_phase0() {
    print_header "Phase 0: Validation Study"

    # Check CUDA
    if ! check_cuda; then
        print_warning "No GPU available. Running mock test only."
        python scripts/phase0_validation.py --mock
        return
    fi

    # Real inference
    python scripts/phase0_inference_hook.py \
        --base_model meta-llama/Llama-2-7b-chat-hf \
        --ea_model yuhuili/EAGLE-llama2-chat-7B \
        --max_samples 200 \
        --output_dir phase0_results

    # Analyze results
    echo ""
    print_header "Phase 0: Analysis"
    python scripts/phase0_validation.py \
        --data_dir phase0_results \
        --output_dir phase0_results
}

# Phase 1: Layer Consistency Training
run_phase1() {
    print_header "Phase 1: Layer Consistency Training"

    # Check CUDA
    if ! check_cuda; then
        print_error "Phase 1 requires GPU. Please run on K8s with H200."
        echo "To deploy on K8s:"
        echo "  kubectl apply -f k8s/secrets.yaml"
        echo "  kubectl apply -f k8s/phase1-job.yaml"
        return 1
    fi

    # Check for training data
    if [ ! -f "data/sharegpt_train.json" ]; then
        print_error "Training data not found at data/sharegpt_train.json"
        echo "Please download the ShareGPT dataset first."
        return 1
    fi

    # Run training
    deepspeed --num_gpus=$(nvidia-smi -L | wc -l) scripts/phase1_train.py \
        --base_model meta-llama/Llama-2-7b-chat-hf \
        --data_path data/sharegpt_train.json \
        --lambda_consistency 0.1 \
        --num_epochs 3 \
        --output_dir checkpoints/phase1
}

# Main
case "${1:-help}" in
    phase0)
        run_phase0
        ;;
    phase1)
        run_phase1
        ;;
    all)
        run_phase0
        echo ""
        run_phase1
        ;;
    mock)
        print_header "Mock Test"
        python scripts/phase0_validation.py --mock
        ;;
    help|*)
        echo "Eagle-HalluShift: Layer Dynamics for Speculative Decoding"
        echo ""
        echo "Usage: ./run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  phase0    Run Phase 0 validation study"
        echo "  phase1    Run Phase 1 training"
        echo "  all       Run all phases"
        echo "  mock      Run mock test (no GPU needed)"
        echo "  help      Show this help"
        echo ""
        echo "For K8s deployment:"
        echo "  kubectl apply -f k8s/secrets.yaml"
        echo "  kubectl apply -f k8s/phase0-job.yaml"
        echo "  kubectl apply -f k8s/phase1-job.yaml"
        ;;
esac
