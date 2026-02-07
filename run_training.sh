#!/bin/bash
# Run training with proper Python path

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python src/training.py --config config/example_config.yaml
