#!/bin/bash

# Sequential training script for translation models with different configs

set -e  # Exit on first error

echo "Starting sequential training runs..."
echo ""

# Config 1: 5d
echo "=== Running config_translation_5_d.yaml ==="
python -m src.objects_game.train_translation config_translation_5_d.yaml
echo "✓ Completed config_translation_5_d.yaml"
echo ""

# Config 2: tf
echo "=== Running config_translation_tf.yaml ==="
python -m src.objects_game.train_translation config_translation_tf.yaml
echo "✓ Completed config_translation_tf.yaml"
echo ""

# Config 3: 5d tf
echo "=== Running config_translation_5_d_tf.yaml ==="
python -m src.objects_game.train_translation config_translation_5_d_tf.yaml
echo "✓ Completed config_translation_5_d_tf.yaml"
echo ""

echo "All training runs completed successfully!"
