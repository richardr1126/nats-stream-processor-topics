"""
Script to quantize the DeBERTa zero-shot classification model for better performance.
This only needs to be run once to create the quantized model.
"""

import os
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.exporters.onnx import main_export
from onnxruntime.quantization import quantize_dynamic, QuantType

# Model configuration
MODEL_NAME = "MoritzLaurer/roberta-base-zeroshot-v2.0-c"
MODEL_CACHE_DIR = "./model_cache_test"
QUANTIZED_MODEL_DIR = "./model_cache_test/roberta_quantized"

def quantize_model():
    """Export and quantize the model."""
    print("=" * 80)
    print("QUANTIZING ROBERTA ZERO-SHOT CLASSIFICATION MODEL")
    print("=" * 80)
    print()
    
    # Create directories
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    os.makedirs(QUANTIZED_MODEL_DIR, exist_ok=True)
    
    print(f"Loading model: {MODEL_NAME}")
    print("Exporting to ONNX format (opset version 14)...")
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
    )
    
    # Create a temporary directory for initial ONNX export
    temp_dir = os.path.join(MODEL_CACHE_DIR, "temp_onnx")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Export using optimum's main_export function with opset 14
    print("Running ONNX export with opset 14...")
    from optimum.exporters.onnx import main_export
    
    main_export(
        model_name_or_path=MODEL_NAME,
        output=temp_dir,
        task="sequence-classification",
        opset=14,
        cache_dir=MODEL_CACHE_DIR,
    )
    
    print("Model exported to ONNX successfully")
    print()
    
    # Find the ONNX model file
    onnx_model_path = os.path.join(temp_dir, "model.onnx")
    regular_model_path = os.path.join(QUANTIZED_MODEL_DIR, "model.onnx")
    quantized_model_path = os.path.join(QUANTIZED_MODEL_DIR, "model_quantized.onnx")
    
    print()
    print("Saving regular ONNX model...")
    shutil.copy2(onnx_model_path, regular_model_path)
    print(f"Regular model saved to: {regular_model_path}")
    
    print()
    print("Quantizing model using ONNX Runtime...")
    print("  - Type: Dynamic quantization")
    print("  - Weight type: INT8")
    print()
    
    # Use onnxruntime's quantize_dynamic directly
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=quantized_model_path,
        weight_type=QuantType.QInt8,
    )
    
    print("Quantization complete!")
    print(f"Quantized model saved to: {quantized_model_path}")
    
    # Save tokenizer and config to the quantized model directory
    print("Saving tokenizer and config...")
    tokenizer.save_pretrained(QUANTIZED_MODEL_DIR)
    
    # Copy config from temp directory
    temp_config = os.path.join(temp_dir, "config.json")
    if os.path.exists(temp_config):
        shutil.copy2(temp_config, QUANTIZED_MODEL_DIR)
    
    # Clean up temp directory
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    print()
    print("=" * 80)
    print("QUANTIZATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Quantized model location: {QUANTIZED_MODEL_DIR}")
    print("You can now use this model in test_classification_new.py")
    

if __name__ == "__main__":
    quantize_model()
