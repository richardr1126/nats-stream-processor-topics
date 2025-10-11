"""
Test script for zero-shot topic classification using ONNX optimized model.
This script tests the deberta-v3-large-zeroshot-v2.0-ONNX model for multi-label topic classification.
"""

import os
import time
import asyncio
from typing import Dict, List
from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

# Define comprehensive topic labels for Bluesky posts
TOPIC_LABELS = [
    "politics",
    "technology",
    "entertainment",
    "sports",
    "news",
    "personal life",
    "art and creativity",
    "science",
    "business",
    "social issues",
    "education",
    "health and wellness",
    "gaming",
    "food and cooking",
    "travel",
    "music",
    "environment",
    "humor and memes",
]

# Model configuration
MODEL_NAME = "richardr1126/roberta-base-zeroshot-v2.0-c-ONNX"
MAX_SEQUENCE_LENGTH = 512
CONFIDENCE_THRESHOLD = 0.3  # Lower threshold for multi-label
MULTI_LABEL = os.getenv("MULTI_LABEL", "true").lower() == "true"  # Enable multi-label classification
HYPOTHESIS_TEMPLATE = "This post is about {}"
MODEL_CACHE_DIR = "./model_cache_test"
QUANTIZED_MODEL_DIR = "./model_cache_test/roberta_quantized"

# Test posts covering various topics
TEST_POSTS = [
    "Just finished implementing a new machine learning model in Python using PyTorch. The training took 6 hours but the results are amazing!",
    "Breaking: New climate change report shows urgent need for action. World leaders must respond now!",
    "Can't believe the game last night! That final shot was incredible. Best basketball game I've seen in years!",
    "Made some delicious pasta carbonara tonight. The secret is using fresh eggs and good quality parmesan. Recipe in thread 🍝",
    "The new Marvel movie was absolutely fantastic! The visual effects were stunning and the story kept me engaged throughout.",
    "Feeling grateful for my family and friends today. Life is too short not to appreciate the people who matter most ❤️",
    "New study shows promising results for cancer treatment. This could be a major breakthrough in medical science.",
    "Just launched my startup! We're building tools to help developers be more productive. Check out our website!",
    "The president's new policy on immigration is causing controversy. What do you think about this approach?",
    "Spent the weekend hiking in Yosemite. The views were breathtaking! Nature is truly amazing 🏔️",
]


class TopicClassifier:
    """Zero-shot topic classifier using DeBERTa model with ONNX optimization."""
    
    def __init__(self, labels: List[str], hypothesis_template: str = HYPOTHESIS_TEMPLATE):
        self.classifier = None
        self.labels = labels
        self.hypothesis_template = hypothesis_template
        self._model_loaded = False
        
    async def initialize(self, use_quantized: bool = True) -> None:
        """Load the ONNX model using transformers pipeline."""
        try:
            if use_quantized:
                # Check if quantized model exists
                if not os.path.exists(os.path.join(QUANTIZED_MODEL_DIR, "model_quantized.onnx")):
                    raise FileNotFoundError(
                        f"Quantized model not found at {QUANTIZED_MODEL_DIR}. "
                        f"Please run 'uv run quantize_model.py' first to create the quantized model."
                    )
                print(f"Initializing topic classifier with QUANTIZED model from: {QUANTIZED_MODEL_DIR}")
                model_file = "model_quantized.onnx"
            else:
                # Check if non-quantized model exists
                if not os.path.exists(os.path.join(QUANTIZED_MODEL_DIR, "model.onnx")):
                    raise FileNotFoundError(
                        f"Non-quantized model not found at {QUANTIZED_MODEL_DIR}. "
                        f"Please run 'uv run quantize_model.py' first to create both models."
                    )
                print(f"Initializing topic classifier with NON-QUANTIZED model from: {QUANTIZED_MODEL_DIR}")
                model_file = "model.onnx"
            
            print(f"Topic labels: {', '.join(self.labels)}")
            print(f"Loading ONNX model ({model_file})...")
            
            def load_model():
                # Load model from local directory
                model = ORTModelForSequenceClassification.from_pretrained(
                    QUANTIZED_MODEL_DIR,
                    file_name=model_file,
                )
                
                tokenizer = AutoTokenizer.from_pretrained(
                    QUANTIZED_MODEL_DIR,
                )
                
                # Create zero-shot classification pipeline
                return pipeline(
                    "zero-shot-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,  # CPU inference
                )
            
            # Run model loading in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            self.classifier = await loop.run_in_executor(None, load_model)
            
            self._model_loaded = True
            print("Topic classifier initialized successfully\n")
            
        except Exception as e:
            print(f"Failed to initialize topic classifier: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def classify_topics(self, text: str) -> Dict:
        """Classify text into topics (multi-label or single-label based on config)."""
        if not self._model_loaded:
            raise RuntimeError("Topic classifier not initialized")
        
        if not text or not text.strip():
            return None
        
        try:
            start_time = time.time()
            
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Run zero-shot classification with multi-label setting
            def run_classification():
                return self.classifier(
                    text,
                    candidate_labels=self.labels,
                    hypothesis_template=self.hypothesis_template,
                    multi_label=MULTI_LABEL,  # Use MULTI_LABEL from config
                )
            
            result = await loop.run_in_executor(None, run_classification)
            
            inference_time = time.time() - start_time
            
            # Process results - filter by confidence threshold
            topics = []
            probabilities = {}
            
            for label, score in zip(result['labels'], result['scores']):
                probabilities[label] = float(score)
                if score >= CONFIDENCE_THRESHOLD:
                    topics.append({
                        'label': label,
                        'confidence': float(score)
                    })
            
            # Sort topics by confidence (highest first)
            topics.sort(key=lambda x: x['confidence'], reverse=True)
            
            classification_result = {
                "topics": topics,
                "probabilities": probabilities,
                "inference_time": inference_time
            }
            
            return classification_result
            
        except Exception as e:
            print(f"Topic classification failed: {e}")
            raise


async def test_classifier_with_config(model_type: str, use_quantized: bool):
    """Test the topic classifier with a specific model configuration."""
    print("=" * 80)
    print(f"TESTING: {model_type}")
    print("=" * 80)
    print()
    
    # Initialize classifier
    classifier = TopicClassifier(labels=TOPIC_LABELS)
    await classifier.initialize(use_quantized=use_quantized)
    
    total_time = 0.0
    
    # Test each post
    for i, post in enumerate(TEST_POSTS, 1):
        print(f"Test {i}/{len(TEST_POSTS)}")
        print("-" * 80)
        print(f"Post: {post[:100]}..." if len(post) > 100 else f"Post: {post}")
        print()
        
        result = await classifier.classify_topics(post)
        
        if result:
            inference_time = result['inference_time']
            total_time += inference_time
            
            print(f"Inference time: {inference_time:.3f}s")
            print(f"Topics found ({len(result['topics'])}):")
            
            for topic in result['topics'][:3]:  # Show top 3
                print(f"  • {topic['label']:20s} - {topic['confidence']:.1%}")
            
            if len(result['topics']) > 3:
                print(f"  ... and {len(result['topics']) - 3} more")
        else:
            print("No topics classified (empty text)")
        
        print()
    
    avg_time = total_time / len(TEST_POSTS)
    print("=" * 80)
    print(f"TOTAL INFERENCE TIME: {total_time:.3f}s")
    print(f"AVERAGE TIME PER POST: {avg_time:.3f}s")
    print("=" * 80)
    print()
    
    return total_time, avg_time


async def test_classifier():
    """Test the topic classifier comparing non-quantized and quantized ONNX models."""
    print("\n")
    print("=" * 80)
    print("ZERO-SHOT TOPIC CLASSIFICATION COMPARISON TEST")
    print("=" * 80)
    print()
    
    results = {}
    
    # Check if both models exist
    has_non_quantized = os.path.exists(os.path.join(QUANTIZED_MODEL_DIR, "model.onnx"))
    has_quantized = os.path.exists(os.path.join(QUANTIZED_MODEL_DIR, "model_quantized.onnx"))
    
    if not has_non_quantized or not has_quantized:
        print("⚠️  Required models not found!")
        if not has_non_quantized:
            print(f"   Missing: {QUANTIZED_MODEL_DIR}/model.onnx")
        if not has_quantized:
            print(f"   Missing: {QUANTIZED_MODEL_DIR}/model_quantized.onnx")
        print()
        print("Please run 'uv run quantize_model.py' first to create both models.")
        return
    
    # Test 1: Non-quantized ONNX
    print("\n🔹 Test 1: Non-Quantized ONNX Model (FP32)")
    print()
    total_time, avg_time = await test_classifier_with_config("NON-QUANTIZED ONNX (FP32)", use_quantized=False)
    results["Non-Quantized"] = {"total": total_time, "avg": avg_time}
    
    print("\n" + "=" * 80)
    print()
    input("Press Enter to continue to quantized model test...")
    print()
    
    # Test 2: Quantized ONNX
    print("\n🔹 Test 2: Quantized ONNX Model (INT8)")
    print()
    total_time, avg_time = await test_classifier_with_config("QUANTIZED ONNX (INT8)", use_quantized=True)
    results["Quantized"] = {"total": total_time, "avg": avg_time}
    
    # Print comparison
    print("\n")
    print("=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Model Type':<25} {'Total Time':<15} {'Avg Time':<15} {'Speedup':<10}")
    print("-" * 80)
    
    non_q = results["Non-Quantized"]
    q = results["Quantized"]
    speedup = non_q["avg"] / q["avg"]
    
    print(f"{'Non-Quantized ONNX':<25} {non_q['total']:>10.3f}s     {non_q['avg']:>10.3f}s     {'1.00x':<10}")
    print(f"{'Quantized ONNX (INT8)':<25} {q['total']:>10.3f}s     {q['avg']:>10.3f}s     {speedup:>7.2f}x")
    
    print()
    print(f"⚡ Speedup: {speedup:.2f}x faster with quantized model!")
    print(f"💾 Model size reduction: ~4x smaller (INT8 vs FP32 weights)")
    print()
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_classifier())
