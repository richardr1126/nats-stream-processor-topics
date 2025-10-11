import os
import asyncio
from typing import Dict, List, Optional
import time

from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

from .config import settings
from .logging_setup import get_logger
from .metrics import (
    topic_predictions_total,
    topic_confidence,
    model_inference_duration_seconds,
    processing_errors_total,
)

logger = get_logger(__name__)


class TopicClassifier:
    """Zero-shot topic classifier using RoBERTa model with ONNX optimization."""
    
    def __init__(self):
        self.classifier = None
        self.topic_labels = settings.TOPIC_LABELS
        self.hypothesis_template = settings.HYPOTHESIS_TEMPLATE
        self._model_loaded = False
        
    async def initialize(self) -> None:
        """Load the ONNX model using transformers pipeline."""
        try:
            logger.info("Initializing topic classifier", model=settings.MODEL_NAME)
            
            # Create model cache directory
            os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
            
            # Load the ONNX model using optimum and create pipeline
            logger.info("Loading ONNX model with pipeline")
            
            # Load model in a thread to avoid blocking
            def load_model():
                # Load ONNX model and tokenizer separately to ensure we use the ONNX version
                model = ORTModelForSequenceClassification.from_pretrained(
                    settings.MODEL_NAME,
                    cache_dir=settings.MODEL_CACHE_DIR,
                    file_name="model_quantized.onnx",
                )
                
                tokenizer = AutoTokenizer.from_pretrained(
                    settings.MODEL_NAME,
                    cache_dir=settings.MODEL_CACHE_DIR,
                )
                
                return pipeline(
                    "zero-shot-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,  # CPU inference
                    hypothesis_template=self.hypothesis_template,
                )
            
            # Run model loading in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            self.classifier = await loop.run_in_executor(None, load_model)
            
            self._model_loaded = True
            logger.info("Topic classifier initialized successfully", topics=self.topic_labels)
            
        except Exception as e:
            logger.error("Failed to initialize topic classifier", error=str(e))
            processing_errors_total.labels(error_type="model_init").inc()
            raise
    
    async def classify_topics(self, text: str) -> Optional[Dict]:
        """Classify text into topics (multi-label or single-label based on config)."""
        if not self._model_loaded:
            raise RuntimeError("Topic classifier not initialized")
        
        if not text or not text.strip():
            return None
        
        try:
            start_time = time.time()
            
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Run zero-shot classification with multi-label setting from config
            def run_classification():
                return self.classifier(
                    text,
                    candidate_labels=self.topic_labels,
                    multi_label=settings.MULTI_LABEL,
                )
            
            result = await loop.run_in_executor(None, run_classification)
            
            inference_time = time.time() - start_time
            model_inference_duration_seconds.observe(inference_time)
            
            # Process results - filter by confidence threshold
            topics = []
            probabilities = {}
            
            for label, score in zip(result['labels'], result['scores']):
                probabilities[label] = float(score)
                if score >= settings.CONFIDENCE_THRESHOLD:
                    topics.append(label)
            
            # If no topics meet threshold, use the top one
            if not topics and result['labels']:
                topics = [result['labels'][0]]
            
            # Get top topic and confidence
            top_topic = result['labels'][0] if result['labels'] else "unknown"
            top_confidence = float(result['scores'][0]) if result['scores'] else 0.0
            
            # Update metrics for each identified topic
            for topic in topics:
                topic_predictions_total.labels(topic=topic).inc()
            
            topic_confidence.observe(top_confidence)
            
            classification_result = {
                "topics": topics,
                "probabilities": probabilities,
                "top_topic": top_topic,
                "top_confidence": top_confidence,
            }
            
            logger.debug("Topic classification complete", 
                        topics=topics, 
                        top_confidence=top_confidence)
            
            return classification_result
            
        except Exception as e:
            logger.error("Topic classification failed", 
                        error=str(e), 
                        error_type=type(e).__name__)
            processing_errors_total.labels(error_type="single_analysis").inc()
            raise


# Global topic classifier instance
topic_classifier = TopicClassifier()
