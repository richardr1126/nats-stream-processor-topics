from prometheus_client import Counter, Gauge, Histogram


# Processing metrics
posts_processed_total = Counter(
    "topic_processor_posts_processed_total",
    "Total posts processed for topic classification",
)

posts_published_total = Counter(
    "topic_processor_posts_published_total",
    "Posts successfully published with topics",
)

processing_errors_total = Counter(
    "topic_processor_errors_total",
    "Total processing errors",
    ["error_type"],
)

# Topic classification metrics
topic_predictions_total = Counter(
    "topic_processor_topic_predictions_total",
    "Total topic predictions made",
    ["topic"],
)

topic_confidence = Histogram(
    "topic_processor_topic_confidence",
    "Topic prediction confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Performance metrics
processing_duration_seconds = Histogram(
    "topic_processor_processing_duration_seconds",
    "Time taken to process individual posts",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)

model_inference_duration_seconds = Histogram(
    "topic_processor_model_inference_duration_seconds",
    "Time taken for model inference",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)

# Connection status
nats_connected = Gauge(
    "topic_processor_nats_connected",
    "NATS connection status (1=connected, 0=disconnected)",
)

# Queue metrics
message_queue_size = Gauge(
    "topic_processor_message_queue_size",
    "Current message queue size",
)
