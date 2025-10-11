import os
from dataclasses import dataclass, field
from typing import List


def _get_topic_labels() -> List[str]:
    """Parse topic labels from environment variable.
    
    - politics-news: Politics, current events, environment, social issues
    - technology-science: Tech, science, business, innovation
    - entertainment-media: TV, movies, music, comedy, gaming
    - sports: All sports and athletics
    - lifestyle: Food, travel, personal interests
    - creative-arts: Art, creativity, design
    """
    return os.getenv(
        "TOPIC_LABELS", 
        "politics-news,technology-science,entertainment-media,sports,lifestyle,creative-arts"
    ).split(",")


@dataclass(frozen=True)
class Settings:
    # NATS
    NATS_URL: str = os.getenv("NATS_URL", "nats://nats.nats.svc.cluster.local:4222")
    INPUT_STREAM: str = os.getenv("INPUT_STREAM", "bluesky-posts")
    OUTPUT_STREAM: str = os.getenv("OUTPUT_STREAM", "bluesky-posts-topics")
    INPUT_SUBJECT: str = os.getenv("INPUT_SUBJECT", "bluesky.posts")
    OUTPUT_SUBJECT: str = os.getenv("OUTPUT_SUBJECT", "bluesky.posts.topics")
    CONSUMER_NAME: str = os.getenv("CONSUMER_NAME", "topic-processor")
    # Queue group for load-balanced consumption across replicas. Defaults to CONSUMER_NAME
    QUEUE_GROUP: str = os.getenv("QUEUE_GROUP", os.getenv("CONSUMER_NAME", "topic-processor"))
    NUM_STREAM_REPLICAS: int = int(os.getenv("NUM_STREAM_REPLICAS", 1))

    # Service
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "nats-stream-processor-topics")

    # Consumer tuning
    ACK_WAIT_SECONDS: int = int(os.getenv("ACK_WAIT_SECONDS", 30))  # How long JetStream waits before redelivery
    MAX_DELIVER: int = int(os.getenv("MAX_DELIVER", 3))  # Max redeliver attempts
    MAX_ACK_PENDING: int = int(os.getenv("MAX_ACK_PENDING", 100))  # Max unacked messages in-flight per consumer
    
    # Output stream de-duplication window (in seconds)
    DUPLICATE_WINDOW_SECONDS: int = int(os.getenv("DUPLICATE_WINDOW_SECONDS", 600))  # 10 minutes

    # Topic Classification Model
    MODEL_NAME: str = os.getenv("MODEL_NAME", "richardr1126/roberta-base-zeroshot-v2.0-c-ONNX")
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/var/cache/models")
    MAX_SEQUENCE_LENGTH: int = int(os.getenv("MAX_SEQUENCE_LENGTH", 512))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.3))
    MULTI_LABEL: bool = os.getenv("MULTI_LABEL", "true").lower() == "true"
    HYPOTHESIS_TEMPLATE: str = os.getenv("HYPOTHESIS_TEMPLATE", "This text is related to {}")
    TOPIC_LABELS: List[str] = field(default_factory=_get_topic_labels)

    # Performance
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", 3))
    RETRY_DELAY: float = float(os.getenv("RETRY_DELAY", 1.0))

    # Health/metrics
    HEALTH_CHECK_PORT: int = int(os.getenv("HEALTH_CHECK_PORT", 8080))
    METRICS_ENABLED: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")


settings = Settings()
