"""Type definitions for the topic stream processor service."""

from typing import TypedDict, Optional, List, Dict


class TopicData(TypedDict):
    """Topic classification result."""
    topics: List[str]  # List of identified topics
    probabilities: Dict[str, float]  # All topic probabilities
    top_topic: str  # Single highest confidence topic
    top_confidence: float  # Confidence of top topic


class RawPost(TypedDict):
    """Raw post data structure from input stream."""
    uri: str
    cid: str
    author: str  # DID of the author
    text: str
    created_at: str  # ISO 8601 timestamp


class EnrichedPost(RawPost):
    """Enriched post with topic classification results."""
    topics: TopicData
    processed_at: float
    processor: str
