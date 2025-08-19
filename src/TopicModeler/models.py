import typing as t
from datetime import UTC, datetime

import pydantic


class WordScore(pydantic.BaseModel):
    word: str
    score: float


class TopicWordDistribution(pydantic.BaseModel):
    topic_id: int
    word_scores: list[WordScore]


class DocumentTopicDistribution(pydantic.BaseModel):
    document_id: str
    topic_id: int
    score: float  # probability, coherence, etc.


class Topic(pydantic.BaseModel):
    id: int
    "The topic identifier"
    label: str | None = None  # user-defined or auto-generated
    top_words: list[str] = pydantic.Field(default_factory=list)
    word_distribution: TopicWordDistribution | None = None
    metadata: dict[str, t.Any] = pydantic.Field(default_factory=dict)


# -----------------------------
# Topic Model Run / Results
# -----------------------------


class TopicModelConfig(pydantic.BaseModel):
    algorithm: str  # e.g. "WCSVNtm", "LDA", "BERTopic"
    num_topics: int | None = None


class TopicModelResult(pydantic.BaseModel):
    model_id: str
    corpus_id: str
    config: TopicModelConfig
    topics: list[Topic]
    doc_topic_distributions: list[DocumentTopicDistribution]
    metrics: dict[str, float] = pydantic.Field(default_factory=dict)
    created_at: datetime = pydantic.Field(default_factory=lambda: datetime.now(UTC))
