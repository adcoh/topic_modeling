import typing as t

from TopicModeler.models import TopicModelConfig


class SentenceId(t.NamedTuple):
    doc_id: int
    sentence_id: int

    def __str__(self) -> str:
        return f"D{self.doc_id}_S{self.sentence_id}"


class WCSVNTConfig(TopicModelConfig):
    algorithm: str = "WCSVNtm"
    alpha_word: float = 0.01
    alpha_doc: float = 0.01
