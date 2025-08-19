from abc import abstractmethod

from TopicModeler.models import TopicModelConfig, TopicModelResult


class BaseTopicModeler:
    def __init__(self, model_config: TopicModelConfig):
        self._config = model_config

    @abstractmethod
    def preprocess_documents(self, documents: list[str]) -> list[list[list[str]]]:
        pass

    @abstractmethod
    def run_topic_modeling(
        self, documents: list[str], *args, **kwargs
    ) -> TopicModelResult:
        pass
