import json

import torch
from loguru import logger
from transformers import StoppingCriteria, StoppingCriteriaList


class GenerationStopper(StoppingCriteria):
    def __init__(
        self,
        stop_tokens: dict[str, list[int | list[int]]] | None = None,
        tokenizer_name: str | None = None,
    ):
        if stop_tokens is None and tokenizer_name is None:
            raise ValueError("Either stop_tokens or tokenizer_name must be provided")
        elif stop_tokens and tokenizer_name:
            loaded_stop_tokens = StopTokensRegistry().get(tokenizer_name)
            for w, t in stop_tokens.items():
                loaded_stop_tokens[w] = t
            stop_tokens = loaded_stop_tokens
        elif stop_tokens is None and tokenizer_name:
            stop_tokens = StopTokensRegistry().get(tokenizer_name)

        self.stop_token_ids = []
        for t in stop_tokens.values():
            if any(isinstance(x, list) for x in t):  # if t is nested list
                for x in t:
                    self.stop_token_ids.append(torch.tensor(x))
            else:
                self.stop_token_ids.append(torch.tensor(t))
            assert isinstance(t, list) or isinstance(t, int)
        self.stop_token_words = stop_tokens.keys()

    def __repr__(self):
        return f"Stopping words: {self.stop_token_words}"

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for t in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(t) :].to("cpu"), t).all():
                return True
        return False

    @property
    def criteria(self):
        return StoppingCriteriaList([self])

    def format(self, sentence: str):
        for w in self.stop_token_words:
            if w in sentence[-len(w) :]:
                sentence = sentence[: -len(w)]
        return sentence


class StopTokensRegistry:
    REGISTRY_PATH = "stop_tokens.json"

    def __init__(self):
        with open(self.REGISTRY_PATH, "r") as f:
            self.registry = json.load(f)

    def get(self, model_name: str):
        if model_name in self.registry.keys():
            return self.registry[model_name]
        else:
            logger.warning(f"No stop tokens found for {model_name}")
            return {}

    def register(
        self, tokenizer_name: str, word: str, stop_token_ids: list[int | list[int]]
    ):
        if tokenizer_name not in self.registry:
            self.registry[tokenizer_name] = {}
        self.registry[tokenizer_name][word] = stop_token_ids

    def remove(self, model_name: str, key: str):
        del self.registry[model_name][key]

    def save(self):
        with open(self.REGISTRY_PATH, "w") as f:
            f.write(self.registry)

    def show(self):
        print(self.registry)
