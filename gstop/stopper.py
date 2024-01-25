import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class GenerationStopper(StoppingCriteria):
    def __init__(self, stop_tokens: dict[str, list[int | list[int]]]):
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
