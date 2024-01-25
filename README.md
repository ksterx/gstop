# gstop
Generation Stopping Criteria for transformers Language Model

## Installation

```bash
pip install gstop
```

## Usage

```python
from gstop import GenerationStopper, STOP_TOKENS_REGISTRY
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
stopper = GenerationStopper(STOP_TOKENS_REGISTRY["mistral"])

input_ids = tokenizer("Hello, world!", return_tensors="pt").input_ids

out = model.generate(input_ids, stopping_criteria=stopper.criteria)
print(stopper.format(tokenizer.decode(out[0])))
```
