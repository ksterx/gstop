# gstop

[![PyPI version](https://badge.fury.io/py/gstop.svg)](https://badge.fury.io/py/gstop)
[![Python Versions](https://img.shields.io/pypi/pyversions/gstop.svg)](https://pypi.org/project/gstop/)

gstop is a Python library that provides generation stopping criteria for Transformers-based language models. It allows you to define custom stop tokens and criteria to control the generation process and prevent the model from generating unwanted or irrelevant content.

## Features

- Define custom stop tokens and criteria for language model generation
- Supports various pre-defined stop token registries for popular language models
- Easy integration with the Transformers library
- Flexible and extensible architecture for adding new stop token registries

## Installation

You can install gstop using pip:

```bash
pip install gstop
```

## Usage

Here's a basic example of how to use gstop with the Transformers library:

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

In this example, we create an instance of `GenerationStopper` using the pre-defined stop tokens registry for the "mistral" model. We then use the `generate` method of the language model to generate text, passing the `stopping_criteria` parameter with the stopper's criteria. Finally, we format the generated text using the `format` method of the stopper to remove any stop tokens.

## Customization

You can customize the stop tokens and criteria by creating your own stop token registry or by modifying the existing ones. The stop token registries are defined in the `common.py` file.

To create a new stop token registry, you can add an entry to the `STOP_TOKENS_REGISTRY` dictionary with the desired stop tokens and their corresponding token IDs.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.
