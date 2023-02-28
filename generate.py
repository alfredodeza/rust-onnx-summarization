from transformers import GPT2Tokenizer
"""
Creates the necessary files for the Tokenizer to work
Requires the `transformers` package installed with Python.
Additionally, use `python -m transformers.onnx --model=gpt2 out/` to create the ONNX model
"""

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained("rsummarizer/out")
tokenizer.save_vocabulary("rsummarizer/out")
