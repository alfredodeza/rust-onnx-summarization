# rust-onnx-summarization
Do summarization using ONNX and Rust


## Use Python and HugginFace to get ONNX
This example uses GPT-2. Install Python and use the `requirement.txt` to install everything you need.

Run the following command:

```
python -m transformers.onnx --model=gpt2 rsummarizer/out/
```

Then run `python generate.py` to create the vocab and merges files that are also needed.

## Help needed
The current implementation doesn't generate any problems with the compiler, but only provides a single word out regardless of the input.

According to ChatGPT this is because _"The current approach takes the index with the maximum value from the output tensor, which only gives you a single word as output. Instead, you should generate the output sequence by sampling from the probability distribution of the model output."_, but my suspicion is that this has to be about the `config.json` and `merges.txt` files that might not be quite right.

Please file a pull request with a plausible fix if you have one!
