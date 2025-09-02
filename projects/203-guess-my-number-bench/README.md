# Guess My Number Bench

This is a benchmark for learning the [inspect-ai](https://inspect.aisi.org.uk/tutorial.html) evaluations framework by the UK AISI.

This bench is me learning how to use an LLM to evaluate is an answer is correct.

The model is simply playing "can you guess the number I'm thinking of from 1 to 10?" against itself.

claude-3-haiku-20240307 failed 5/5 times.

Run the eval with:
```bash
python -m inspect_ai eval guess-my-number-bench.py --epochs 5
```