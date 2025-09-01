# PeanutBench

This is a benchmark for learning the [inspect-ai](https://inspect.aisi.org.uk/tutorial.html) evaluations framework by the UK AISI.

It simply asks an LLM: "What's something you would eat honey roasted? Reply with just an emoji."
The answer, of course, is "🥜".

claude-3-haiku-20240307 passed the test 5/5 times.

Run the eval with:
```bash
python -m inspect_ai eval peanutbench.py
```