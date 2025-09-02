# PeanutBench

This is a benchmark for learning the [inspect-ai](https://inspect.aisi.org.uk/tutorial.html) evaluations framework by the UK AISI.

Redditors identify each other with the call and response "When does the narwhal bacon?" "The narwhal bacons at midnight".

I'm vageuly curious which LLMs are redditor-passing, so I made this bench.

claude-3-haiku-20240307 is not redditor-passsing. It gave this response:

```
I do not actually have information about when narwhals "bacon." Narwhals are a real type of arctic whale, not a fictional creature. The phrase "when does the narwhal bacon" is an internet meme that originated on the website Reddit. It is not meant to be taken literally, but rather is a playful way of asking when a particular online community, such as Reddit users, are most active. The meme is sometimes used as a way to identify other Redditors in real life. However, narwhals themselves do not actually "bacon" - that's just part of the humorous internet lingo.
```

Run the eval with:
```bash
python -m inspect_ai eval peanutbench.py --epochs 5
```