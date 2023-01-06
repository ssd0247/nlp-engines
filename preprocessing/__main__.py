"""Used for test runs on the preprocessing library, as of now.

It provides a command-line interface for testing the preprocessing package.
When the `preprocess` package is run from the command line, this script is run.

"""
from pprint import pprint
import preprocessing as pp

dummy_corpus = ["The brown fox wasn't that quick and he couldn't win the race.",
        "Hey that's great deal! I just bought a phone for $199.",
        "@@You'll (learn) a **lot** in the book. Python is an amazing language!@@"]
    
token_list = [pp._tokenize_text(text) for text in dummy_corpus]
pprint(token_list)