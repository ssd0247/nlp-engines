import random
import math
import numpy as np
import nltk
from preprocessing import preprocess

# Assume that we are getting the context vector from encoders.
# Encoder encodes sentences in one language (generates context vector)
# Decoder decodes sentences in another language (consumes context vector)
# 
# PROCEDURE -
# ---------
# 1). Stack of encoders, release one context vector, that is used in decoder in every time step.
# 2). The lowest encoder gets the embedding of each word in a sentence in a (num_of_words, embedding_len)
#       format. The number of words change after full front traversal.
#
# The models are used for making a probability distribution on the complete vocabulary for a single full-pass
# through the decoder.
#
# Fully-connected networks are run independent of each other. They are multilayer perceptron.
# The output is a vector of vectors that has dimension same as the input ie, (num_of_words, embedding_len).
#
# This context vector is used in a downstream task to figure out the next word.
#   
# In the case of multiple docs, combine the 

embedding_len = 50
vocab_len = 500
# 1 - then alright that word in there in the sentence.
# 0 - that word is not in the sentence.
def preprocess_text(raw_text):
    tokens = preprocess(raw_text) #[docs, sents, words]
    return tokens

text = [[[i for i in range(embedding_len)] for _ in num_words] for _ in num_sents]
sents = []
for sent in text:
    

def determine_vocab_len(raw_text):
    '''Determine the useful words in the complete collection of documents.
    '''
    pass

context = [random.randint(1, 5) / 10 for _ in range(len_text * len_vocab)]
context = np.array(context).reverse(len_text, len_vocab)

if __name__ == '__main__':
    # Used for checking the functionality of dummy datasets.
    # For time-being, it's used for tests and finding bugs.