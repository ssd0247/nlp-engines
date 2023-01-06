import random
import math
import numpy as np

# Assume that we are getting the context vector from encoders.
# Encoder encodes sentences in one language (generates context vector)
# Decoder decodes sentences in another language (consumes context vector)
# 
# PROCEDURE -
# ---------
# 1). Stack of encoders, release one context vector, that is used in decoder in every time step.
# 2). The lowest encoder gets the embedding of each word in a sentence in a (number_of_words, vocab_len)
#       format. The number of words change after every translation
#
# ADVANTANGES -
# -----------
# 1). This way we can design, build and test the ENCODER and DECODER separately for the
#       "figuring out" the same task. They are connected later via the use of context vector.
# 2). They are allowed to use the same computing resources as they can be run independently.
#
#
# DISADVANTAGES - 
# -------------
# 1). More novel ways are needed to be searched for sincec not every architecture support
#       such a "complete-context-vector-handing-over" behavior.
# 2). How multi-modality models are incorporated in such a framework of building encoder-decoder
#       architectures.
#
# The models are used for making a probability distribution on the complete vocabulary for this text.
# This context vector is used in a downstream task to figure out the next word.
#   
# In the case of multiple docs, combine the 

len_text = 10
len_vocab = 5

context = [random.randint(1, 5) / 10 for _ in range(len_text * len_vocab)]
context = np.array(context).reverse(len_text, len_vocab)

