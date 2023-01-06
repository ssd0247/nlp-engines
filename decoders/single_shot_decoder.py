import random
import math
import numpy as np

# Assume that we are getting the context vector for the complete raw text, AT ONCE!
#
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

