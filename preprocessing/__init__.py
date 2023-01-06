"""# Public API to interact with the preprocessing library.

### WHY DO WE NEED TO PREPROCESS
- With respect to the task, the raw text needs to be sanitized to generate
appropriate text representation for downstream computations and tasks.

### SUB-COMPONENTS OF TEXT PREPROCESSING
- The preprocessing may include sub-operations like (add if more preprocessing steps are required):
    - Stopword removal.
    - Case Conversions.
    - Special character removal.
    - Extra whitespace removal.
    - Expanding contractions.
    - Abbreviation resolving.
    - Correcting misspelled words.
    - POS-Tagging.

"""

import re
import string
from pprint import pprint

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
import contractions


def _tokenize_text(text):
    sentences = sent_tokenize(text)
    word_tokens = [word_tokenize(sent) for sent in sentences]
    return word_tokens

def _remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    word_tokens = list(filter(None, [pattern.sub(r'', token) for token in tokens]))
    return word_tokens


def _remove_characters_before_tokenization(sentence, keep_apostrophes=False):
    sentence = sentence.strip()
    if keep_apostrophes:
        PATTERN = r'[?|$|&|*|%|@|(|)|~]' # add other characters here to remove them    
    else:
        PATTERN = r'[^a-zA-Z0-9 ]' # only extract alpha-numeric characters and whitespace
    filtered_sentence = re.sub(PATTERN, r'', sentence)
    return filtered_sentence


def _expand_contractions(sentence, contraction_mapping):
        """Expand the contractions.
        
        Parameters
        ----------
        sentence : (vector of words)
            list of words in a sentence.
        
        contraction_mapping : dict
            a pre-built map of contractions and their elongated forms.
            Implemented using `dict()`.
        
        Returns
        -------
        expanded tokens/words as a vector of words.

        """
        contractions_pattern = re.compile(
            '({})'.format('|'.join(contraction_mapping.keys())),
            flags=re.IGNORECASE|re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match) \
                if contraction_mapping.get(match) \
                    else contraction_mapping.get(match.lower())
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction
        
        expanded_sentence = contractions_pattern.sub(expand_match, sentence)
        return expanded_sentence

def _remove_stopwords(tokens, lang='english'):
    stopword_list = nltk.corpus.stopwords.words(lang)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens

def _correct_misspelled_tokens_default(old_word):
    """Default spell-checker. Native while loop used. [ITERATIVE ALGORITHM]
    Encouraged to be used on `large-datasets`.
    """
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    step = 1

    while True:
        # check for semantically correct word
        if wordnet.synsets(old_word):
            #print("Final correct word: ", old_word)
            break
        # remove one repeated character
        new_word = repeat_pattern.sub(match_substitution, old_word)

        if new_word != old_word:
            #print("Step: {}\t Word: {}".format(step, new_word))
            step += 1 # update step
            # update old word to last substitution state
            old_word = new_word
            continue
        else:
            #print("Final word: ", new_word)
            break
    
    return new_word

def _correct_mispelled_tokens(tokens):
    """Spell-checker. Uses inner function `replace`. [RECURSIVE ALGORITHM]"""
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
    
    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens

def preprocess(text, rem_chars_after_tok=False, pos=False, lower=True, correct_misspelled=True, default_spellchecker=True):
    """# [Public-API]"""
    # Get a filtered list of tokens from raw text
    cleaned_corpus = []
    if rem_chars_after_tok:
        tokens_list = _tokenize_text(text)
        for sentence_tokens in tokens_list:
            cleaned_corpus.append(list(filter(None, [_remove_characters_after_tokenization(tokens) for tokens \
                in sentence_tokens])))
    else:
        cleaned_corpus = [_remove_characters_before_tokenization(sentence, keep_apostrophes=True) for sentence in text]
    
    # expand all the contractions in the set of filtered tokens
    expanded_corpus = [_expand_contractions(sentence, contractions.contractions_map) \
        for sentence in cleaned_corpus]
    
    # Case conversions
    if lower:
        case_insensitive_corpus = [[[tok.lower() for tok in sent] for sent in doc] for doc in expanded_corpus]
    else:
        case_insensitive_corpus = [[[tok.upper() for tok in sent] for sent in doc] for doc in expanded_corpus]

    # Removing stopwords
    expanded_corpus_tokens = [_tokenize_text(text) for text in case_insensitive_corpus]
    filtered_list = [[_remove_stopwords(tokens) for tokens in sentence_tokens] for sentence_tokens in expanded_corpus_tokens]

    # Correcting mispelled words
    correct_tokens = []
    if correct_misspelled:
        if default_spellchecker:
            pass
            for doc in filtered_list:
                correct_tokens.append([])
                for sent in doc:
                    correct_tokens.append([])
                    for tok in sent:
                        correct_tokens[-1][-1].append(_correct_misspelled_tokens_default(tok))
        else:
            for doc in filtered_list:
                correct_tokens.append([])
                for sent in doc:
                    correct_tokens[-1].append(_correct_mispelled_tokens(sent))
    else:
        correct_tokens = filtered_list