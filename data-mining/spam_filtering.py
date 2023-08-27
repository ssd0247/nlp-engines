import string
from collections import Counter

import nltk

import numpy as np
import pandas as pd

REQ_STARTING_CHARS = set(string.ascii_lowercase + string.digits)

def load_data(verbose=False):
    data_path = r"./spam_ham_dataset.csv"
    df = pd.read_csv(data_path)
    if verbose:
        print("Data Summary:\n", df.describe())
        print("Data Feature Name:\n", df.columns)
        print("Unique Labels Assigned:\n", np.unique(df.loc[:, 'label']))
    return df

def is_dataset_balanced(spam_text, ham_text, tol=0.2):
    data_skew = len(spam_text) / len(ham_text)
    print(f"Data skewness factor (spam/ham ratio): {data_skew}")
    if tol > data_skew:
        print("Data needs to add more spam examples")
    else:
        print("Above minimum tolerance level ✅\nContinuing without any class balancing pre-processing task....")
    return

def get_text(docs: pd.Series):
    text = []
    indices = docs.index
    for idx in indices:
        doc = str(docs[idx]).replace('\r\n', ' ')
        text.append(doc)
    return text


def get_vocabs(text: list):
    vocabs = []
    for doc in text:
        sents = nltk.sent_tokenize(doc)
        words = [nltk.word_tokenize(sent) for sent in sents]
        corr_words = []
        for sent in words:
            for word in sent:
                if word.lower()[0] in REQ_STARTING_CHARS:
                    corr_words.append(word.lower())
        vocabs.append(Counter(corr_words))

    return vocabs

def set_idf(vocabs: list[dict], total_docs):
    '''Vocabs already contain the `term-frequency (TF)`. We have to multiply them with
    the remaining part, aka, the `inverse-document-frequency (IDF)`.'''
    all_tokens_by_docs = [set(key for key in vocab) for vocab in vocabs]
    for vocab in vocabs:
        for token in vocab.keys():
            n_docs = 0
            for keys in all_tokens_by_docs:
                if token in keys:
                    n_docs += 1
            idf = np.log(total_docs / n_docs)
            vocab[token] *= idf

    return vocabs

def show_details(spam_vocabs, ham_vocabs):
    ch_ham_idxs, ch_spam_idxs = set(), set()
    ham_idxs, spam_idxs = list(range(len(ham_vocabs))), list(range(len(spam_vocabs)))
    
    for i in range(1, 11, 1):
        while True:
            curr_spam_idx = np.random.choice(spam_idxs)
            if curr_spam_idx in ch_spam_idxs:
                continue
            ch_spam_idxs.add(curr_spam_idx)
            spam_vocab = spam_vocabs[curr_spam_idx]
            break

        while True:
            curr_ham_idx = np.random.choice(ham_idxs)
            if curr_ham_idx in ch_ham_idxs:
                continue
            ch_ham_idxs.add(curr_ham_idx)
            ham_vocab = ham_vocabs[curr_ham_idx]
            break
        
        print(f"Top 5 most common words in spam mail #{i}:\n", spam_vocab.most_common(5))
        print()
        print(f"Top 5 most common words in authentic mail #{i}:\n", ham_vocab.most_common(5))
        print('\n', '*'*50 ,'\n')

if __name__ == '__main__':
    df = load_data()    

    spam_docs = df.loc[np.where(df.loc[:, 'label'] == 'spam')[0], 'text']
    ham_docs = df.loc[np.where(df.loc[:, 'label'] == 'ham')[0], 'text']

    spam_text = get_text(spam_docs)
    ham_text = get_text(ham_docs)

    is_dataset_balanced(spam_text, ham_text)
    
    spam_vocabs = get_vocabs(spam_text)
    ham_vocabs = get_vocabs(ham_text)

    spam_vocabs = set_idf(spam_vocabs, len(spam_vocabs))
    ham_vocabs = set_idf(ham_vocabs, len(ham_vocabs))

    show_details(spam_vocabs, ham_vocabs)
    
    # all spam vocabs need to be negative scores
    for vocab in spam_vocabs:
        for key in vocab.keys():
            vocab[key] *= -1