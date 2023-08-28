import argparse
import string
from collections import Counter

import nltk

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REQ_STARTING_CHARS = set(string.ascii_lowercase + string.digits)

STOPWORDS = nltk.corpus.stopwords.words("english")

def load_data(data_path, train_size, verbose=False):
    df = pd.read_csv(data_path)

    cols = ["Unnamed: 0", "text", "label_num", "label"]
    
    X, y = df.drop(["label"], axis=1).to_numpy(), df["label"].to_numpy()
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=train_size, shuffle=True, random_state=123, stratify=y)
    train_y = train_y.reshape(*train_y.shape, 1)
    test_y = test_y.reshape(*test_y.shape, 1)
    train_npy, test_npy = np.hstack((train_X, train_y)), np.hstack((test_X, test_y))
    train_df, test_df = pd.DataFrame(train_npy, columns=cols), pd.DataFrame(test_npy, columns=cols)

    if verbose:
        print("Data Summary:\n", df.describe())
        print("Data Feature Name:\n", df.columns)
        print("Unique Labels Assigned:\n", np.unique(df.loc[:, 'label']))

    return train_df, test_df

def is_dataset_balanced(spam_text, ham_text, tol=0.2):
    data_skew = len(spam_text) / len(ham_text)
    print(f"Data skewness factor (spam/ham ratio): {data_skew}")
    if tol > data_skew:
        print("Data needs to add more spam examples")
        return False
    else:
        print("Above minimum tolerance level ✅\nContinuing without any class balancing pre-processing task....")
        return True

def get_text(docs: pd.Series):
    text = []
    indices = docs.index
    for idx in indices:
        doc = str(docs[idx]).replace('\r\n', ' ')
        text.append(doc)
    return text


def get_vocabs(text: list, remove_stops=False):
    vocabs = []
    for doc in text:
        sents = nltk.sent_tokenize(doc)
        words = [nltk.word_tokenize(sent) for sent in sents]
        corr_words = []
        for sent in words:
            for word in sent:
                if word.lower()[0] in REQ_STARTING_CHARS:
                    if remove_stops and word.lower() in STOPWORDS:
                        continue
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

def validation_run(test_df, *vocabs):
    test_txt = get_text(test_df.loc[:, "text"])
    correct_labels = test_df.loc[:, "label"]
    test_vocabs = get_vocabs(test_txt, remove_stops=True)

    spam_voc, ham_voc = vocabs

    total_pred, correct_pred = 0, 0
    
    for idx, t_voc in enumerate(test_vocabs):
        total_pred += 1
        correct_label = correct_labels[idx] 
        total_score = 0.0

        for word in t_voc.keys():
            for voc in spam_voc:
                if word in voc:                    
                    total_score += voc[word]
            
            for voc in ham_voc:
                if word in voc:
                    total_score += voc[word]
                    
        if total_score > 0 and correct_label == 'ham':
            correct_pred += 1
        if total_score < 0 and correct_label == 'spam':
            correct_pred += 1
    
    return correct_pred / total_pred

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str, help="Provide the path to the dataset's raw CSV file",
        nargs="?")
    
    parser.add_argument(
        "-t",  "--train_size", default=0.8, type=float,
        help="(Optional Argument): Provide the train size for split. Default = 0.8",
        nargs="?", required=False)
    
    args = parser.parse_args()
    
    TRAIN_SIZE = args.train_size
    DATA_PATH = args.path

    train_df, test_df = load_data(DATA_PATH, TRAIN_SIZE)

    spam_docs = train_df.loc[np.where(train_df.loc[:, 'label'] == 'spam')[0], 'text']
    ham_docs = train_df.loc[np.where(train_df.loc[:, 'label'] == 'ham')[0], 'text']

    spam_text = get_text(spam_docs)
    ham_text = get_text(ham_docs)

    is_dataset_balanced(spam_text, ham_text)
    
    spam_vocabs = get_vocabs(spam_text, remove_stops=True)
    ham_vocabs = get_vocabs(ham_text, remove_stops=True)

    spam_vocabs = set_idf(spam_vocabs, len(spam_vocabs))
    ham_vocabs = set_idf(ham_vocabs, len(ham_vocabs))

    #show_details(spam_vocabs, ham_vocabs)

    # all spam vocabs need to be negative scores
    for vocab in spam_vocabs:
        for key in vocab.keys():
            vocab[key] *= -1
    
    # Now utilize test data (test_df)
    acc = validation_run(test_df, spam_vocabs, ham_vocabs)
    print(f"Accuracy of the TF-IDF based model : {acc * 100} %")