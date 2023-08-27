# This directory contains data mining algorithms

- These are classical algorithms that are used on data, rather than using data to learn a machine learning model of some sort.

- These algorithms *may* build a very **crude model**, the genesis of which don't involve any kind of `learning` (in a sense we generally associate with machine learning models). For example:

    * [spam_filtering.py](https://www.github.com/ssd0247/nlp-engines/data-mining/spam_filtering.py) contains a <u>*unique dictionary for a single document(email)containing all the words from the vocabulary with their corresponding counts*</u>.

## ALGORITHMS:

### SPAM-FILTERING

> Premise : We have a dataset containing a total of `N` email subject & body in string datatype format (UTF-8). Each such text is also assigned a label of the form `SPAM` vs `NOT-SPAM`. First thing done is concatenate the subject with the body of the email, so that we have a single long piece of textual data.

1. Collect all emails (referred to as `doc` from now on) labeled `SPAM`.

2. Start with a single doc.

3. Find out the vocabulary of that doc. Use `set()` data structure in Python to get a collection of unique words in that doc.

4. Build a count dictionary (utilizing `dict() OR Counter() from 'collections' std module`) with keys(== words of the set built above) and values(== counts term-frequencies {TF} of these words in the current doc).

5. Do the same for all docs. (can be parallelized!)

6. Now, we have `N` dictionaries.

7. Now we use them in such a way that:

    For each key in each dictionary:
    
    - Divide the total number of docs (== N) by the number of dictionaries that have thay key (== n), getting -> N/n.

    - Take a natural logarithm of that, getting -> ln(N/n).

    - This gives us the inverse document frequency {IDF}.

    - Multiply value of the current key in the current document with this IDF, getting TF-IDF weight/score for the word (== key) in the vocab of the current doc.

8. Repeat steps (2) to (7) for `NOT-SPAM` category emails.

> Assign **negative** values to weights when dealing with SPAM docs and **positive** values to weights when dealing with NOT-SPAM docs.

> NOTE BELOW ->

Hypothesis : This algorithm shouldn't require typical preprocessing step involving `REMOVING STOPWORDS`. SPAM docs give low magnitude negative TF-IDF scores to STOPWORDS (like close to 0 from -ve side) and NOT-SPAM docs give low magnitude positive TF-IDF scores to STOPWORDS (like close to 0 from +ve side).

So when applying algorithm to a new email: that is, using the dictionaries formed --> WEIGHTS OF INDIVIDUAL STOPWORDS MORE OR LESS CANCELS EACH OTHER OUT!

Testing Hypothesis : Build vocabulary via two approaches
(1) Without removing STOPWORDS
(2) After removing STOPWORDS
Compare the accuracy of classifying new emails into SPAM/NOT-SPAM in both the above cases. If it's same upto a desired tolerance, then hypothesis holds, else reject.