from collections import defaultdict
import os, sys
import math

def train_naive_bayes(docs, classes, method):
    num_documents = len(docs)

    # Class count and build vocab from all docs
    classes_count = defaultdict(int)
    vocab = set()
    big_doc = defaultdict(list)  # class: list of all words associated with class (contains duplicates)
    for doc in docs:
        classes_count[doc[0]] += 1
        vocab.update(doc)
        if(method == 'multinomial'):
            big_doc[doc[0]] += (doc[1:])
            print(doc)
        elif(method == "binary"):
            new_doc = [*set(doc[1:])]
            big_doc[doc[0]] += new_doc  # Removes duplicates; main difference in binary naive bayes
            print("Old doc: ", doc[1:])
            print("New doc: ", new_doc)
    vocab -= set(classes)   # Remove class names from vocab

    # Calculate prior: P(c) for each class
    log_prior = defaultdict(float)
    for c in classes:
        log_prior[c] = math.log(classes_count[c]/num_documents)

    # Calculate P(w|c) for each word in vocab
    log_likelihood = defaultdict(float)
    for w in vocab:
        for c in classes:
            count = big_doc[c].count(w)
            # Without smoothing: likelihood[(w,c)] = count/len(big_doc[c])
            # Add-one smoothing
            log_likelihood[(w,c)] = math.log((count + 1)/(len(big_doc[c]) + len(vocab)))  # denominator: total words in class + total words in vocab

    return log_likelihood, log_prior, vocab

def test_naive_bayes(log_likelihood, log_prior, vocab, classes, test_doc, method):
    if method == 'binary':  # Remove duplicate words in test doc
        test_doc = [*set(test_doc)]
        print("New doc: ", test_doc)

    sum = {}
    for c in classes:
        sum[c] = log_prior[c]
        for word in test_doc:
            if word in vocab:   # ignores words in test document not in vocabulary from training
                sum[c] += log_likelihood[word, c]

    print(sum)
    max_class = max(sum, key=sum.get)
    return (max_class, sum[max_class])

def main():
    print('\n')

    script_dir = os.path.dirname(__file__)
    f_train = open(os.path.join(script_dir, 'data/train/train2.txt'), "r")
    f_test = open(os.path.join(script_dir, "data/test/test2.txt"), "r")
    classes = ['pos', 'neg']
    method = "binary"  # multinomial/binary

    # Returns each line into a list
    docs = f_train.readlines()
    test_docs = f_test.readlines()

    # Remove newline character
    for i, doc in enumerate(docs):
        docs[i] = doc.strip().split()

    for i, doc in enumerate(test_docs):
        test_docs[i] = doc.strip().split()

    # Handle training data
    log_likelihood, log_prior, vocab = train_naive_bayes(docs, classes, method)
    
    print('\nTest Results:')
    # Handle testing data
    count = 0
    for td in test_docs:
        count += 1
        print(("Doc {}: {}").format(count, td))
        result = test_naive_bayes(log_likelihood, log_prior, vocab, classes, td, method)
        print(result)

    print('\n')

if __name__ == "__main__":
    main()