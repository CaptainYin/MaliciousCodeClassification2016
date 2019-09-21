import numpy as np
import re
import os
import itertools
from collections import Counter

files =['1.txt', '2.txt', '3.txt', '4.txt', '6.txt', '7.txt','8.txt', '9.txt']# 

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\?\?", " ", string)
    return string.strip()

def merge2(string):
    num1 = len(string)
    string2 = []
    if num1%2:
        string = string + ['##']
        num1 = num1 +1
    ssd = []
    for i in range(0, num1, 2):
        ssd = string[i] + string[i+1]
        string2.append(ssd)
    return string2


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    #define the data directory where the templates live
   
    data_dir = "./ready/"
    #store all of the class data in a list
    class_data = []
    label_list = []
    #default_list = []
    print "process load_data_and_labels"
    # Load data from files
    for i in files:
        print data_dir+i
        with open(data_dir+i, 'r') as f:
            examples = [line.strip() for line_num , line in enumerate(f) if line_num < 300]
        #examples = list(open(data_dir+i).readlines())  
        #if len(examples)>200:
        #    examples200 = examples[:200]
        #    examples = [s.strip() for s in examples200]
        #else: 
        #    examples = [s.strip() for s in examples]
        ##append these examples to the list of lists
        ##examples is multi line list
            class_data.append(examples)
        #make the label list as long as the numbe rof classes
        #default_list.append(0)
        print len(examples)
    print len(class_data) 
    #strip()  If omitted or None, the chars argument defaults to removing whitespace.
    # concat class examples
    counter = 0
    x_text = []
    for class_examples in class_data:
        #set the label
        temp_list = [0,0,0,0,0,0,0,0]#zerolist(class_data)
        temp_list[counter] = 1
        label_list.append(temp_list)

        if len(class_examples)>43: 
            if counter == 0:
                x_text = class_examples
            else:
                x_text = x_text + class_examples
        else:
            # num_p = 200 - len(class_examples)
            x_text = x_text + class_examples# + ['00'] * num_p 
        counter += 1
    
    #clean and split two byte
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split() for s in x_text]
    x_text = [merge2(s1) for s1 in x_text]

   # Generate labels
    print 'class_data'
    print len(class_data)
    final_labels = []
    counter = 0
    for class_examples in class_data:
        print label_list[counter]
        if len(class_examples)>43:
            final_labels.append([label_list[counter] for _ in class_data[counter]])
        else:
            
            final_labels.append([label_list[counter] for _ in class_data[counter]])#range(200)])
        counter += 1
    print 'final'
    print len(final_labels)
    y = np.concatenate(final_labels, 0)
    #this is no change,return para
    print 'x_text'
    print len(x_text)
    print 'y'
    print len(y)
    print y
    return [x_text, y]


def pad_sentences(sentences, padding_word="##"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    print "processing pad_sentences"
    sequence_length = 10000#max(len(x) for x in sentences)
    print sequence_length
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < 100000:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[-100000:]
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    print "processing build_vocab"
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    
    # If n is omitted or None, most_common() returns all elements in the 
    #counter. Elements with equal counts are ordered arbitrarily
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    print ("vocabulary size:{:d} in the data_input".format(len(vocabulary)))
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    print "processing build_input_data"
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]
def load_data_d2c():
    """
    read data from local 
    """
    #return [x, y, vocabulary, vocabulary_inv]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
