import re

import gensim.downloader as dl
import gensim
import numpy
from matplotlib import pyplot as plt
from sklearn import decomposition
import string


def remove_punctuation(line):
    # Use the string.punctuation constant to get a string containing all ASCII punctuation characters
    punctuation = string.punctuation

    # Iterate through the line and remove any characters that are in the punctuation string
    no_punct = ""
    perv = ""
    for char in line:
        if char not in punctuation:
            no_punct += char
        elif char == "/" and perv not in punctuation:
            no_punct += char
        perv = char


    # Return the modified string
    return no_punct


def make_list_of_words_from_train_file():
    words_list_and_tag = []
    words_list = []
    with open("pos/pos/data/ass1-tagger-train", "r") as file:
        for line in file:
            words_in_line = []
            if line != "\n":
                no_punc = remove_punctuation(line)
                words_in_line.append(no_punc.split())
                tuple_words = [tuple(re.split(r"/", s) for s in words_in_line)]
                words_list_and_tag.append(tuple_words)
                words_list.append(word[0] for word in tuple_words)
    return words_list, words_list_and_tag

def load_the_models():
    model = dl.load("word2vec-google-news-300")
    return model

def print_similar_vector(sentences,model):
    for sentence in sentences:
        similar_vectors = model.most_similar(positive=[sentence[:1]], topn=5)
        print(sentence)
        print("***************************************")
        print(similar_vectors)
        break

def static_word_vector():
    w2v_model = load_the_models()
    print("finish loading the model")
    list_of_words = make_list_of_words_from_train_file()
    print("finish making the list of words")
    print_similar_vector(list_of_words,w2v_model)
    print("finish printing the similar vectors")


if __name__ == '__main__':
    static_word_vector()