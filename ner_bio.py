import sys
import codecs
import one_No_word_vectors
from collections import Counter


def read_data(fname):
    for line in codecs.open(fname):
        line = line.strip().split()
        tagged = [x.rsplit("/", 1) for x in line]
        yield tagged


def tag_each_word(f_train_name):
    map_1 = {}
    for line in read_data(f_train_name):
        for word in line:
            origin_word = map_1.get(word[0], None)
            word_to_insert = word[0]
            the_tag = (word[1])
            if origin_word == None:
                map_1[word_to_insert] = [(the_tag, 1)]
            else:
                bool_exist = False
                for i, tag in enumerate(map_1[word_to_insert]):
                    if tag[0] == the_tag:
                        map_1[word_to_insert][i] = (map_1[word_to_insert][i][0], map_1[word_to_insert][i][1] + 1)
                        bool_exist = True
                if not bool_exist:
                    map_1[word_to_insert].append((the_tag, 1))

    # MAYBE REMOVE IT
    # sorting map1
    for k, v in map_1.items():
        v.sort(key=lambda a: a[1], reverse=True)

    return map_1


def predict_ner_word(crop_words, tag_word_before, tag_word_after, tag):
    VB_list = ["VB", "VBD", "VBN", "VBG", "VBP", "VBZ", "POS"]
    LOC_list = ["IN", "TO"]
    ORG_list = ["DT"]
    MISC_list = ["NN", "NNP"]
    if len(tag) != 0:
        counts = Counter(tag)
        # print(counts)
        most_common = counts.most_common(1)[0][0]
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_counts) < 2:
            return "".join("/" + str(most_common) + " ")
        else:
            second_most_common = sorted_counts[1][0]
            print(most_common + " and " + second_most_common)
            if most_common != "/O":
                return "".join("/" + str(most_common) + " ")
            else:
                return "".join("/" + str(second_most_common) + " ")

    else:
        if tag_word_after in VB_list:
            return "".join("/I-PER ")
        elif tag_word_before in LOC_list:
            return "".join("/I-LOC ")
        elif tag_word_before in ORG_list:
            return "".join("/I-ORG ")
        if tag_word_after in MISC_list:
            return "".join("/I-MISC ")
    return "".join("/O ")


def create_file_BIO(map_train, map_1, map_2):
    count_false, count_true = 0, 0
    word_before_I = None
    word_after_I = None
    # means that we found an I word that need specific tag
    start_crop_I = False
    crop_words = []
    tag = []
    to_write = ""
    f_test = "ner/ner/test.blind"
    with open(f_test, 'r') as f:
        line = f.readline()
        while line != '':
            tag.clear()
            crop_words.clear()
            # first split by space
            line = line.split(' ')
            for word in line:
                if not word[0].isupper():
                    if not start_crop_I:
                        word_before_I = word
                        if word[-1] == "\n":
                            to_write += word[:-1] + "/O " + "\n"
                        else:
                            to_write += word + "/O "
                    else:
                        word_after_I = word
                        start_crop_I = False
                        # Needs to predict based on the word before and after prediction
                        tag_word_before = one_No_word_vectors.predict_word_tag(map_1, map_2, word_before_I, "")
                        if len(crop_words) != 0:
                            tag_word_after = one_No_word_vectors.predict_word_tag(map_1, map_2, word_after_I,
                                                                                  crop_words[-1])
                        ner_word_prediction = predict_ner_word(crop_words, tag_word_before, tag_word_after, tag)
                        for w in crop_words:
                            if w[-1] == "\n":
                                to_write += w[:-1] + ner_word_prediction + "\n"
                            else:
                                to_write += w + ner_word_prediction

                        if word[-1] == "\n":
                            to_write += word[:-1] + "/O " + "\n"
                        else:
                            to_write += word + "/O "
                        word_after_I = ""
                        word_before_I = ""
                        tag.clear()
                        crop_words.clear()
                else:
                    start_crop_I = True
                    origin_word = map_train.get(word, None)
                    if origin_word is not None:
                        crop_words.append(word)
                        tag.append(map_train[word][0][0])
                    else:
                        crop_words.append(word)
            line = f.readline()

            # checks if the sentence ends with upper case letter word
            if start_crop_I:
                word_after_I = ""
                start_crop_I = False
                # Needs to predict based on the word before and after prediction
                tag_word_before = one_No_word_vectors.predict_word_tag(map_1, map_2, word_before_I, "")
                if len(crop_words) != 0:
                    tag_word_after = one_No_word_vectors.predict_word_tag(map_1, map_2, word_after_I,
                                                                          crop_words[-1])
                ner_word_prediction = predict_ner_word(crop_words, tag_word_before, tag_word_after, tag)
                for w in crop_words:
                    if w[-1] == "\n":
                        to_write += w[:-1] + ner_word_prediction + "\n"
                    else:
                        to_write += w + ner_word_prediction



    with open("predicted_file.txt", 'w') as f:
        f.write(to_write)

    # calculate and return the accuracy
    return "worked?"


if __name__ == '__main__':
    f_train = "ner/ner/train"
    map_train = tag_each_word(f_train)
    print(map_train)

    map_1, map_2 = one_No_word_vectors.count_words_tagging()
    print(create_file_BIO(map_train, map_1, map_2))
