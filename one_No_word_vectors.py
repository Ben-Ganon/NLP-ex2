from collections import Counter


def count_words_tagging():
    map_1 = {}
    map_2 = {}
    tryee = ""
    tryee2 = ""
    with open("pos/pos/data/ass1-tagger-train", 'r') as f:
        line = f.readline()
        while line != '':
            word_map_2 = None
            # first split  by space
            line = line.split(' ')
            for word in line:
                word = word.split('/')

                word_to_insert = "/".join(word[:-1])
                the_tag = word[-1]

                # setting up the 2 map - one word after
                if word_map_2 is not None:
                    origin_word = map_2.get(word_map_2, None)
                    if origin_word == None:
                        map_2[word_map_2] = [the_tag]
                    else:
                        map_2[word_map_2].append(the_tag)

                # check if the word is in the map already
                origin_word = map_1.get(word_to_insert, None)
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

                word_map_2 = word_to_insert
            line = f.readline()

    # sorting map1
    for k, v in map_1.items():
        v.sort(key=lambda a: a[1], reverse=True)

    map2_to_test = {}
    # remove dups map2
    for k, v in map_2.items():
        counts = dict(Counter(v))
        dups_list = {key: value for key, value in counts.items()}
        map2_to_test[k] = dups_list

    for k, v in map2_to_test.items():
        sortedDict = sorted(v.items(), key=lambda x: x[1], reverse=True)
        the_max_tag = sortedDict[0][0]
        map_2[k] = the_max_tag

    return map_1, map_2


def predict_word_tag(map_1, map_2, word_to_check, word_map_2):
    tag1, tag2 = None, None
    origin_word1 = map_1.get(word_to_check, None)
    if origin_word1 is not None:
        tag1 = map_1[word_to_check][0][0]

    origin_word2 = map_2.get(word_map_2, None)
    if origin_word2 is not None:
        tag2 = map_2[word_map_2][0][0]

    if tag1 == tag2:
        return tag1

    if tag1 is not None:
        return tag1

    if tag2 is not None:
        return tag2

    if word_to_check[0].isupper():
        return "NNP"

    return "NN"


def check_accuracy(map_1, map_2):
    count_false, count_true = 0, 0
    with open("pos/pos/data/ass1-tagger-dev", 'r') as f:
        line = f.readline()
        while line != '':
            word_map_2 = None
            # first split  by space
            line = line.split(' ')
            for word in line:
                word = word.split('/')

                word_to_check = "/".join(word[:-1])
                the_tag = word[-1]

                # check if the word is in the map already
                prediction = predict_word_tag(map_1, map_2, word_to_check, word_map_2)
                if prediction == the_tag:
                    count_true += 1
                else:
                    count_false += 1
                word_map_2 = word_to_check

            line = f.readline()

    # calculate and return the accuracy
    return ((count_true) / (count_true + count_false))


if __name__ == "__main__":
    map_1, map_2 = count_words_tagging()
    # print(map_1)
    # print(map_2)

    accuracy = check_accuracy(map_1, map_2)
    print("The accuracy is " + str(accuracy * 100) + "%")
