import numpy as np
import gensim.downloader as dl


def load_the_models():
    model = dl.load("word2vec-google-news-300")
    return model


def get_dict_of_all_words_and_tags():
    with open("pos/pos/data/ass1-tagger-train", 'r') as f:
        words_and_tags = {}
        for line in f:
            for word_tag in line.split():
                word, tag = word_tag.rsplit('/', 1)
                if word in words_and_tags:
                    words_and_tags[word].add(tag)
                else:
                    words_and_tags[word] = {tag}
    return words_and_tags


def make_mean_tag_vectors(tags_vector):
    for tag, vectors in tags_vector.items():
        tags_vector[tag] = np.mean(vectors, axis=0)
    return tags_vector


def make_from_words_tags_words_vector(words_and_tags, model):
    tag_to_vectors = {}
    for word, tags in words_and_tags.items():
        try:
            vec = model[word]
            for tag in tags:
                if tag in tag_to_vectors:
                    tag_to_vectors[tag].append(vec)
                else:
                    tag_to_vectors[tag] = [vec]
        except KeyError:
            pass
    return tag_to_vectors


def return_tag_by_static_vector(word, model, tag_to_vectors):
    try:
        vector = model[word]
        max_sim = 0
        max_tag = ''
        for tag, vectors in tag_to_vectors.items():
            matrix = np.array(vectors)
            sim = matrix.dot(vector) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector))
            sim = np.max(sim)
            if sim > max_sim:
                max_sim = sim
                max_tag = tag
        return max_tag
    except KeyError:
        return None


def predict_by_common_occurrence(map_1, word_to_check):
    tag1 = None
    origin_word1 = map_1.get(word_to_check, None)
    if origin_word1 is not None:
        tag1 = map_1[word_to_check][0][0]

    if tag1 is not None:
        return tag1, False

    return "", True


def count_words_tagging():
    map_1 = {}
    with open("pos/pos/data/ass1-tagger-train", 'r') as f:
        line = f.readline()
        while line != '':
            # first split  by space
            line = line.split(' ')
            for word in line:
                word = word.split('/')
                word_to_insert = "/".join(word[:-1])
                the_tag = word[-1].rstrip('\n')


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

            line = f.readline()

    # sorting map1
    for k, v in map_1.items():
        v.sort(key=lambda a: a[1], reverse=True)

    return map_1

def return_the_accuracy(tag_to_vectors, model, map1):
    with open("POS_preds_2.txt", 'w') as result:
        with open("pos/pos/data/ass1-tagger-dev", 'r') as f:
            count = 0
            correct = 0
            for line in f:
                for word_tag in line.split():
                    word, tag_correct = word_tag.rsplit('/', 1)
                    w2v_pred, check = predict_by_common_occurrence(map1, word)
                    if check:
                        w2v_pred = return_tag_by_static_vector(word, model, tag_to_vectors)
                    if w2v_pred == tag_correct:
                        correct += 1
                    count += 1
                    result.write(f'{word}/{w2v_pred} ')
                    if w2v_pred != tag_correct:
                        print(f'word: {word} tag: {tag_correct}, predict-tag: {w2v_pred}')
                result.write('\n')
    print(f'num of words: {count}, correct words: {correct}')
    return correct / count


if __name__ == '__main__':
    word2vec = load_the_models()
    map1 = count_words_tagging()
    words_and_tags = get_dict_of_all_words_and_tags()
    tag_to_vectors = make_from_words_tags_words_vector(words_and_tags, word2vec)
    accuracy = return_the_accuracy(tag_to_vectors, word2vec, map1)
    print(accuracy)
