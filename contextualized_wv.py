from transformers import RobertaTokenizer, RobertaModel, pipeline
import numpy as np
from numpy.linalg import norm
import torch
import json
import linecache
from collections import Counter


def cos(v1, v2):
    return np.dot(v1, v2)/(norm(v1) * norm(v2))

def max_pooling(v):
    output_v = np.empty((128), dtype=float)
    for i in range(0, len(v), 6):
        local_v = [abs(v[j]) for j in range(i, i+6)]
        max_f = max(local_v)
        output_v[int(i / 6)] = max_f
    return output_v


feature_extractor_glob = pipeline('feature-extraction', model='roberta-base', torch_dtype=torch.float32, return_tensor=True, framework="pt")


def get_parts():
    word_parts = []
    with open("pos/pos/data/ass1-tagger-train") as tagged:
        for line in tagged:
            split_line = line.split(' ')
            for j, word_tag in enumerate(split_line):
                pair = word_tag.split('/')
                word = pair[0]
                tag = pair[-1]
                tag = tag.strip('\n')
                if tag not in word_parts:
                    word_parts.append(tag)
    return word_parts


def get_wtotag_alt():
    word_to_tag = {}
    with open("speech_parts.json") as p:
        parts = json.load(p)
    with open("pos/pos/data/ass1-tagger-train") as tagged:
        for i, line in enumerate(tagged):
            if i % 1000 == 0:
                print(f'getting w_t line {i}')
            if line is None:
                continue
            split_line = line.strip('\n')
            split_line = split_line.split(' ')
            sentence = [pair.split('/')[0] for pair in split_line]
            sentence = " ".join(sentence)
            sentence_features = feature_extractor_glob(sentence, return_tensor=True, framework="pt", requires_grad=False)
            sentence_features = torch.FloatTensor(sentence_features)
            for j, word_tag in enumerate(split_line):
                pair = word_tag.split('/')
                word = pair[0]
                tag = pair[-1]
                tag_feat = sentence_features[0][j + 1]
                if word_to_tag.get(word, False):
                    if word_to_tag.get(word).get(parts[tag], None) is not None:
                        word_to_tag[word][parts[tag]] += tag_feat
                    else:
                        word_to_tag[word][parts[tag]] = tag_feat
                else:
                    word_to_tag[word] = {parts[tag]: tag_feat}
    return word_to_tag


def get_wtotag():
    word_to_tag = {}
    with open("speech_parts.json") as p:
        parts = json.load(p)
    with open("pos/pos/data/ass1-tagger-train") as tagged:
        for i, line in enumerate(tagged):
            if i % 10000 == 0:
                print(f'getting w_t line {i}')
            if line is None:
                continue
            split_line = line.split(' ')
            sentence = [pair[0] for pair in split_line]
            sentence = " ".join(sentence)
            sentence_features = feature_extractor_glob(sentence)
            for j, word_tag in enumerate(split_line):
                pair = word_tag.split('/')
                word = pair[0]
                tag = pair[-1]
                tag_feat = sentence_features[0][j+1]
                tag_feat = max_pooling(tag_feat)
                if word_to_tag.get(word, False):
                    if word_to_tag.get(word).get(parts[tag], False):
                        word_tup = word_to_tag[word][parts[tag]]
                        word_to_tag[word][parts[tag]] = (word_tup[0] + 1, word_tup[1] + tag_feat, word_tup[2] + 1)
                    else:
                        word_to_tag[word][parts[tag]] = (1, tag_feat, 1)
                else:
                    word_to_tag[word] = {parts[tag]: (1, tag_feat, 1)}
    return word_to_tag


def write_dict(word_to_tag, filename):
    with open(filename, 'w') as fp:
        json.dump(word_to_tag, fp=fp, indent=4)



def get_parts_json():
    parts = get_parts()
    parts = {i: val for i, val in enumerate(parts)}
    revd = dict([reversed(i) for i in parts.items()])
    parts.update(revd)
    write_dict(parts, "speech_parts.json")


def write_tagcount():
    tag_count = {}
    with open("speech_parts.json") as p:
        parts = json.load(p)
    with open("pos/pos/data/ass1-tagger-train") as tagged:
        for i, line in enumerate(tagged):
            if i % 10000 == 0:
                print(f'getting tag count line {i}')
            if line is None:
                continue
            split_line = line.strip('\n')
            split_line = split_line.split(' ')
            for j, word_tag in enumerate(split_line):
                pair = word_tag.split('/')
                word = pair[0]
                tag = pair[-1]
                if tag_count.get(word, False):
                    if tag_count.get(word).get(parts[tag], False):
                        tag_count[word][parts[tag]] += 1
                    else:
                        tag_count[word][parts[tag]] = 1
                else:
                    tag_count[word] = {parts[tag]: 1}
    with open('tag_count.json', 'w') as out:
        json.dump(tag_count, out, indent=4)


def handle_no_word(word_f, t_v, parts):
    max_sim = (0, 0)
    word_f.reshape((768, 1))
    for tag, vec in t_v.items():
        # sim = cos(word_f, vec)

        vec_tensor = torch.FloatTensor(vec)
        sim = torch.matmul(vec_tensor, word_f).max()
        if sim > max_sim[1]:
            max_sim = (tag, sim)
    return max_sim[0]


def get_prediction(word, word_features, parts, tag_count, w_t, t_v):
    word_tag_feat = w_t.get(word, None)
    if tag_count is None or word_tag_feat is None:
        return handle_no_word(word_features, t_v, parts)

    max_tag = max(tag_count, key=tag_count.get)
    max_tag_val = tag_count[max_tag]
    sum = 0
    for k, v in tag_count.items():
        sum += int(v)
    if max_tag_val / sum > 0.85:
        return str(parts[str(max_tag)])

    max_sim = (0, "")
    word_features.reshape((768, 1))
    for tag, vec in t_v.items():
        vec_tensor = torch.FloatTensor(vec)
        sim = torch.matmul(vec_tensor, word_features).max()
        if sim.item() > max_sim[0]:
            max_sim = (sim, tag)


    # max_sim = (0, 0)
    # for tag, features in word_tag_feat.items():
    #     tag_features = np.array(features)
    #     sim = cos(tag_features, word_features)
    #     if sim > max_sim[0]:
    #         max_sim = (sim, tag)

    return parts[max_sim[1]]


def write_tag_vec():
    tag_vec = {}
    with open("speech_parts.json") as p:
        parts = json.load(p)
    with open("pos/pos/data/ass1-tagger-train") as tagged:
        for i, line in enumerate(tagged):
            if line is None:
                continue
            if i % 100 == 0:
                print(f'getting tag count line {i}')
                found_all_tags = True
                for k, v in parts.items():
                    if isinstance(v, str) and v not in tag_vec.keys():
                        found_all_tags = False
                if found_all_tags:
                    break
            split_line = line.split(' ')
            sentence = line.split(' ')
            sentence_str = (" ".join(sentence)).strip('\n')
            sentence_features = feature_extractor_glob(sentence_str, return_tensor=True, framework="pt",
                                                       requires_grad=False)
            sentence_features = torch.FloatTensor(sentence_features)
            for j, word_tag in enumerate(split_line):
                pair = word_tag.split('/')
                word = pair[0]
                tag = pair[-1]
                tag = tag.strip('\n')
                if tag_vec.get(tag, None) is None:
                    tag_vec[tag] = sentence_features[0][j + 1]
                else:
                    tag_vec[tag] = torch.vstack((tag_vec[tag], sentence_features[0][j+1]))
    with open('tag_vec.json', 'w') as out:
        for k_t, vec in tag_vec.items():
            tag_vec[k_t] = vec.tolist()
        json.dump(tag_vec, out, indent=4)


def get_dev_predict(w_t, t_v):
    with open('speech_parts.json') as f:
        parts = json.load(f)
    with open('tag_count.json', 'r') as data:
        tag_count = json.load(data)
    with open("pos/pos/data/ass1-tagger-dev-input") as dev:
        with open("dev_test_2.txt", 'w') as out:
            for j, line in enumerate(dev):
                if j % 100 == 0:
                    print(f'passed {j} iteration')
                sentence = line.split(' ')
                sentence_str = (" ".join(sentence)).strip('\n')
                sentence_features = feature_extractor_glob(sentence_str, return_tensor=True, framework="pt", requires_grad=False)
                sentence_features = torch.FloatTensor(sentence_features)
                for i, word in enumerate(sentence):
                    word = word.strip('\n')
                    prediction = get_prediction(word, sentence_features[0][i + 1], parts, tag_count.get(word, None), w_t, t_v)
                    out.write(f'{word}/{prediction} ')
                out.write('\n')


def test_dev():
    sum = 0
    correct = 0

    with open("dev_test.txt", 'r') as pred_dev, open("pos/pos/data/ass1-tagger-dev") as dev:
        for pred_line, dev_line in zip(pred_dev, dev):
            pred_line = pred_line.strip('\n')
            dev_line = dev_line.strip('\n')
            split_pred = pred_line.split(" ")[:-1]
            split_pred = [pair.split('/')[1] for pair in split_pred]
            split_dev = dev_line.split(" ")[:-1]
            split_dev = [pair.split('/')[1] for pair in split_dev]

            for p1, p2 in zip(split_pred, split_dev):
                if p1 == p2:
                    correct += 1
                sum += 1
    return correct/sum


def get_test_results(w_t, t_v):
    with open('speech_parts.json') as f:
        parts = json.load(f)
    with open('tag_count.json', 'r') as data:
        tag_count = json.load(data)
    with open("pos/pos/data/ass1-tagger-test-input") as test:
        with open("POS_preds_3.txt", 'w') as out:
            for j, line in enumerate(test):
                if j % 100 == 0:
                    print(f'tested {j} iteration')
                sentence = line.split(' ')
                sentence_str = (" ".join(sentence)).strip('\n')
                sentence_features = feature_extractor_glob(sentence_str, return_tensor=True, framework="pt",
                                                           requires_grad=False)
                sentence_features = torch.FloatTensor(sentence_features)
                for i, word in enumerate(sentence):
                    word = word.strip('\n')
                    prediction = get_prediction(word, sentence_features[0][i + 1], parts, tag_count.get(word, None),
                                                w_t, t_v)
                    prediction = prediction.strip('\n')
                    out.write(f'{word}/{prediction} ')
                out.write('\n')




def write_word_tags():
    w_t = get_wtotag_alt()
    for k, dict_w in w_t.items():
        for k_t, vec in dict_w.items():
            w_t[k][k_t] = vec.tolist()
    with open('word_to_tag.json', 'w') as dict:
        json.dump(w_t, dict, indent=1)


def load_wtotag():
    with open('word_to_tag.json', 'r') as data:
        w_t = json.load(data)
    return w_t


def load_tv():
    with open('tag_vec.json', 'r') as data:
        t_v = json.load(data)
    return t_v


def main():
    # write_word_tags()
    w_t = load_wtotag()
    t_v = load_tv()
    get_dev_predict(w_t, t_v)
    print("acc: ", str(test_dev()))
    # write_tag_vec()
    # write_tagcount()
    # get_test_results(w_t, t_v)


if __name__ == "__main__":
    main()
