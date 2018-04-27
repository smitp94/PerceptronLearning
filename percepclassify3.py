import json
import sys
import re
import numpy as np

file_read = 'data/vanillamodel.txt'
# file_read = sys.argv[1]
# file_read_average = 'data/averagedmodel.txt'

file_write = "data/percepoutput.txt"

unique_words = []
Weight_posnegV = np.zeros(len(unique_words), dtype=float)
Weight_TFV = np.zeros(len(unique_words), dtype=float)
Bias = np.zeros(2, dtype=float)

# Weight_posnegA = np.zeros(len(unique_words), dtype=float)
# Weight_TFA = np.zeros(len(unique_words), dtype=float)
# Bias_avg = np.zeros(2, dtype=float)


def read_param():
    global unique_words
    global Weight_posnegV
    global Weight_TFV
    global Bias
    fhv = open(file_read, encoding='utf8')

    all_dict_v = json.load(fhv)
    Weight_posnegV = all_dict_v[0]
    Weight_TFV = all_dict_v[1]
    Bias = all_dict_v[2]
    unique_words = all_dict_v[3]


def remove_punctuation_lower(contents):
    text = ' '.join(contents)
    regex = re.compile('[%s]' % re.escape("!\"#$%&()*+,-./:;<=>?@[\]^_{|}~"))
    text = regex.sub(' ', text)

    # for numbers
    regex = re.compile(r'\d+')
    text = regex.sub('*no*', text)

    text = text.lower()
    return text.split()


stopwords_punc = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "I", "I'd", "I'll", "I'm", "I've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
stopwords = remove_punctuation_lower(stopwords_punc)


def is_stopword(word):
    global stopwords
    stop_wo_punc = remove_punctuation_lower(stopwords)
    if word in set(stop_wo_punc):
        return True
    else:
        return False


def classify():
    global unique_words
    global Weight_posnegV
    global Weight_TFV
    global Bias
    global stopwords

    answer = {}

    with open("data/dev-text.txt", encoding='utf8') as f:
    # with open(sys.argv[2], encoding='utf8') as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    for line in content:
        contents = line.split()
        if contents[0] not in answer:
            id = contents[0]
            answer[id] = {}
            answer[id]["pos_neg"] = ""
            answer[id]["true_fake"] = ""

        # print(contents[1:])
        x_vector = np.zeros(len(unique_words), dtype=int)
        sent = remove_punctuation_lower(contents[1:])
        for w in sent:
            if w in unique_words: # change to fn is_stopword(w)
                index = unique_words.index(w)
                x_vector[index] += 1

            y_posneg = np.sum(np.multiply(Weight_posnegV, x_vector)) + Bias[0]
            y_tf = np.sum(np.multiply(Weight_TFV, x_vector)) + Bias[1]

            if y_posneg > 0:
                answer[id]["pos_neg"] = "Pos"
            else:
                answer[id]["pos_neg"] = "Neg"

            if y_tf > 0:
                answer[id]["true_fake"] = "True"
            else:
                answer[id]["true_fake"] = "Fake"

    write_file(answer)


def write_file(answer):
    fh = open(file_write, 'w', encoding='utf8')
    flag = 0
    for k in answer.keys():
        if flag == 0:
            flag = 1
            fh.write(k)
        else:
            fh.write("\n"+k)
        fh.write(" " + answer[k]["true_fake"])
        fh.write(" " + answer[k]["pos_neg"])
    fh.close()


read_param()
classify()