import json
import sys
import re
import numpy as np

file_write_vanilla = 'data/vanillamodel.txt'
file_write_average = 'data/averagedmodel.txt'

records = []
unique_words = []
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "I", "I'd", "I'll", "I'm", "I've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
# Totals = {}


class Record:
    def __init__(self, id, t_f, pos_neg, text):
        self.id = id
        self.t_f = t_f
        self.pos_neg = pos_neg
        self.text = text


def remove_punctuation_lower(contents):
    text = ' '.join(contents)
    regex = re.compile('[%s]' % re.escape("!\"#$%&()*+,-./:;<=>?@[\]^_{|}~"))
    text = regex.sub(' ', text)
    text = text.lower()
    return text.split()


def read_file():
    global records
    with open("data/train-labeled.txt", encoding='utf8') as f:
    # with open(sys.argv[1], encoding='utf8') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for line in content:
        contents = line.split()
        id = contents[0]
        t_f = contents[1]
        pos_neg = contents[2]

        # For Punctuation Removal
        text = remove_punctuation_lower(contents[3::])

        r = Record(id, t_f, pos_neg, text)
        records.append(r)


def is_stopword(word):
    global stopwords
    stop_wo_punc = remove_punctuation_lower(stopwords)
    if word in set(stop_wo_punc):
        return True
    else:
        return False


def nbmodel_write(Weight_posnegV, Weight_TFV, Bias, Weight_posnegA, Weight_TFA, Bias_avg, unique_words):
    fv = open(file_write_vanilla, 'w', encoding='utf8')
    fa = open(file_write_average, 'w', encoding='utf8')
    cv = json.loads("[{0},{1},{2},{3}]".format(json.dumps(Weight_posnegV), json.dumps(Weight_TFV), json.dumps(unique_words), json.dumps(Bias)))
    ca = json.loads("[{0},{1},{2},{3}]".format(json.dumps(Weight_posnegA), json.dumps(Weight_TFA), json.dumps(unique_words), json.dumps(Bias_avg)))
    fv.write(json.dumps(cv))
    fa.write(json.dumps(ca))


def percept():
    global records
    global unique_words
    global stopwords

    # For getting all unique words
    for r in records:
        for w in r.text:
            if w not in unique_words and w not in stopwords:
                unique_words.append(w)

    # unique_words = [w for w in unique_words if w not in stopwords] # To remove stopwords

    # Initializing
    # Vanilla
    Weight_posnegV = np.zeros(len(unique_words), dtype=float)
    Weight_TFV = np.zeros(len(unique_words), dtype=float)
    Bias = np.zeros(2, dtype=float)

    # Average
    Weight_posnegA = np.zeros(len(unique_words), dtype=float)
    Weight_TFA = np.zeros(len(unique_words), dtype=float)
    Bias_avg = np.zeros(2, dtype=float)
    cache_posnegA = np.zeros(len(unique_words), dtype=float)
    cache_TFA = np.zeros(len(unique_words), dtype=float)
    cache_b = np.zeros(2, dtype=float)

    x_vector = np.zeros(len(unique_words), dtype=int)

    for itr in range(30):
        counter = 0
        for doc in records:
            if doc.pos_neg == "Pos":
                y = 1
            else:
                y = -1
            for w in doc.text:
                if w not in stopwords:
                    index = unique_words.index(w)
                    x_vector[index] += 1
            # print(x_vector)

            activation = np.sum(np.multiply(Weight_posnegV, x_vector)) + Bias[0]

            if y*activation <= 0:
                Weight_posnegV = Weight_posnegV + (y * x_vector)
                cache_posnegA = cache_posnegA + (y * counter * x_vector)
                Bias[0] += y
                cache_b[0] += (y * counter)

            if doc.t_f == "True":
                y = 1
            else:
                y = -1
            for w in doc.text:
                if w not in stopwords:
                    index = unique_words.index(w)
                    x_vector[index] += 1

            activation = np.sum(np.multiply(Weight_TFV, x_vector)) + Bias[1]

            if y*activation <= 0:
                Weight_TFV = Weight_TFV + (y*x_vector)
                cache_TFA = cache_TFA + (y * counter * x_vector)
                Bias[1] += y
                cache_b[1] += (y * counter)

            counter += 1
            x_vector = np.zeros(len(unique_words), dtype=int)

    Weight_posnegA -= (cache_posnegA/counter)
    Weight_TFA -= (cache_TFA/counter)
    Bias_avg -= (Bias/counter)

    nbmodel_write(Weight_posnegV, Weight_TFV, Bias, Weight_posnegA, Weight_TFA, Bias_avg, unique_words)


read_file()
percept()
