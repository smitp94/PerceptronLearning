import json
import sys
import re
import numpy as np

file_write = 'data/nbmodel.txt'

records = []
unique_words = set()
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


def nbmodel_write(words, len_unique):
    global Prior_Totals
    global Totals
    f = open(file_write, 'w', encoding='utf8')
    c = json.loads("[{0},{1},{2},{3}]".format(json.dumps(Prior_Totals), json.dumps(Totals), json.dumps(words), json.dumps(len_unique)))
    f.write(json.dumps(c))


def percept():
    global records
    global unique_words
    all_words = []

    # For getting all unique words
    for r in records:
        all_words.extend(r.text)
    unique_words = set(all_words)
    unique_words = [w for w in unique_words if w not in stopwords] # To remove stopwords
    # print(unique_words)
    # Initializing
    w = np.zeros(len(unique_words), dtype=float)

    b = np.zeros(2,dtype=float)
    # print(b)
    for iter in range(30):
        for feature in range(unique_words):
            # activation = w[feature]*x + b[0]
            pass



read_file()
percept()
