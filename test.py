import re
import string


def f():
    text = "123 this 34 is how we do 256-580000009 hi"
    regex = re.compile('[%s]' % re.escape("!\"#$%&()*+,-./:;<=>?@[\]^_{|}~"))
    text = regex.sub(' ', text)
    regex = re.compile(r'\d+')
    text = regex.sub('*no*', text)

    print(text.split())


f()
