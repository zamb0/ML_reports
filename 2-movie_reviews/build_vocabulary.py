import collections
import os

import aclImdb.porter as porter

def _remove_punctuation(text: str) -> str:
    punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for punctuation in punct:
        text = text.replace(punctuation, ' ')
    return text

def _read_document(filename: str) -> list:
    f = open(filename, encoding='utf-8')
    s = open('aclImdb/stopwords.txt', encoding='utf-8')
    text = f.read()
    stop = s.read()
    f.close()
    s.close()

    text = _remove_punctuation(text)

    words = []
    for word in text.split():
        if len(word)>2 and word.lower() not in stop:
            words.append(word.lower())
    
    return words

def _write_vocabulary(vocabulary: collections.Counter(), filename: str, n: int) -> None:
    f = open(filename, 'w', encoding='utf-8')
    for word, count in vocabulary.most_common(n):
        f.write(word + '\n')
    f.close()


def build(n_words: int):
   
    voc  = collections.Counter()
   
    for dir in os.listdir('aclImdb/smalltrain/'):
        for filename in os.listdir('aclImdb/smalltrain/'+dir):
            words = _read_document('aclImdb/smalltrain/'+dir+'/'+filename)
            voc.update(words)
            #porter_words = []
            #for word in words:
            #    porter_words.append(porter.stem(word))
            #    
            #voc.update(porter_words)

    print('Vocabulary size:', len(voc))
    _write_vocabulary(voc, 'vocabulary.txt', n_words)