import numpy as np
import os
import build_vocabulary
import aclImdb.porter as porter

#BOW = Bag of Words
def _load_vocabulary(filename: str) -> list:
    voc = {}
    f = open(filename, encoding='utf-8')
    words = f.read()
    f.close()
    n = 0
    for word in words.split():
        voc[word] = n
        n += 1
    return voc

def _remove_punctuation(text: str) -> str:
    punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for punctuation in punct:
        text = text.replace(punctuation, ' ')
    return text

def _read_document(filename: str, voc, stop) -> list:
    f = open(filename, encoding='utf-8')
    text = f.read()
    f.close()

    bow = np.zeros(len(voc))
    text = _remove_punctuation(text)

    for word in text.split():
        if word.lower() in voc and word.lower() not in stop:
            bow[ voc[word.lower()] ] += 1

    #for word in text.split():
    #        if (porter.stem(word.lower()) in voc) and (porter.stem(word.lower()) not in stop):
    #            bow[ voc[ porter.stem(word.lower()) ] ] += 1

    return bow

def extract_from_dir(kind: str, dim_voc : int) -> np.ndarray:
    if kind == 'train' or kind == 'smalltrain':
        build_vocabulary.build(dim_voc)

    voc = _load_vocabulary('vocabulary.txt')
    stop = _load_vocabulary('aclImdb/stopwords.txt')

    documents = []
    labels = []

    for dir in os.listdir('aclImdb/'+kind+'/'):
        for filename in os.listdir('aclImdb/'+kind+'/'+dir):
            bow = _read_document('aclImdb/'+kind+'/'+dir+'/'+filename, voc, stop)
            documents.append(bow)
            labels.append(1 if dir == 'pos' else 0)

    X = np.array(documents)
    Y = np.array(labels)

    data = np.concatenate([X, Y[:, None]], 1)

    #np.savetxt('smalltrain.txt.gz', data)
    np.save(kind+'.npy', data)


    #print(data)