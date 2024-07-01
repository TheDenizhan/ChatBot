import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer




def tokenize(sentence):
    """
        Cümleyi kelimeler/dizeler haline ayır
    Bir dize bir kelime, noktalama işareti veya sayı olabilir.
        """
    return nltk.word_tokenize(sentence)


stemmer = PorterStemmer()
def stem(word):
    """
    Kök bulma = kelimenin kök halini bulmak
    """
    return stemmer.stem(word.lower())




def bag_of_words(tokenized_sentence, words):
    """
    Kelimelerin bulunduğu bir dizi döndür: cümlede var olan her
    bilinen kelime için 1, aksi takdirde 0
    """

    sentence_words = [stem(word) for word in tokenized_sentence]

    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag
