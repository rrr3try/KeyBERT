from pprint import pprint

import nltk
import numpy as np
from scipy.cluster import hierarchy

from keybert import KeyBERT


def read_csv(path):
    with open(path) as file:
        data = map(lambda string: string.split(';', maxsplit=3), file.readlines())
    return list(data)


class Extractor(KeyBERT):
    def __init__(self, *args, **kwargs):
        if len(args) > 0 or kwargs.get('model') is None:
            super().__init__(model='distilbert-base-nli-stsb-mean-tokens')
        else:
            super().__init__(*args, **kwargs)

    def cluster(self, texts, threshold=.9, n_gram_range=(1, 3), use_mmr=True, diversity=0.73, **kwargs):
        all_keywords = self.simple_extract_keywords(
            texts, n_gram_range=n_gram_range,
            use_mmr=use_mmr, diversity=diversity, **kwargs
        )
        short = [' '.join(keywords) for keywords in all_keywords]

        embeddings = self.model.encode(short)

        linkage = hierarchy.linkage(embeddings, method='average', metric='cosine')
        clusters = hierarchy.fcluster(linkage, threshold, criterion='distance')

        return clusters, all_keywords

    def simple_extract_keywords(self, texts, **kwargs):
        all_keywords = []
        for text in texts:
            keywords = self.extract_keywords(
                text,
                top_n=int(len(nltk.word_tokenize(text)) ** 0.45),
                nr_candidates=len(nltk.word_tokenize(text)),
                **kwargs,
            )
            all_keywords.append(keywords)

        return all_keywords

    @staticmethod
    def concatenate_texts_by_cluster(texts, clusters):
        concatenated = {}

        for index, cluster_num in enumerate(clusters):
            if concatenated.get(cluster_num) is None:
                concatenated[cluster_num] = []
            else:
                concatenated[cluster_num].append(texts[index])

        return concatenated


def main():
    """ Example of using clustering with KeyBert """

    d = read_csv("/home/retry/code/KeyBERT/src/Запросы к Textman (1).csv")
    texts = list(set([t[0] for t in d]))
    pprint(texts, width=300)
    print(len(texts))

    extractor = Extractor()

    for thresh in np.linspace(0.3, 0.99, 15):
        print(thresh, "=" * 96)
        clusters, keywords = extractor.cluster(texts, thresh)
        print(clusters)
        pprint(keywords)
        print(clusters.max(), "=" * 100)


def extract():
    d = read_csv("/home/retry/code/KeyBERT/src/Запросы к Textman (1).csv")
    texts = list(set([t[0] for t in d]))
    pprint(texts, width=300)
    print(len(texts))

    extractor = Extractor()
    all_keywords = extractor.simple_extract_keywords(
        texts, n_gram_range=(1, 3), use_mmr=True, diversity=0.73
    )
    pprint(all_keywords)


if __name__ == '__main__':
    main()
    extract()
