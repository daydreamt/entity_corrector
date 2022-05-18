import numpy as np
from typing import List
from pyxdameraulevenshtein import damerau_levenshtein_distance
from sklearn.neighbors import BallTree


class EntityCorrector:
    def __init__(self, entities: List[str], max_size=50, construct_balltree=False):
        self.max_size = max_size
        self.entities = [entity.lower() for entity in entities]
        self.n_entities = len(self.entities)

        text = ""
        for sentence in self.entities:
            text += sentence

        tokenized_text = list(text)
        self.token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}

        self.X = np.array([self.sentence_to_vector(x) for x in entities])
        # ENH: I believe it works, but there's a small chance that BT bugged due to the encoding:
        # levenshtein distance <>  0 if x == y, irrelevant if x <> 0
        # => Hence might need modified pivot construction
        self.construct_balltree = construct_balltree
        if self.construct_balltree:
            self.tree = BallTree(
                self.X, leaf_size=2, metric=damerau_levenshtein_distance
            )
        else:
            self.tree = None

    def sentence_to_vector(self, sentence):
        tokenized_text = list(sentence.lower()[: self.max_size])
        tokens = [
            self.token2idx[token] if token in self.token2idx else (self.max_size)
            for token in tokenized_text
        ]
        return np.pad(tokens, (0, self.max_size - len(tokens)))

    def get_nearest_within_linear(self, sentence, within=3):
        """ Runs in O(M*N), where N is the number of data and M is the max of the sentence length,
    and the maximum sentence length in the dataset"
    """
        encoding = self.sentence_to_vector(sentence)
        distances = []
        idxes = []
        for idx, x in enumerate(self.X):
            distance = damerau_levenshtein_distance(encoding, x)
            if distance <= within:
                distances.append(distance)
                idxes.append(idx)
        return [self.entities[idx] for idx in idxes]

    def get_nearest_(self, sentence, k=5):
        if not self.construct_balltree:
            raise (
                (
                    "Requires self.construct_balltree to be True, but current implementation bugged"
                )
            )
        k = min(k, self.n_entities)
        print(k)
        dist, ind = self.tree.query([self.sentence_to_vector(sentence)], k=k)
        dists = dist[0]
        idxes = ind[0]

        res = []
        for dist, idx in zip(list(dists), list(idxes)):
            res.append((dist, self.entities[idx]))
        return res

    def get_nearest_within(self, sentence, within=3):
        if not self.construct_balltree:
            raise (
                Exception(
                    "Requires self.construct_balltree to be True, but current implementation bugged"
                )
            )

        idxes = self.tree.query_radius([self.sentence_to_vector(sentence)], r=within)
        return [self.entities[idx] for idx in idxes[0]]

    def get_corrected_linear(self, sentence):
        """
      Due to a slight issue to how this damerau_levenshtein behaves due to the encoding 

       x
      array([ 1, 17, 10, 12, 17,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])

      y 
      array([ 1, 17, 10,  1, 12, 17,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])

      damerau_levenshtein_distance(x, y)
      2
      whereas
      damerau_levenshtein_distance("atlnta", "atlanta")
      1
      => Hence as a O(1) workaround, this default function
      searches for within=3 and then applies the levenshtein distance 
      again.
      )
      """
        candidates = self.get_nearest_within_linear(sentence, within=2)
        res = [x for x in candidates if damerau_levenshtein_distance(sentence, x) <= 1]
        return res

    def get_corrected_bt(self, sentence):
        if not self.construct_balltree:
            raise (
                Exception(
                    "Requires self.construct_balltree to be True, but current implementation bugged"
                )
            )
        candidates = self.get_nearest_within(sentence, within=2)
        res = [x for x in candidates if damerau_levenshtein_distance(sentence, x) <= 1]
        return res

    def get_corrected(self, sentence):
        if self.construct_balltree:
            return self.get_corrected_bt(sentence)
        else:
            return self.get_corrected_linear(sentence)
