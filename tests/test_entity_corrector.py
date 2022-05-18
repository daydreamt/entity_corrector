import pytest
from entity_corrector.entity_corrector import EntityCorrector, damerau_levenshtein_distance

data = [
    "the cat ate the bag",
    "the cbt ate the bag",
    "don't cry for me",
    "cat out of the bag",
    "the dog ate the cat",
    "the cat is out of the bag's bag",
    "the",
    "then",
    "my name is nikolaos",
]

ec = EntityCorrector(data, construct_balltree=True)


def test_tokenization():
    assert (
        damerau_levenshtein_distance(
            ec.sentence_to_vector("the cat ate the bag"),
            ec.sentence_to_vector("the cbt ate the bag"),
        )
        == 1
    )
    assert (
        damerau_levenshtein_distance(
            ec.sentence_to_vector("the cat ate the bag"),
            ec.sentence_to_vector("the cat ate the bag!"),
        )
        == 1
    )

    # Uknown tokens
    results = ec.get_nearest_within_linear("the cat ate the ba!", 1)
    n_results = len(results)
    assert (n_results) == 1

    results_bt = ec.get_nearest_within("the cat ate the ba!", 1)
    assert len(results_bt) == 1


def test_retrieval():
    assert len(ec.get_nearest_within_linear("the cbt ate the bag")) == 2


def test_retrieval_bt():
    assert len(ec.get_nearest_within("the cbt ate the bag")) == 2
