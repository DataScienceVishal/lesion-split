"""The network, the baselines, and the scoring, on arrays built here.

None of this needs the 2.8 GB corpus. What it needs is a label distribution
shaped like HAM10000's, because most of the failure modes worth catching only
appear when one class is two thirds of the data.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lesionsplit import model as models
from lesionsplit.data import CLASSES, class_weights

# The real proportions, rounded. Using a balanced fixture would make every test
# below pass for the wrong reason, since the whole point is what imbalance does.
SHARES = (0.033, 0.051, 0.110, 0.011, 0.111, 0.669, 0.015)


def labels(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).choice(len(CLASSES), size=n, p=SHARES)


def test_the_network_maps_a_batch_of_images_to_one_score_per_class():
    out = models.SmallCNN()(torch.zeros(4, 3, 64, 64))
    assert out.shape == (4, len(CLASSES))


def test_the_first_block_strides_so_the_experiment_fits_in_ten_minutes():
    """Regression on a performance decision that is easy to undo by accident.

    Without the stride, one training step measured 456 ms against 155 ms with
    it, and the six-run experiment went from 34 minutes to 12. Somebody
    simplifying the loop back to a uniform stride would not see a test fail
    unless one asserts on it.
    """
    first = models.SmallCNN().features[0]
    assert first.stride == (2, 2)


def test_majority_class_scores_the_majority_share():
    train, test = labels(4000, 0), labels(1000, 1)
    got = models.majority_class(train, test)
    commonest = np.bincount(test, minlength=len(CLASSES)).max() / len(test)
    assert got["accuracy"] == pytest.approx(commonest, abs=1e-9)


def test_majority_class_has_terrible_balanced_accuracy():
    """The reason accuracy alone is not reported anywhere in this project.

    A constant predictor scores about two thirds on accuracy and one seventh of
    one class on balanced accuracy. A reader shown only the first number cannot
    tell that the model never emits six of the seven classes.
    """
    got = models.majority_class(labels(4000, 0), labels(1000, 1))
    assert got["accuracy"] > 0.6
    assert got["balanced_accuracy"] == pytest.approx(1 / len(CLASSES), abs=1e-9)


def test_stratified_chance_lands_near_one_over_seven_on_balanced_accuracy():
    got = models.stratified_chance(labels(8000, 0), labels(4000, 1), seed=0)
    assert 0.05 < got["balanced_accuracy"] < 0.25


def test_a_perfect_prediction_scores_one_on_every_metric():
    truth = labels(500)
    got = models.scores(truth, truth.copy())
    assert got == {"accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0}


def test_scores_reports_three_numbers_that_can_disagree():
    """A fixture where accuracy is high and balanced accuracy is not.

    If these three ever move together on every input, reporting all three is
    decoration. This is the case that shows they do not.
    """
    truth = np.array([5] * 90 + [0] * 10)
    predicted = np.full(100, 5)
    got = models.scores(truth, predicted)
    assert got["accuracy"] == pytest.approx(0.90)
    assert got["balanced_accuracy"] == pytest.approx(0.50)
    assert got["macro_f1"] < got["accuracy"]


def test_class_weights_are_largest_for_the_rarest_class():
    weights = class_weights(labels(10000))
    counts = np.bincount(labels(10000), minlength=len(CLASSES))
    assert weights.argmax().item() == counts.argmin()
    assert weights.argmin().item() == counts.argmax()


def test_class_weights_survive_a_class_that_never_appears():
    """A lesion split can hand an arm a training set missing a rare class.

    df and vasc are about one percent each, so with an unlucky seed one of them
    can miss the training side entirely. Dividing by that zero would give an
    infinite weight and the loss would become nan on the first batch.
    """
    present = np.array([0, 1, 2, 4, 5, 6] * 50)
    weights = class_weights(present)
    assert torch.isfinite(weights).all()
    assert len(weights) == len(CLASSES)
