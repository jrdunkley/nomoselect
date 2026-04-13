"""Tests for nomoselect.tasks — task family constructors."""
import numpy as np
import pytest

from nomoselect.tasks import (
    TaskFamily,
    fisher_task_family,
    equal_weight_task_family,
    minority_emphasis_family,
    pairwise_task_family,
    custom_task_family,
)


@pytest.fixture
def two_class_data():
    """Simple 2-class, 3-dimensional whitened data."""
    class_means_w = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    class_sizes = np.array([30.0, 70.0])
    return class_means_w, class_sizes


@pytest.fixture
def three_class_data():
    """3-class, 4-dimensional whitened data with unequal sizes."""
    class_means_w = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    class_sizes = np.array([10.0, 50.0, 40.0])
    return class_means_w, class_sizes


class TestFisherTaskFamily:
    def test_returns_task_family(self, two_class_data):
        tf = fisher_task_family(*two_class_data)
        assert isinstance(tf, TaskFamily)

    def test_aggregate_is_single_element(self, two_class_data):
        tf = fisher_task_family(*two_class_data)
        assert len(tf.aggregate) == 1

    def test_aggregate_is_symmetric(self, two_class_data):
        tf = fisher_task_family(*two_class_data)
        agg = tf.aggregate[0]
        np.testing.assert_allclose(agg, agg.T, atol=1e-14)

    def test_weights_sum_to_one(self, two_class_data):
        tf = fisher_task_family(*two_class_data)
        np.testing.assert_allclose(tf.weights.sum(), 1.0)

    def test_weights_are_sample_proportions(self, two_class_data):
        class_means_w, class_sizes = two_class_data
        tf = fisher_task_family(class_means_w, class_sizes)
        np.testing.assert_allclose(tf.weights, [0.3, 0.7])

    def test_individual_count_matches_classes(self, three_class_data):
        tf = fisher_task_family(*three_class_data)
        assert len(tf.individual) == 3

    def test_aggregate_equals_sum_of_individual(self, three_class_data):
        tf = fisher_task_family(*three_class_data)
        reconstructed = sum(tf.individual)
        np.testing.assert_allclose(tf.aggregate[0], reconstructed, atol=1e-14)

    def test_labels_default(self, two_class_data):
        tf = fisher_task_family(*two_class_data)
        assert tf.labels == ["class_0", "class_1"]

    def test_labels_custom(self, two_class_data):
        tf = fisher_task_family(*two_class_data, class_labels=["A", "B"])
        assert tf.labels == ["A", "B"]


class TestEqualWeightTaskFamily:
    def test_weights_equal(self, three_class_data):
        tf = equal_weight_task_family(*three_class_data)
        np.testing.assert_allclose(tf.weights, [1/3, 1/3, 1/3])

    def test_different_from_fisher_with_unequal_sizes(self, three_class_data):
        tf_f = fisher_task_family(*three_class_data)
        tf_e = equal_weight_task_family(*three_class_data)
        # Should differ because class sizes are unequal
        assert not np.allclose(tf_f.aggregate[0], tf_e.aggregate[0])


class TestMinorityEmphasisFamily:
    def test_weights_favour_small_class(self, three_class_data):
        tf = minority_emphasis_family(*three_class_data)
        # Class 0 has size 10, should have largest weight
        assert tf.weights[0] > tf.weights[1]
        assert tf.weights[0] > tf.weights[2]

    def test_weights_sum_to_one(self, three_class_data):
        tf = minority_emphasis_family(*three_class_data)
        np.testing.assert_allclose(tf.weights.sum(), 1.0)


class TestPairwiseTaskFamily:
    def test_number_of_pairs(self, three_class_data):
        tf = pairwise_task_family(*three_class_data)
        # 3 choose 2 = 3
        assert len(tf.individual) == 3
        assert len(tf.labels) == 3

    def test_pair_labels(self, three_class_data):
        class_means_w, class_sizes = three_class_data
        tf = pairwise_task_family(class_means_w, class_sizes, ["A", "B", "C"])
        assert "A_vs_B" in tf.labels
        assert "A_vs_C" in tf.labels
        assert "B_vs_C" in tf.labels

    def test_aggregate_symmetric(self, three_class_data):
        tf = pairwise_task_family(*three_class_data)
        agg = tf.aggregate[0]
        np.testing.assert_allclose(agg, agg.T, atol=1e-14)


class TestCustomTaskFamily:
    def test_basic(self):
        d = 3
        tasks = [np.eye(d), np.diag([1.0, 0.0, 0.0])]
        tf = custom_task_family(tasks)
        assert len(tf.individual) == 2

    def test_custom_weights(self):
        d = 3
        tasks = [np.eye(d), np.diag([1.0, 0.0, 0.0])]
        tf = custom_task_family(tasks, weights=np.array([0.8, 0.2]))
        np.testing.assert_allclose(tf.weights, [0.8, 0.2])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            custom_task_family([])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            custom_task_family([np.eye(3), np.eye(4)])


class TestValidation:
    def test_negative_class_sizes(self, two_class_data):
        class_means_w, _ = two_class_data
        with pytest.raises(ValueError, match="positive"):
            fisher_task_family(class_means_w, np.array([-1.0, 5.0]))

    def test_wrong_shape(self):
        with pytest.raises(ValueError):
            fisher_task_family(np.array([1.0, 2.0]), np.array([5.0]))
