import math

import pytest

from cantrack_ai.embeddings import cosine_similarity, parse_embedding


class TestCosineSimilarity:
    def test_identical_vectors_are_1(self):
        assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_scale_does_not_matter(self):
        assert cosine_similarity([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_0(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors_are_minus_1(self):
        assert cosine_similarity([1, 2], [-1, -2]) == pytest.approx(-1.0)

    def test_known_value(self):
        assert cosine_similarity([1, 1], [1, 0]) == pytest.approx(1 / math.sqrt(2))

    @pytest.mark.parametrize(
        "a,b", [([], []), ([1, 2], [1, 2, 3]), ([0, 0], [1, 2]), ([1, 2], [0, 0])]
    )
    def test_degenerate_inputs_are_0_never_raise(self, a, b):
        assert cosine_similarity(a, b) == 0.0


class TestParseEmbedding:
    def test_list_passes_through_as_floats(self):
        assert parse_embedding([0.1, 0.2, 3]) == [0.1, 0.2, 3.0]

    def test_pgvector_string_is_parsed(self):
        # Supabase's REST layer returns a pgvector column as "[0.1,0.2,...]"
        assert parse_embedding("[0.1,0.2,0.3]") == [0.1, 0.2, 0.3]

    def test_pgvector_string_with_spaces(self):
        assert parse_embedding("[0.1, 0.2]") == [0.1, 0.2]

    @pytest.mark.parametrize(
        "value", [None, "garbage", "[1, 2", "{}", "[\"a\"]", ["a", "b"], {"a": 1}, 5, ""]
    )
    def test_anything_else_is_none(self, value):
        assert parse_embedding(value) is None

    def test_a_string_embedding_scores_like_a_list(self):
        as_list = [0.3, 0.4, 0.5]
        as_string = "[0.3,0.4,0.5]"
        query = [0.3, 0.4, 0.5]
        assert cosine_similarity(query, parse_embedding(as_string)) == pytest.approx(
            cosine_similarity(query, as_list)
        )
