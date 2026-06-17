# tests/test_explain.py

"""Minimal smoke tests for SHAP output shape and sort order."""

from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def feature_cols():
    return ["EXT_SOURCE_2", "CREDIT_INCOME_RATIO", "AGE_YEARS"]


@pytest.fixture
def mock_explainer():
    explainer = MagicMock()
    shap_row = np.array([0.3, -0.2, 0.1])
    explainer.shap_values.return_value = [-shap_row.reshape(1, -1), shap_row.reshape(1, -1)]
    return explainer


def test_returns_correct_count(mock_explainer, feature_cols):
    from src.models.explain import explain_single

    result = explain_single(mock_explainer, np.random.rand(1, 3), feature_cols, n_top=2)
    assert len(result) == 2


def test_sorted_by_absolute_value(mock_explainer, feature_cols):
    from src.models.explain import explain_single

    result = explain_single(mock_explainer, np.random.rand(1, 3), feature_cols, n_top=3)
    abs_vals = [abs(r["shap_value"]) for r in result]
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_plot_waterfall_creates_file(tmp_path, mock_explainer, feature_cols):
    from src.models.explain import plot_waterfall

    mock_explainer.expected_value = [0.5, 0.08]
    output = tmp_path / "waterfall.png"
    plot_waterfall(mock_explainer, np.random.rand(1, 3), feature_cols, output)
    assert output.exists()
