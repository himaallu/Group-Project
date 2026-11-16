"""Shared pytest fixtures: paths to the fake data and the joined fake table."""

from pathlib import Path

import pytest

from src.eval.loaders import join_gold, load_jsonl, load_preds

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_gold():
    """Fake gold records (10 originals + 17 twins) in the shared record format."""
    return load_jsonl(FIXTURES / "fake_gold.jsonl")


@pytest.fixture
def fake_preds():
    """Fake predictions for every fake gold record."""
    return load_preds(FIXTURES / "fake_preds.jsonl")


@pytest.fixture
def fake_df(fake_preds, fake_gold):
    """Joined table (predictions + gold) used by the metric tests."""
    df, missing = join_gold(fake_preds, fake_gold)
    assert missing == []
    return df
