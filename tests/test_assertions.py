import pytest

# content of test_sample.py
def func(x):
    return x + 1


def test_answer():
    assert func(4) == 5


def test_simple():
    with pytest.raises(ValueError, match="must be 0 or None"):
        raise ValueError("value must be 0 or None")
    with pytest.raises(ValueError, match=r"must be \d+$"):
        raise ValueError("value must be 42")
    # with pytest.raises(ValueError, match="some wrong text"):
    #     raise ValueError("value must be 42")
