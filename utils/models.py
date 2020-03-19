import pickle
from pathlib import Path

from typing import Any


def lazy_property(fn) -> Any:
    """
    Decorator that makes a property lazy-evaluated.
    :param fn:
    :return:
    """

    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


def binary_dump(path: Path, model: Any) -> None:
    """
    Dump model given in path given
    :param path:
    :param model:
    :return:
    """
    # If any parents doesn't exist, make it.
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(mode='wb+') as f:
        pickle.dump(model, f)


def load(path: Path) -> Any:
    """
    Load a model from path given
    :param path:
    :return:
    """
    with path.open(mode='rb') as f:
        return pickle.load(f)
