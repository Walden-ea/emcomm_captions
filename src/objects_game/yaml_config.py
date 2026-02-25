"""Utility for reading arguments from a YAML configuration file.

The training scripts expect all of their options to be available as
attributes of an ``argparse.Namespace``.  The helpers defined here make
it easy to read a YAML file and produce a compatible namespace; they
also expose a small convenience function for merging the values from a
config file with a pre‑existing namespace (typically the defaults
produced by an argument parser).

Having this logic in a separate module keeps the core training code
clean and allows the YAML reading to be reused elsewhere (for example,
if other experiments want to be driven from a config file).
"""

from __future__ import annotations

import pathlib
from argparse import Namespace
from typing import Optional, Union

import yaml


def load_yaml_config(path: Union[str, pathlib.Path]) -> Namespace:
    """Load a YAML file and return an ``argparse.Namespace`` containing its
    top-level keys as attributes.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file containing configuration options.

    Returns
    -------
    Namespace
        Configuration values.  If the file is empty, an empty namespace is
        returned.
    """

    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML configuration file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ValueError("Top-level YAML document must be a mapping/dictionary")

    return Namespace(**data)


def merge_with_defaults(
    base: Namespace, override: Namespace, *, inplace: bool = False
) -> Namespace:
    """Return a new namespace where values from ``override`` take precedence.

    Parameters
    ----------
    base : Namespace
        Namespace containing default values.
    override : Namespace
        Namespace with values that should replace those in ``base``.
    inplace : bool, optional
        If True, ``base`` is modified in-place and returned.  Otherwise a
        shallow copy is made.
    """
    if inplace:
        target = base
    else:
        target = Namespace(**vars(base))

    for key, val in vars(override).items():
        setattr(target, key, val)
    return target
