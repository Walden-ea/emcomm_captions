import tempfile
import textwrap
import os
from argparse import Namespace

from src.objects_game import yaml_config, train


def test_load_and_merge(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(textwrap.dedent("""
    train_samples: 123
    n_distractors: 7
    """))

    ns = yaml_config.load_yaml_config(cfg)
    assert isinstance(ns, Namespace)
    assert ns.train_samples == 123
    assert ns.n_distractors == 7

    # ensure merging with defaults works and overwrites values
    defaults = Namespace(train_samples=1000, n_distractors=3, sender_lr=0.1)
    merged = yaml_config.merge_with_defaults(defaults, ns)
    assert merged.train_samples == 123
    assert merged.n_distractors == 7
    assert merged.sender_lr == 0.1


def test_get_params_accepts_yaml(tmp_path, monkeypatch):
    # create dummy yaml that only specifies a couple of fields
    cfg = tmp_path / "cfg2.yaml"
    cfg.write_text(textwrap.dedent("""
    train_samples: 50000
    n_distractors: 5
    mode: gs
    vocab_size: 30
    """))

    # call get_params with a list containing the yaml path
    args = train.get_params([str(cfg)])
    assert args.train_samples == 50000
    assert args.n_distractors == 5
    assert args.vocab_size == 30
    assert args.mode == "gs"
    # some defaults should still be present
    assert hasattr(args, "sender_hidden")
    assert args.sender_hidden == 50
