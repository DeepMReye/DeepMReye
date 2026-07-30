"""Checkpoints must carry their own architecture.

``train_jepa.py`` and ``eval_probe.py`` have separate argument parsers with
separate defaults, and they have already disagreed once (encoder depth 4 against
6). A checkpoint that does not record its shape either fails to load with an
unhelpful size-mismatch traceback, or -- on some future refactor where the
shapes happen to line up -- loads into the wrong model and reports a number
nobody can reproduce.
"""
import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from eval_probe import build_model  # noqa: E402
from train_jepa import ARCH_KEYS, save_checkpoint  # noqa: E402

# Every key in ARCH_KEYS has to be here: save_checkpoint reads them off the
# training args, so a shape the checkpoint format grew (`use_tr`) and the test
# did not is an AttributeError at save time, not a silent omission.
SMALL = dict(embed_dim=32, encoder_depth=1, predictor_depth=1, num_heads=2,
             use_tr=True)


def _trained_args(**overrides):
    return Namespace(window_size=100, **{**SMALL, **overrides})


def test_checkpoint_records_the_architecture_it_was_trained_with(tmp_path):
    from deepmreye.models.jepa import JEPAModel

    args = _trained_args()
    model = JEPAModel(**SMALL)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, args, epoch=7)

    state = torch.load(path, map_location="cpu")
    assert state["epoch"] == 7
    assert state["arch"] == {k: getattr(args, k) for k in ARCH_KEYS}


def test_eval_loads_the_checkpoint_shape_not_its_own_defaults(tmp_path):
    from deepmreye.models.jepa import JEPAModel

    path = tmp_path / "last.pt"
    save_checkpoint(path, JEPAModel(**SMALL), _trained_args(), epoch=1)

    # Deliberately mismatched: these are the eval script's defaults, not the
    # trained shape. The checkpoint's own record must win.
    eval_args = Namespace(embed_dim=256, encoder_depth=6, predictor_depth=3, num_heads=8)
    model = build_model(eval_args, torch.device("cpu"), str(path))

    assert model.embed_dim == SMALL["embed_dim"]


def test_saving_is_atomic(tmp_path):
    """An interrupted save must not leave a half-written checkpoint in place of
    a good one -- training writes `last.pt` every epoch."""
    from deepmreye.models.jepa import JEPAModel

    path = tmp_path / "last.pt"
    save_checkpoint(path, JEPAModel(**SMALL), _trained_args(), epoch=1)
    save_checkpoint(path, JEPAModel(**SMALL), _trained_args(), epoch=2)

    assert not list(tmp_path.glob("*.tmp"))
    assert torch.load(path, map_location="cpu")["epoch"] == 2
