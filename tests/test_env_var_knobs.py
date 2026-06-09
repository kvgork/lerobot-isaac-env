"""test_env_var_knobs — verify LEROBOT_ISAAC_* env vars are read by pick_and_place.

These tests are import-time behavioural so they monkeypatch os.environ before
reimporting the module."""
from __future__ import annotations

import importlib
import os
import sys

import pytest


@pytest.fixture
def reimport_pick_and_place(monkeypatch):
    """Drop the module from sys.modules so re-import picks up the patched env."""

    def _reimport():
        mod_name = "lerobot_isaac_env.tasks.pick_and_place"
        sys.modules.pop(mod_name, None)
        return importlib.import_module(mod_name)

    return _reimport


def test_progress_weight_default(monkeypatch, reimport_pick_and_place):
    monkeypatch.delenv("LEROBOT_ISAAC_PROGRESS_WEIGHT", raising=False)
    mod = reimport_pick_and_place()
    assert mod._PROGRESS_WEIGHT == 10.0


def test_progress_weight_zero(monkeypatch, reimport_pick_and_place):
    monkeypatch.setenv("LEROBOT_ISAAC_PROGRESS_WEIGHT", "0")
    mod = reimport_pick_and_place()
    assert mod._PROGRESS_WEIGHT == 0.0


def test_object_pos_default(monkeypatch, reimport_pick_and_place):
    for k in ("LEROBOT_ISAAC_OBJECT_X", "LEROBOT_ISAAC_OBJECT_Y", "LEROBOT_ISAAC_OBJECT_Z"):
        monkeypatch.delenv(k, raising=False)
    mod = reimport_pick_and_place()
    # Default moved INSIDE SO-101 reach (~0.346 m) 2026-06-09 — the prior
    # (0.5, 0.1)=0.51 m default was beyond reach so grasp/lift/place never fired.
    assert mod._OBJECT_POS == (0.22, 0.05, 0.05)
    assert (mod._OBJECT_POS[0] ** 2 + mod._OBJECT_POS[1] ** 2) ** 0.5 < 0.30


def test_object_pos_home_curriculum(monkeypatch, reimport_pick_and_place):
    monkeypatch.setenv("LEROBOT_ISAAC_OBJECT_X", "0.30")
    monkeypatch.setenv("LEROBOT_ISAAC_OBJECT_Y", "0.05")
    monkeypatch.setenv("LEROBOT_ISAAC_OBJECT_Z", "0.05")
    mod = reimport_pick_and_place()
    assert mod._OBJECT_POS == (0.30, 0.05, 0.05)
