"""Tests for the package itself: that it imports and states its version."""

import re

import facial_joints_layout


def test_package_states_a_semantic_version():
    version = facial_joints_layout.__version__

    assert re.fullmatch(r"\d+\.\d+\.\d+", version)
