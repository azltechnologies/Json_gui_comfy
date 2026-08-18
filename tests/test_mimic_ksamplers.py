"""Tests for json_gui.scripts.mimic_ksamplers."""

from __future__ import annotations

from json_gui.scripts import mimic_ksamplers


def test_simple_k_sampler_exposes_seed_configuration() -> None:
    """SimpleKSampler should store the sampler configuration in a plain dict."""
    node = mimic_ksamplers.SimpleKSampler(
        seed=123,
        steps=12,
        cfg=7.5,
        sampler_name="euler",
        scheduler="normal",
        denoise=0.9,
        use_tune=True,
    )

    assert node.key() == "simple_k_sampler"
    assert node.use_tune is True
    assert node._to_dict() == {
        "seed": 123,
        "steps": 12,
        "cfg": 7.5,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 0.9,
    }

