"""Tests for json_gui.scripts.mimic_classes."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from json_gui.scripts import mimic_classes


def test_rotator_identity_path_keeps_image_and_returns_noop() -> None:
    """A zero-degree rotation should return the image unchanged."""
    node = mimic_classes.Rotator(angle=0)
    image = torch.ones((1, 2, 2, 3))

    result_image, undo = node._process_impl(image)

    assert torch.equal(result_image, image)
    assert undo(image).equal(image)


def test_empty_latent_uses_vae_and_records_unsaved_tensor() -> None:
    """EmptyLatent should decode an empty latent and store the preview tensor."""
    node = mimic_classes.EmptyLatent(width=16, height=16, batch_size=1, image_name="<None>")
    vae = MagicMock()
    vae.decode.return_value = torch.ones((1, 3, 2, 2))

    latent = node._process_impl(vae)

    assert latent.shape == (1, 16, 2, 2)
    assert len(node.unsaved_tensors) == 1
    assert vae.decode.call_count == 1


def test_prompts_encode_both_prompts_into_datawrappers() -> None:
    """Prompt encoding should wrap positive and negative conditionings."""
    clip = MagicMock()
    clip.tokenize.side_effect = [["pos"], ["neg"]]
    clip.encode_from_tokens_scheduled.side_effect = [["encoded-pos"], ["encoded-neg"]]
    node = mimic_classes.Prompts(positive="sun", negative="rain")

    pos, neg = node._process_impl(clip)

    assert pos.get() == ["encoded-pos"]
    assert neg.get() == ["encoded-neg"]
    assert clip.tokenize.call_count == 2

