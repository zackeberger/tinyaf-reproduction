"""Check the model's beat/RR input shapes, units, padding, and inference mode."""

import numpy as np
import pytest
import torch

from python_model import pipeline
from python_model.runtime import load_model, predict_one


def test_predict_one_input_units_and_padding(monkeypatch):
    beats = np.zeros((20, 100), dtype=np.float64)
    beats[:2] = np.arange(200).reshape(2, 100)
    rr = np.zeros(20, dtype=np.int64)
    rr[:2] = [73, 41]

    monkeypatch.setattr(pipeline, "stack_beats", lambda trace, **kwargs: (beats, rr, 2))
    model = load_model("TINYAF-S").train()
    inputs = []

    def capture_inputs(net, args):
        assert not net.training
        assert not torch.is_grad_enabled()
        inputs.extend(args)

    handle = model.register_forward_pre_hook(capture_inputs)
    try:
        predict_one(np.zeros(2560), 256, model)
    finally:
        handle.remove()
    beat_tensor, rr_seconds = inputs

    assert beat_tensor.shape == (1, 20, 100)
    assert rr_seconds.shape == (1, 20)
    assert beat_tensor.dtype == rr_seconds.dtype == torch.float32
    assert beat_tensor.device.type == rr_seconds.device.type == "cpu"
    np.testing.assert_array_equal(beat_tensor[0].numpy(), beats)
    torch.testing.assert_close(
        rr_seconds[0, :2], torch.tensor([0.73, 0.41]), rtol=0, atol=0
    )
    assert torch.count_nonzero(rr_seconds[0, 2:]) == 0


@pytest.mark.parametrize("beat_shape,rr_shape", [
    ((1, 2000), (1, 20)),
    ((1, 20, 100), (1, 19)),
    ((2, 20, 100), (1, 20)),
])
def test_model_rejects_wrong_input_shapes(beat_shape, rr_shape):
    model = load_model("TINYAF-S")
    with pytest.raises(ValueError, match="expected"):
        model(torch.zeros(beat_shape), torch.zeros(rr_shape))
