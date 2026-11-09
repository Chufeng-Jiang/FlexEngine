import pytest
import torch

from flexengine.execution.cuda_graph import DecodeGraphManager


def test_bucket_selection_pads_up_without_delaying():
    manager = DecodeGraphManager(device="cpu")

    assert [manager.bucket_for(n) for n in (1, 2, 3, 4, 5, 8)] == [1, 2, 4, 4, 8, 8]
    assert manager.bucket_for(9) is None
    assert manager.eligible(1) is False


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_capture_replay_padding_stable_addresses_and_parity():
    manager = DecodeGraphManager(device="cuda:0")

    def decode(input_ids, positions, active_mask):
        return (input_ids.float() + positions.float()) * active_mask

    manager.capture(
        4,
        decode,
        {
            "input_ids": torch.ones(4, 1, dtype=torch.long, device="cuda:0"),
            "positions": torch.zeros(4, 1, dtype=torch.long, device="cuda:0"),
        },
    )
    addresses = manager.staging_addresses(4)
    inputs = {
        "input_ids": torch.tensor([[3], [7], [11]], device="cuda:0"),
        "positions": torch.tensor([[2], [4], [6]], device="cuda:0"),
    }

    got = manager.replay(3, inputs)
    expected = decode(**inputs, active_mask=torch.ones(3, 1, device="cuda:0"))

    torch.testing.assert_close(got, expected)
    assert manager.staging_addresses(4) == addresses
    assert manager.stats()["replays"] == 1
    assert manager.stats()["padded_slots"] == 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_unsupported_bucket_uses_eager_fallback():
    manager = DecodeGraphManager(device="cuda:0")
    inputs = {"input_ids": torch.arange(9, device="cuda:0").reshape(9, 1)}

    output = manager.run(lambda input_ids: input_ids + 1, inputs)

    torch.testing.assert_close(output, inputs["input_ids"] + 1)
    assert manager.stats()["fallbacks"] == 1
