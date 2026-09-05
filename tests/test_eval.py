"""Tests for the bits-per-byte evaluation math (prepare.evaluate_val_bpb).

Anchors the constant 8.0 that evaluate_quality.py relies on: a model that
predicts a uniform distribution over the 256 byte values has cross-entropy
ln(256) nats, i.e. exactly 8.0 bits per byte. Runs on CPU, no data download.
"""
import math

import torch

from prepare import evaluate_val_bpb, get_batch, VOCAB_SIZE


class _UniformModel(torch.nn.Module):
    """Predicts a uniform 256-way distribution: cross-entropy == ln(256) nats."""

    def forward(self, x, y):
        loss = torch.tensor(math.log(VOCAB_SIZE))
        return None, loss


def test_uniform_logits_give_bpb_8():
    torch.manual_seed(0)  # get_batch uses the global RNG; seed for determinism
    val_data = torch.randint(0, VOCAB_SIZE, (10_000,), dtype=torch.long)
    bpb = evaluate_val_bpb(_UniformModel(), val_data, batch_size=4, block_size=64,
                           device="cpu", n_tokens=4 * 64 * 5)
    assert abs(bpb - 8.0) < 1e-6  # ln(256)/ln(2) == 8 up to float roundoff


def test_bpb_is_nats_over_ln2():
    # bpb = cross_entropy_nats / ln(2); a 2-nat loss is 2/ln2 bits per byte.
    class _TwoNat(torch.nn.Module):
        def forward(self, x, y):
            return None, torch.tensor(2.0)

    torch.manual_seed(0)
    val_data = torch.randint(0, VOCAB_SIZE, (5_000,), dtype=torch.long)
    bpb = evaluate_val_bpb(_TwoNat(), val_data, batch_size=2, block_size=32,
                           device="cpu", n_tokens=2 * 32 * 3)
    assert abs(bpb - 2.0 / math.log(2)) < 1e-6


def test_get_batch_shapes_and_determinism():
    data = torch.arange(1000, dtype=torch.long)
    torch.manual_seed(123)
    x1, y1 = get_batch(data, batch_size=8, block_size=16, device="cpu")
    assert x1.shape == (8, 16) and y1.shape == (8, 16)
    # y is x shifted by one (next-token targets).
    torch.manual_seed(123)
    x2, y2 = get_batch(data, batch_size=8, block_size=16, device="cpu")
    assert torch.equal(x1, x2) and torch.equal(y1, y2)  # same global seed -> same batch
    # check the shift relationship on the first row
    start = int(x1[0, 0].item())
    assert torch.equal(x1[0], torch.arange(start, start + 16))
    assert torch.equal(y1[0], torch.arange(start + 1, start + 17))


# --------------------------------------------------------------------------
# Fixed evaluation set (pre-registered experiment 4)
# --------------------------------------------------------------------------
def _tiny_model():
    """A deterministic model whose loss depends on the batch it is given.

    A constant-loss stub would pass the bit-identical test trivially, without
    the eval set ever having to be fixed. This one keys on the data.
    """
    class _M(torch.nn.Module):
        def forward(self, x, y):
            return None, x.float().mean() / 100.0
    return _M()


def test_fixed_eval_draws_nothing_from_the_global_rng():
    """The whole point: building the set must not perturb training data order.

    get_batch() draws from the global stream, so a single draw here would shift
    every subsequent training batch and change the runs being measured.
    """
    import prepare
    val = torch.randint(0, VOCAB_SIZE, (20_000,), dtype=torch.long)
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    # .__wrapped__ bypasses the lru_cache: a cache hit would do no work at all
    # and pass this test without the generator ever being exercised.
    prepare._fixed_offsets.__wrapped__(len(val), 4, 64, 4 * 64 * 8, 0)
    assert torch.equal(torch.get_rng_state(), before), \
        "fixed_eval_batches consumed from the global RNG stream"


def test_fixed_eval_is_bit_identical_across_calls():
    """The pre-registered acceptance criterion: sd exactly 0.0, not merely small."""
    from prepare import evaluate_val_bpb as ev
    val = torch.randint(0, VOCAB_SIZE, (20_000,), dtype=torch.long)
    m = _tiny_model()
    vals = []
    for _ in range(12):
        torch.manual_seed(torch.randint(0, 10_000, (1,)).item())  # global RNG churns
        vals.append(ev(m, val, batch_size=4, block_size=64, device="cpu",
                       n_tokens=4 * 64 * 8, fixed=True))
    assert len(set(vals)) == 1, f"fixed eval is not bit-identical: {sorted(set(vals))}"


def test_unfixed_eval_still_varies():
    """Guards the comparison: if the default were also constant, the test above
    would prove nothing about the fixed path."""
    from prepare import evaluate_val_bpb as ev
    val = torch.randint(0, VOCAB_SIZE, (20_000,), dtype=torch.long)
    m = _tiny_model()
    vals = {ev(m, val, batch_size=4, block_size=64, device="cpu",
               n_tokens=4 * 64 * 8) for _ in range(12)}
    assert len(vals) > 1, "the resampling path should vary between calls"


def test_fixed_offsets_are_in_range_and_shaped():
    from prepare import fixed_eval_batches, FIXED_EVAL_TOKENS
    val = torch.randint(0, VOCAB_SIZE, (500_000,), dtype=torch.long)
    off = fixed_eval_batches(val, batch_size=32, block_size=256)
    assert off.shape == (FIXED_EVAL_TOKENS // (32 * 256), 32) == (128, 32)
    assert off.min() >= 0
    # The target window reads i + block_size + 1, so this bound must hold exactly.
    assert (off.max() + 256 + 1) <= len(val)


def test_fixed_set_identity_depends_on_seed_and_size():
    from prepare import fixed_eval_batches
    val = torch.randint(0, VOCAB_SIZE, (500_000,), dtype=torch.long)
    a = fixed_eval_batches(val, 32, 256, 32 * 256 * 8, 0)
    assert torch.equal(a, fixed_eval_batches(val, 32, 256, 32 * 256 * 8, 0))
    assert not torch.equal(a, fixed_eval_batches(val, 32, 256, 32 * 256 * 8, 1))


def test_default_path_is_unchanged():
    """Old numbers must stay reproducible by the code that produced them."""
    from prepare import evaluate_val_bpb as ev, EVAL_TOKENS
    val = torch.randint(0, VOCAB_SIZE, (20_000,), dtype=torch.long)
    m = _tiny_model()
    torch.manual_seed(7)
    a = ev(m, val, batch_size=4, block_size=64, device="cpu", n_tokens=4 * 64 * 5)
    torch.manual_seed(7)
    b = ev(m, val, batch_size=4, block_size=64, device="cpu", n_tokens=4 * 64 * 5)
    assert a == b, "the default path must remain a pure function of the global seed"
    # And n_tokens=None must still mean EVAL_TOKENS, not the fixed size.
    import inspect
    assert inspect.signature(ev).parameters["n_tokens"].default is None
    assert EVAL_TOKENS == 100_000
