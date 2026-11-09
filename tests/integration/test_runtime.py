from dataclasses import replace
from queue import Empty

from conftest import TinyAdapter

from flexengine.runtime import BusyError, Engine
from flexengine.scheduler.state import State


def drain(req):
    events = []
    while True:
        try:
            events.append(req.events.get_nowait())
        except Empty:
            break
    return events


def test_two_requests_continuous_batching_and_free(engine):
    a = engine.submit("abcd", max_tokens=5)
    b = engine.submit("z", max_tokens=3)
    for _ in range(20):
        if a.state is State.FINISHED and b.state is State.FINISHED:
            break
        engine.step()
    assert (a.state, b.state) == (State.FINISHED, State.FINISHED)
    assert len(a.generated_ids) == 5 and len(b.generated_ids) == 3
    assert a.text and b.text and a.text != b.text
    assert engine.cache.used_blocks == 0
    assert any(event.kind == "done" for event in drain(a))
    engine.cache.assert_invariants()


def test_cancel_mid_generation(engine):
    a = engine.submit("hello", max_tokens=9)
    b = engine.submit("world", max_tokens=4)
    for _ in range(2):
        engine.step()
    assert engine.cancel(a.request_id)
    assert a.state is State.CANCELLED
    for _ in range(15):
        engine.step()
    assert b.state is State.FINISHED
    assert engine.cache.used_blocks == 0
    engine.cache.assert_invariants()


def test_backpressure(conf):
    e = Engine(replace(conf, max_waiting=1), TinyAdapter())
    try:
        e.submit("first", max_tokens=2)
        try:
            e.submit("second", max_tokens=2)
            assert False, "Expected BusyError"
        except BusyError:
            pass
    finally:
        e.stop()


def test_cache_exhaustion_no_GPU_OOM(conf):
    # 1 block for each of 2 simultaneous prompts plus a small spare pool.
    cfg = replace(conf, kv_num_blocks=4, kv_block_size=4, max_model_len=16,
                  max_num_seqs=2, max_num_batched_tokens=3)
    e = Engine(cfg, TinyAdapter())
    try:
        a = e.submit("abcde", max_tokens=6)
        b = e.submit("fghij", max_tokens=6)
        for _ in range(80):
            e.step()
            if a.state in (State.FINISHED,State.FAILED) and b.state in (State.FINISHED, State.FAILED):
                break
        assert e.cache.used_blocks == 0
        e.cache.assert_invariants()
        assert a.state == b.state == State.FINISHED
    finally:
        e.stop()


def test_preemption_preserves_outputs(conf):
    cfg = replace(conf, kv_num_blocks=3, kv_block_size=4, max_model_len=12,
                  max_num_seqs=2, max_num_batched_tokens=2)
    e = Engine(cfg, TinyAdapter())
    try:
        a = e.submit("abc", max_tokens=5)
        b = e.submit("xyz", max_tokens=5)
        for _ in range(200):
            e.step()
            if a.state is State.FINISHED and b.state is State.FINISHED:
                break
        assert a.state == b.state == State.FINISHED
        assert len(a.generated_ids) == len(b.generated_ids) == 5
        assert e.cache.free_blocks == cfg.kv_num_blocks
    finally:
        e.stop()


def test_recompute_preemption_no_duplicate_generation(engine):
    a = engine.submit("abcd", max_tokens=7)
    for _ in range(4):
        engine.step()
    emitted_before = list(a.generated_ids)
    assert emitted_before
    with engine.lock:
        engine._preempt(a)
        assert a.state is State.PREEMPTED
        assert a.cached_tokens == 0
        assert a.replaying is True
    for _ in range(40):
        engine.step()
        if a.state is State.FINISHED:
            break
    assert a.state is State.FINISHED
    assert a.generated_ids[:len(emitted_before)] == emitted_before
    assert len(a.generated_ids) == 7
    assert engine.cache.free_blocks == engine.cache.layout.num_blocks
