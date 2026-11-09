from dataclasses import replace

from conftest import TinyAdapter

from flexengine.runtime import Engine
from flexengine.scheduler.state import State


def finish(engine: Engine, request, limit: int = 80) -> None:
    for _ in range(limit):
        engine.step()
        if request.state in (State.FINISHED, State.FAILED):
            break
    assert request.state is State.FINISHED


def prefix_engine(conf, **changes) -> Engine:
    options = {"prefix_cache": True, "prefix_cache_max_entries": 4, **changes}
    config = replace(conf, **options)
    return Engine(config, TinyAdapter())


def test_runtime_prefix_hit_skips_complete_blocks_and_records_saved_tokens(conf):
    engine = prefix_engine(conf)
    try:
        warm = engine.submit("abcdefgh", max_tokens=1)
        finish(engine, warm)
        request = engine.submit("abcdefghij", max_tokens=2)

        engine.step()

        assert request.prefix_cache_hit is True
        assert request.saved_prefill_tokens == 8
        assert request.cached_tokens >= 8
        assert engine.prefix_cache.hits == 1
        assert engine.snapshot()["prefix_saved_tokens"] >= 8
    finally:
        engine.stop()


def test_prefix_cache_output_matches_disabled_runtime(conf):
    baseline = Engine(replace(conf, prefix_cache=False), TinyAdapter())
    cached = prefix_engine(conf)
    try:
        expected = baseline.submit("abcdefghij", max_tokens=5)
        finish(baseline, expected)
        warm = cached.submit("abcdefgh", max_tokens=1)
        finish(cached, warm)
        actual = cached.submit("abcdefghij", max_tokens=5)
        finish(cached, actual)

        assert actual.generated_ids == expected.generated_ids
        assert actual.text == expected.text
        assert actual.saved_prefill_tokens == 8
    finally:
        baseline.stop()
        cached.stop()


def test_cancelling_one_shared_request_preserves_peer_and_cache(conf):
    engine = prefix_engine(conf)
    try:
        warm = engine.submit("abcdefgh", max_tokens=1)
        finish(engine, warm)
        first = engine.submit("abcdefghX", max_tokens=4)
        second = engine.submit("abcdefghY", max_tokens=4)
        engine.step()
        assert first.prefix_cache_hit and second.prefix_cache_hit

        assert engine.cancel(first.request_id)
        finish(engine, second)

        assert second.state is State.FINISHED
        assert engine.prefix_cache.shared_blocks == 2
        engine.cache.assert_invariants()
    finally:
        engine.stop()


def test_runtime_lru_evicts_old_prefix(conf):
    engine = prefix_engine(conf, prefix_cache_max_entries=1)
    try:
        first = engine.submit("abcdefgh", max_tokens=1)
        finish(engine, first)
        second = engine.submit("ijklmnop", max_tokens=1)
        finish(engine, second)

        miss = engine.submit("abcdefghZ", max_tokens=1)
        engine.step()

        assert engine.prefix_cache.evictions >= 1
        assert miss.prefix_cache_hit is False
        assert miss.saved_prefill_tokens == 0
    finally:
        engine.stop()
