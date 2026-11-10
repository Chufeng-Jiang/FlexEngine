from flexengine.engine.scheduler.policies import Scheduler
from flexengine.engine.scheduler.state import Request, State


def req(name, prompt, cached=0, generated=None):
    r = Request(name, prompt, 10)
    r.cached_tokens = cached
    r.generated_ids = generated or []
    r.transition(State.RUNNING)
    return r


def test_token_budget_and_chunked_prefill():
    a, b = req("a", list(range(30))), req("b", list(range(20)))
    plan = Scheduler("fcfs", 16, 1000, 100).plan([a, b])
    assert sum(len(c.input_ids) for c in plan) <= 16
    assert len(plan[0].input_ids) == 16


def test_decode_first_order():
    prefill = req("p", [2,3,4,5,6])
    decoding = req("d", [1,2], cached=2, generated=[7])
    plan = Scheduler("decode_first", 4, 1000, 100).plan([prefill, decoding])
    assert plan[0].request_id == "d"
    assert len(plan[0].input_ids) == 1
    assert len(plan[1].input_ids) == 3


def test_adaptive_reserves_decode_tokens_and_bounds_prefill_chunks():
    prefill = req("p", list(range(40)))
    decodes = [req(f"d{i}", [1, 2], cached=2, generated=[7]) for i in range(3)]

    plan = Scheduler("adaptive", 8, 1000, 100, prefill_chunk_size=4).plan([prefill, *decodes])

    assert [chunk.request_id for chunk in plan[:3]] == ["d0", "d1", "d2"]
    assert len(next(chunk for chunk in plan if chunk.request_id == "p").input_ids) == 4
    assert sum(len(chunk.input_ids) for chunk in plan) == 7


def test_adaptive_overdue_prefill_progresses_without_consuming_decode_reservation():
    prefill = req("p", list(range(40)))
    prefill.arrived -= 2.0
    decode = req("d", [1, 2], cached=2, generated=[7])

    plan = Scheduler("adaptive", 5, 1000, 100, prefill_chunk_size=4).plan([prefill, decode])

    assert {chunk.request_id for chunk in plan} == {"p", "d"}
    assert len(next(chunk for chunk in plan if chunk.request_id == "p").input_ids) == 4
    assert len(next(chunk for chunk in plan if chunk.request_id == "d").input_ids) == 1


def test_lifecycle_invalid_transition():
    import pytest
    a = req("a", [1, 2])
    a.transition(State.FINISHED)
    with pytest.raises(ValueError):
        a.transition(State.RUNNING)
