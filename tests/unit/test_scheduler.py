from flexengine.scheduler.policies import Scheduler
from flexengine.scheduler.state import Request, State


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


def test_lifecycle_invalid_transition():
    import pytest
    a = req("a", [1, 2])
    a.transition(State.FINISHED)
    with pytest.raises(ValueError):
        a.transition(State.RUNNING)
