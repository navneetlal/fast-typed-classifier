import asyncio
import random

import pytest

from fast_typed_classifier import client
from fast_typed_classifier.convert import RequestError, build_question_set
from fast_typed_classifier.engine import ModelNotLoaded, Overloaded, ShuttingDown, default_limits

from conftest import fake_engine, tag_for

TRIAGE = build_question_set(client.SUPPORT_TRIAGE)
SHORT = build_question_set([client.noul("spam", "Is this spam?"), client.choice("lang", "Language?", ["en", "de"])])

ENGLISH = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
GERMAN = "Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die doppelte Abbuchung."
HINDI = "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"


def run(coro):
    return asyncio.run(coro)


async def started(engine):
    await engine.start()
    return engine


def assert_own_rows(result, state, qset):
    """Every answer came from this request's own row for that question."""
    assert list(result.answers) == list(qset.ids)
    for j, qid in enumerate(qset.ids):
        assert result.answers[qid]["tag"] == tag_for(state, j), (qid, state)


def test_single_request():
    async def main():
        engine = await started(fake_engine())
        result = await engine.classify(ENGLISH, TRIAGE)
        await engine.stop()
        return engine, result

    engine, result = run(main())
    assert_own_rows(result, ENGLISH, TRIAGE)
    assert result.routing["model"] == "english"
    assert result.answers["department"]["lang"] == "en"  # detected language goes to calibration
    assert result.batch_size == 1
    assert result.input_tokens == sum(len(ENGLISH) + 1 for _ in TRIAGE.ids)
    assert result.queue_ms >= 0 and result.inference_ms >= 0
    # the forward pass ran on the inference thread, not the event loop
    assert {c["thread"] for c in engine.agents["english"].calls} == {"ftc-infer_0"}


def test_routing_and_language_follow_router_predict():
    async def main():
        engine = await started(fake_engine())
        out = await asyncio.gather(
            engine.classify(GERMAN, TRIAGE),
            engine.classify(HINDI, TRIAGE),
            engine.classify(ENGLISH, TRIAGE, lang="fr"),        # explicit lang routes and calibrates
            engine.classify(ENGLISH, TRIAGE, model="multilingual"),
        )
        await engine.stop()
        return out

    german, hindi, fr, forced = run(main())
    assert german.routing["model"] == "multilingual" and german.answers["urgency"]["lang"] == "de"
    assert german.answers["urgency"]["model"] == "multilingual"
    assert hindi.routing["model"] == "multilingual" and hindi.answers["urgency"]["lang"] is None
    assert fr.routing["model"] == "multilingual" and fr.answers["urgency"]["lang"] == "fr"
    assert forced.routing["model"] == "multilingual" and forced.routing["reason"].startswith("explicit model")
    assert forced.answers["urgency"]["lang"] is None  # Router.predict: no detection when model is explicit


def test_many_concurrent_requests_get_their_own_answers_and_batch():
    """Mixed checkpoints, question sets and lengths, all at once: nothing crosses over."""
    rng = random.Random(7)
    words = "refund invoice outage login charge cancel plan urgent please help thanks".split()
    cases = []
    for i in range(120):
        base = rng.choice([ENGLISH, GERMAN, HINDI])
        state = "%s %d %s" % (base, i, " ".join(rng.choice(words) for _ in range(rng.randint(0, 60))))
        if i % 5 == 0:
            state = {"subject": "case %d" % i, "body": state}
        cases.append((state, rng.choice([TRIAGE, SHORT])))

    async def main():
        engine = await started(fake_engine(delay=0.005, max_batch_rows=64, max_batch_tokens=100_000))
        results = await asyncio.gather(*(engine.classify(s, q) for s, q in cases))
        await engine.stop()
        return engine, results

    engine, results = run(main())
    for (state, qset), result in zip(cases, results):
        assert_own_rows(result, state, qset)
    calls = [c for a in engine.agents.values() for c in a.calls]
    assert sum(c["rows"] for c in calls) == sum(len(q.ids) for _, q in cases)  # each row forwarded once
    assert max(r.batch_size for r in results) > 1  # requests queued behind a pass batched together
    assert all(c["rows"] <= 64 for c in calls)
    assert {r.routing["model"] for r in results} == {"english", "multilingual"}


def queue_behind_blocker(engine, states, qset):
    """Classify `states` while a first request holds the inference thread, so they all queue."""
    async def go():
        blocker = asyncio.create_task(engine.classify("block " * 5, qset))
        await asyncio.sleep(0.01)
        results = await asyncio.gather(*(engine.classify(s, qset) for s in states))
        await blocker
        await engine.stop()
        return results
    return go()


def test_short_rows_are_not_padded_to_a_long_one():
    """A batch whose rows fit the token budget unpadded, but not padded to its longest row, is
    sorted by length and split into passes."""
    states = ["x" * n for n in (10, 400, 12, 11, 13)]  # Latin, no language: routed to the default

    async def main():
        engine = await started(fake_engine(delay=0.05, max_batch_rows=16, max_batch_tokens=1000))
        return engine, await queue_behind_blocker(engine, states, SHORT)

    engine, results = run(main())
    for state, result in zip(states, results):
        assert_own_rows(result, state, SHORT)
    assert {r.batch_size for r in results} == {5}  # one batch...
    calls = engine.agents["english"].calls[1:]
    assert [(c["rows"], c["padded_len"]) for c in calls] == [(8, 14), (2, 401)]  # ...in two passes (+1: the tag token)


def test_batches_respect_row_and_token_budgets():
    rng = random.Random(3)
    states = ["x" * rng.randint(5, 300) + str(i) for i in range(40)]

    async def main():
        engine = await started(fake_engine(delay=0.01, max_batch_rows=12, max_batch_tokens=900))
        return engine, await queue_behind_blocker(engine, states, SHORT)

    engine, results = run(main())
    for state, result in zip(states, results):
        assert_own_rows(result, state, SHORT)
    calls = engine.agents["english"].calls[1:]
    assert sum(c["rows"] for c in calls) == 2 * len(states)
    for c in calls:
        assert c["rows"] <= 12
        assert c["rows"] * c["padded_len"] <= 900 or c["rows"] == 1
    assert max(r.batch_size for r in results) > 1


def test_cancelled_request_is_skipped():
    async def main():
        engine = await started(fake_engine(delay=0.2))
        first = asyncio.create_task(engine.classify(ENGLISH, SHORT))
        await asyncio.sleep(0.05)  # first is in its forward pass
        doomed = asyncio.create_task(engine.classify(ENGLISH + " doomed", SHORT))
        await asyncio.sleep(0.02)  # doomed is queued
        doomed.cancel()
        kept = await engine.classify(ENGLISH + " kept", SHORT)
        await first
        await engine.stop()
        return engine, doomed, kept

    engine, doomed, kept = run(main())
    assert doomed.cancelled()
    assert_own_rows(kept, ENGLISH + " kept", SHORT)
    assert tag_for(ENGLISH + " doomed", 0) not in engine.agents["english"].forwarded_tags()


def test_one_failing_request_does_not_fail_its_batch():
    async def main():
        engine = await started(fake_engine(delay=0.05, max_batch_rows=64))
        engine.agents["english"].fail_tags.add(tag_for(ENGLISH + " bad", 0))
        blocker = asyncio.create_task(engine.classify(ENGLISH, SHORT))
        await asyncio.sleep(0.01)
        states = [ENGLISH + " good %d" % i for i in range(5)] + [ENGLISH + " bad"]
        results = await asyncio.gather(*(engine.classify(s, SHORT) for s in states), return_exceptions=True)
        await blocker
        # the server keeps serving afterwards
        after = await engine.classify(ENGLISH + " after", SHORT)
        await engine.stop()
        return states, results, after

    states, results, after = run(main())
    for state, result in zip(states[:-1], results[:-1]):
        assert_own_rows(result, state, SHORT)
    assert isinstance(results[-1], RuntimeError) and "injected failure" in str(results[-1])
    assert_own_rows(after, ENGLISH + " after", SHORT)


def test_overload_is_rejected_not_queued():
    async def main():
        engine = await started(fake_engine(delay=0.1, max_inflight=3))
        results = await asyncio.gather(*(engine.classify(ENGLISH + str(i), SHORT) for i in range(8)),
                                       return_exceptions=True)
        await engine.stop()
        return results

    results = run(main())
    rejected = [r for r in results if isinstance(r, Overloaded)]
    assert len(rejected) == 5 and len(results) - len(rejected) == 3


def test_request_errors():
    async def main():
        engine = await started(fake_engine(models=("english",)))
        errors = []
        for kwargs in [dict(state=HINDI), dict(state=ENGLISH, model="typed-decisions"),
                       dict(state=ENGLISH, model="no-such-model")]:
            try:
                await engine.classify(kwargs.pop("state"), SHORT, **kwargs)
            except Exception as e:  # noqa: BLE001
                errors.append(e)
        await engine.stop()
        return errors

    not_loaded_hi, not_loaded_td, unknown = run(main())
    assert isinstance(not_loaded_hi, ModelNotLoaded) and "multilingual" in str(not_loaded_hi)
    assert isinstance(not_loaded_td, ModelNotLoaded)
    assert isinstance(unknown, RequestError) and "no-such-model" in str(unknown)


def test_multilingual_only_server_answers_english():
    async def main():
        engine = await started(fake_engine(models=("multilingual",)))
        auto = await engine.classify(ENGLISH, SHORT)
        errors = []
        for kwargs in [dict(model="english"), dict(model="typed-decisions")]:
            try:
                await engine.classify(ENGLISH, SHORT, **kwargs)
            except ModelNotLoaded as e:
                errors.append(e)
        await engine.stop()
        return auto, errors

    auto, errors = run(main())
    assert auto.routing["model"] == "multilingual"
    assert auto.routing["reason"] == "English Latin text; english is not loaded, so multilingual answers"
    assert auto.answers["spam"]["model"] == "multilingual" and auto.answers["spam"]["lang"] == "en"
    assert len(errors) == 2  # an explicit checkpoint is never swapped


def test_shutdown():
    async def main():
        engine = await started(fake_engine(delay=0.1))
        running = asyncio.create_task(engine.classify(ENGLISH, SHORT))
        await asyncio.sleep(0.02)
        engine.begin_shutdown()
        with pytest.raises(ShuttingDown):
            await engine.classify(ENGLISH, SHORT)
        result = await running  # in-flight work still completes
        await engine.stop()
        return result

    assert_own_rows(run(main()), ENGLISH, SHORT)


def test_max_wait_batches_staggered_arrivals():
    async def main():
        engine = await started(fake_engine(max_wait_ms=50, max_batch_rows=64))
        tasks = []
        for i in range(4):
            tasks.append(asyncio.create_task(engine.classify(ENGLISH + str(i), SHORT)))
            await asyncio.sleep(0.005)
        results = await asyncio.gather(*tasks)
        await engine.stop()
        return results

    assert {r.batch_size for r in run(main())} == {4}


def test_warm_up_runs_every_checkpoint():
    engine = fake_engine()
    engine.warm_up()
    assert all(agent.calls for agent in engine.agents.values())


def test_default_limits_by_device():
    assert default_limits("cuda").max_batch_rows == 512
    assert default_limits("cpu").max_batch_rows == 8
    assert default_limits("cpu", max_batch_rows=32, max_batch_tokens=None).max_batch_rows == 32
