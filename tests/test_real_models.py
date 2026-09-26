"""End to end with the real checkpoints: gRPC answers must match laya's Router.predict.

Slow (loads two checkpoints, ~3 GB). Run with `pytest -m models`; skipped when the checkpoints
are not in .hf-cache/. Runs on CPU; set FTC_TEST_DEVICE=mps or FTC_TEST_DEVICE=cuda to test a GPU.
"""
import os
import time

import grpc
import pytest

from fast_typed_classifier import client
from fast_typed_classifier.convert import answers_from_proto
from fast_typed_classifier.engine import Engine
from fast_typed_classifier.v1 import classifier_pb2 as pb
from fast_typed_classifier.v1 import classifier_pb2_grpc as pb_grpc

from conftest import ROOT, running_server

DEVICE = os.environ.get("FTC_TEST_DEVICE", "cpu")
# A request that shares a batch can be padded differently than alone. On CPU that changes nothing
# measurable. On GPUs, laya also switches to fp16 autocast once a forward pass has enough rows
# (MPS: 5), so a batched request can run in fp16 where it would run alone in fp32; measured up to
# 0.0062 on MPS. Chosen labels must still match exactly.
BATCHED_TOL = 5e-3 if DEVICE == "cpu" else 2e-2

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(not (ROOT / ".hf-cache" / "hub" / "models--convaiinnovations--laya").is_dir(),
                       reason="checkpoints not downloaded to .hf-cache/"),
]

# The README's question set, written as laya dicts independently of convert.py.
TRIAGE = {
    "department": {"type": "choice", "instructions": "Which department should handle this request?",
                   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                                "sales": "pricing, new contracts", "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    "refund_requested": {"type": "noul", "instructions": "Does the user explicitly request a refund?"},
}
DEPARTMENT = {"department": TRIAGE["department"]}
MIXED = {
    "sentiment": {"type": "choice", "instructions": "What is the sentiment?", "criteria": ["positive", "neutral", "negative"]},
    "spam": {"type": "noul", "instructions": "Is this spam?",
             "criteria": {"true": "unsolicited advertising", "false": "a genuine customer message"},
             "labels": {"true": "A", "false": "B"}},
    "severity": {"type": "score", "instructions": "How severe is the problem?",
                 "criteria": ["none", "minor", "moderate", "major", "critical"]},
}
EMAIL = {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
         "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."}
LONG = " ".join(["Our dashboard has been timing out for three days and nobody has replied to our tickets."] * 20)

# (name, state, questions, predict kwargs)
CASES = [
    ("english_email", EMAIL, TRIAGE, {}),
    ("hindi", "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।", TRIAGE, {}),
    ("german", "Der Kunde wurde zweimal belastet und möchte sofort eine Rückerstattung, sonst kündigt er.", TRIAGE, {}),
    ("spanish", "La aplicación se cierra cada vez que abro la configuración.", DEPARTMENT, {}),
    ("conversation", [{"role": "user", "content": "My login stopped working."},
                      {"role": "agent", "content": "Have you tried resetting your password?"},
                      {"role": "user", "content": "Yes, and now I am locked out. This is blocking my whole team!"}],
     TRIAGE, {}),
    ("french_lang_hint", "Je veux annuler mon abonnement immédiatement.", TRIAGE, {"lang": "fr"}),
    ("english_forced_multilingual", EMAIL, TRIAGE, {"model": "multilingual"}),
    ("long_truncated", LONG, TRIAGE, {"max_len": 64}),
    ("mixed_types", "BUY CHEAP WATCHES NOW!!! Limited offer, click here.", MIXED, {}),
    ("no_letters", "!!! ??? 123", DEPARTMENT, {}),
]


def to_pb_questions(questions):
    out = []
    for qid, q in questions.items():
        if q["type"] == "choice":
            out.append(client.choice(qid, q["instructions"], q["criteria"]))
        elif q["type"] == "score":
            out.append(client.score(qid, q["instructions"], q["criteria"]))
        else:
            crit = q.get("criteria", {})
            msg = client.noul(qid, q["instructions"], crit.get("true", ""), crit.get("false", ""))
            if "labels" in q:
                msg.noul.true_label, msg.noul.false_label = q["labels"]["true"], q["labels"]["false"]
            out.append(msg)
    return out


def to_request(case, request_id=""):
    name, state, questions, kwargs = case
    return client.request(state, to_pb_questions(questions), request_id=request_id or name, **kwargs)


@pytest.fixture(scope="module")
def served():
    engine = Engine.load(["english", "multilingual"], device=DEVICE)
    assert engine.device.startswith(DEVICE), engine.device
    # References first, one at a time: they share the engine's loaded agents.
    references = {name: engine.router.predict(state, questions, **kwargs) for name, state, questions, kwargs in CASES}
    engine.warm_up()
    with running_server(engine) as server:
        with grpc.insecure_channel(server.target) as channel:
            yield pb_grpc.ClassifierStub(channel), references


def max_diff(response, reference, tol):
    """Compare a gRPC response to Router.predict's result; return the largest numeric difference."""
    assert not response.HasField("error"), response.error
    ref_routing = reference["routing"]
    assert (response.routing.model, response.routing.reason) == (ref_routing["model"], ref_routing["reason"])
    assert response.input_tokens == reference["usage"]["input_tokens"]
    got = answers_from_proto(response)
    assert set(got) == set(reference["answers"])
    worst = 0.0
    for qid, ref in reference["answers"].items():
        a = got[qid]
        assert a["type"] == ref["type"]
        if ref["type"] == "choice":
            assert a["choice"] == ref["choice"], qid
            pairs = list(zip(a["probabilities"], ref["probabilities"].values()))
        elif ref["type"] == "score":
            pairs = [(a["score"], ref["score"])] + list(zip(a["probabilities"], ref["probabilities"].values()))
        else:
            pairs = [(a["noul"], ref["noul"])]
        pairs += [(a["confidence"], ref["confidence"]), (a["answer_confidence"], ref["answer_confidence"]),
                  (a["act_probability"], ref["action"]["act_probability"])]
        for x, y in pairs:
            worst = max(worst, abs(x - y))
    assert worst <= tol, worst
    return worst


def test_each_case_matches_router_predict(served):
    stub, references = served
    worst = 0.0
    for case in CASES:
        response = stub.Classify(to_request(case), timeout=60)
        worst = max(worst, max_diff(response, references[case[0]], tol=1e-4))
    print("\nsequential: largest difference from Router.predict = %g" % worst)


def test_concurrent_requests_match_router_predict(served):
    """Every case three times, all at once, so requests share batches with other checkpoints'
    and other question sets' requests. Each must still get its own answers."""
    stub, references = served
    futures = [(case, stub.Classify.future(to_request(case, "%s-%d" % (case[0], i)), timeout=120))
               for i in range(3) for case in CASES]
    worst = 0.0
    batch_sizes = []
    for case, future in futures:
        response = future.result()
        assert response.request_id.startswith(case[0])
        worst = max(worst, max_diff(response, references[case[0]], tol=BATCHED_TOL))
        batch_sizes.append(response.timing.batch_size)
    assert max(batch_sizes) > 1
    print("\nconcurrent: largest difference = %g, batch sizes seen = %s" % (worst, sorted(set(batch_sizes))))


def test_stream_and_batch_match_router_predict(served):
    stub, references = served
    responses = list(stub.ClassifyStream(iter([to_request(c) for c in CASES]), timeout=120))
    assert sorted(r.request_id for r in responses) == sorted(c[0] for c in CASES)
    for r in responses:
        max_diff(r, references[r.request_id], tol=BATCHED_TOL)
    batch = stub.ClassifyBatch(pb.ClassifyBatchRequest(requests=[to_request(c) for c in CASES]), timeout=120)
    assert [r.request_id for r in batch.responses] == [c[0] for c in CASES]
    for r in batch.responses:
        max_diff(r, references[r.request_id], tol=BATCHED_TOL)


def test_single_request_latency_is_model_bound(served):
    """The service adds little on top of the forward pass (measured on an idle server)."""
    stub, _ = served
    req = to_request(CASES[0])
    stub.Classify(req, timeout=60)
    overheads = []
    for _ in range(5):
        t0 = time.perf_counter()
        r = stub.Classify(req, timeout=60)
        total_ms = (time.perf_counter() - t0) * 1000
        overheads.append(total_ms - r.timing.inference_ms)
    overheads.sort()
    print("\nend-to-end minus inference: median %.2f ms" % overheads[2])
    assert overheads[2] < 25  # tokenization + routing + gRPC, on a laptop CPU
