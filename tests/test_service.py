"""The gRPC surface, over a real socket, with fake checkpoints."""
import threading
import time

import grpc
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection_pb2, reflection_pb2_grpc

from fast_typed_classifier import client
from fast_typed_classifier.v1 import classifier_pb2 as pb
from fast_typed_classifier.v1 import classifier_pb2_grpc as pb_grpc

from conftest import fake_engine, tag_for

ENGLISH = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
HINDI = "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"
QUESTIONS = client.SUPPORT_TRIAGE


@pytest.fixture
def stub(server_factory):
    server = server_factory(fake_engine())
    with grpc.insecure_channel(server.target) as channel:
        yield pb_grpc.ClassifierStub(channel)


def test_classify(stub):
    r = stub.Classify(client.request(ENGLISH, QUESTIONS, request_id="r1"), timeout=10)
    assert r.request_id == "r1"
    assert set(r.answers) == {"department", "urgency", "churn_risk", "refund_requested"}
    dept = r.answers["department"]
    assert dept.type == pb.QUESTION_TYPE_CHOICE and dept.choice == "billing"
    assert list(dept.probabilities) == [1.0, 0.0, 0.0, 0.0]
    assert r.answers["urgency"].WhichOneof("value") == "score" and len(r.answers["urgency"].probabilities) == 3
    assert r.answers["churn_risk"].noul == 0.25 and list(r.answers["churn_risk"].probabilities) == [0.75, 0.25]
    assert (r.routing.model, r.routing.detected_script, r.routing.detected_language) == ("english", "latin", "en")
    assert r.timing.batch_size == 1 and r.input_tokens > 0
    assert not r.HasField("error")


def test_classify_json_state_and_overrides(stub):
    r = stub.Classify(client.request({"subject": "Refund", "body": ENGLISH}, QUESTIONS, model="multilingual"),
                      timeout=10)
    assert r.routing.model == "multilingual" and r.routing.reason == "explicit model='multilingual'"
    r = stub.Classify(client.request(HINDI, QUESTIONS), timeout=10)
    assert r.routing.model == "multilingual" and r.routing.detected_script == "devanagari"


@pytest.mark.parametrize("request_, message", [
    (pb.ClassifyRequest(questions=QUESTIONS), "either text or json"),
    (pb.ClassifyRequest(json="{oops", questions=QUESTIONS), "not valid JSON"),
    (pb.ClassifyRequest(text="hi"), "at least one question"),
    (pb.ClassifyRequest(text="hi", questions=[client.noul("a", "x"), client.noul("a", "y")]), "used twice"),
    (pb.ClassifyRequest(text="hi", questions=[pb.Question(id="a", instructions="x")]), "choice, score or noul"),
    (pb.ClassifyRequest(text="hi", questions=QUESTIONS, model="gpt"), "gpt"),
    (pb.ClassifyRequest(text="hi", questions=QUESTIONS, max_len=100_000), "above the limit of 8192"),
])
def test_invalid_argument(stub, request_, message):
    with pytest.raises(grpc.RpcError) as e:
        stub.Classify(request_, timeout=10)
    assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert message in e.value.details()


def test_checkpoint_not_loaded(stub):
    with pytest.raises(grpc.RpcError) as e:
        stub.Classify(client.request(ENGLISH, QUESTIONS, model="typed-decisions"), timeout=10)
    assert e.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert "typed-decisions" in e.value.details()


def test_classify_batch(stub):
    shared = [client.noul("spam", "Is this spam?")]
    req = pb.ClassifyBatchRequest(questions=shared, requests=[
        client.request(ENGLISH, request_id="a"),                      # uses the batch questions
        client.request(HINDI, QUESTIONS, request_id="b"),             # its own questions
        pb.ClassifyRequest(request_id="c", questions=shared),         # no state: fails alone
        client.request(ENGLISH + " again", request_id="d"),
    ])
    r = stub.ClassifyBatch(req, timeout=10)
    assert [x.request_id for x in r.responses] == ["a", "b", "c", "d"]
    assert set(r.responses[0].answers) == {"spam"}
    assert set(r.responses[1].answers) == {q.id for q in QUESTIONS}
    assert r.responses[2].error.code == grpc.StatusCode.INVALID_ARGUMENT.value[0]
    assert "text or json" in r.responses[2].error.message and not r.responses[2].answers
    assert not r.responses[3].HasField("error")


def test_classify_stream(stub):
    ids = ["s%d" % i for i in range(40)]
    requests = [client.request("%s %s" % (ENGLISH if i % 2 else HINDI, i), QUESTIONS, request_id=rid)
                for i, rid in enumerate(ids)]
    requests.insert(10, pb.ClassifyRequest(request_id="bad", text="x"))  # no questions
    responses = list(stub.ClassifyStream(iter(requests), timeout=20))
    by_id = {r.request_id: r for r in responses}
    assert len(responses) == len(requests) and set(by_id) == set(ids) | {"bad"}
    assert by_id["bad"].error.code == grpc.StatusCode.INVALID_ARGUMENT.value[0]
    for rid in ids:
        assert set(by_id[rid].answers) == {q.id for q in QUESTIONS}
        i = int(rid[1:])
        assert by_id[rid].routing.model == ("english" if i % 2 else "multilingual")


def test_empty_stream(stub):
    assert list(stub.ClassifyStream(iter([]), timeout=10)) == []


def test_route_does_not_run_the_model(server_factory):
    engine = fake_engine()
    server = server_factory(engine)
    with grpc.insecure_channel(server.target) as channel:
        stub = pb_grpc.ClassifierStub(channel)
        r = stub.Route(client.request("Der Kunde wurde zweimal belastet", QUESTIONS), timeout=10)
        assert (r.model, r.detected_language) == ("multilingual", "de")
        assert r.reason == "Latin script but language looks like 'de', not English"
        r = stub.Route(client.request("hello", model="typed-decisions"), timeout=10)  # questions optional
        assert r.model == "typed-decisions"
        with pytest.raises(grpc.RpcError) as e:
            stub.Route(pb.ClassifyRequest(), timeout=10)
        assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert not any(a.calls for a in engine.agents.values())


def test_server_info(stub):
    info = stub.GetServerInfo(pb.GetServerInfoRequest(), timeout=10)
    assert info.device == "cpu"
    assert [m.name for m in info.models] == ["english", "multilingual"]
    assert info.models[0].repo == "convaiinnovations/laya" and info.models[0].max_len == 512
    assert (info.max_batch_rows, info.max_batch_tokens, info.max_inflight) == (8, 4096, 1024)
    assert info.laya_version == "0.3.20"


def test_health_and_reflection(server_factory):
    server = server_factory(fake_engine())
    with grpc.insecure_channel(server.target) as channel:
        health = health_pb2_grpc.HealthStub(channel)
        for service in ("", "fast_typed_classifier.v1.Classifier"):
            status = health.Check(health_pb2.HealthCheckRequest(service=service), timeout=5).status
            assert status == health_pb2.HealthCheckResponse.SERVING
        refl = reflection_pb2_grpc.ServerReflectionStub(channel)
        reply = next(refl.ServerReflectionInfo(iter([reflection_pb2.ServerReflectionRequest(list_services="")])))
        names = {s.name for s in reply.list_services_response.service}
        assert {"fast_typed_classifier.v1.Classifier", "grpc.health.v1.Health"} <= names


def test_deadline_exceeded_request_is_dropped(server_factory):
    engine = fake_engine(delay=0.5)
    server = server_factory(engine)
    with grpc.insecure_channel(server.target) as channel:
        stub = pb_grpc.ClassifierStub(channel)
        blocker = stub.Classify.future(client.request(ENGLISH + " blocker", QUESTIONS), timeout=10)
        time.sleep(0.1)
        with pytest.raises(grpc.RpcError) as e:
            stub.Classify(client.request(ENGLISH + " late", QUESTIONS), timeout=0.2)
        assert e.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
        blocker.result()
        time.sleep(0.6)
    # the timed-out request was cancelled while queued, so its rows never reached the model
    assert tag_for(ENGLISH + " late", 0) not in engine.agents["english"].forwarded_tags()


def test_overload_returns_resource_exhausted(server_factory):
    server = server_factory(fake_engine(delay=0.3, max_inflight=2))
    with grpc.insecure_channel(server.target) as channel:
        stub = pb_grpc.ClassifierStub(channel)
        futures = [stub.Classify.future(client.request(ENGLISH + str(i), QUESTIONS), timeout=10) for i in range(6)]
        codes = []
        for f in futures:
            try:
                f.result()
                codes.append(grpc.StatusCode.OK)
            except grpc.RpcError as e:
                codes.append(e.code())
    assert codes.count(grpc.StatusCode.OK) == 2
    assert codes.count(grpc.StatusCode.RESOURCE_EXHAUSTED) == 4


def test_graceful_shutdown_finishes_in_flight_requests(server_factory):
    engine = fake_engine(delay=0.5)
    server = server_factory(engine, grace=5)
    channel = grpc.insecure_channel(server.target)
    stub = pb_grpc.ClassifierStub(channel)
    in_flight = stub.Classify.future(client.request(ENGLISH, QUESTIONS), timeout=10)
    time.sleep(0.1)
    stopper = threading.Thread(target=server.stop)
    stopper.start()
    response = in_flight.result()  # completes despite the shutdown
    assert set(response.answers) == {q.id for q in QUESTIONS}
    stopper.join(10)
    assert not stopper.is_alive()
    with pytest.raises(grpc.RpcError) as e:
        stub.Classify(client.request(ENGLISH, QUESTIONS), timeout=2)
    assert e.value.code() == grpc.StatusCode.UNAVAILABLE
    channel.close()


def test_concurrent_unary_calls_batch_without_crossing(server_factory):
    engine = fake_engine(delay=0.02, max_batch_rows=64)
    server = server_factory(engine)
    states = ["%s case %d" % (ENGLISH if i % 3 else HINDI, i) for i in range(60)]
    with grpc.insecure_channel(server.target) as channel:
        stub = pb_grpc.ClassifierStub(channel)
        futures = [stub.Classify.future(client.request(s, QUESTIONS, request_id=str(i)), timeout=20)
                   for i, s in enumerate(states)]
        responses = [f.result() for f in futures]
    assert [r.request_id for r in responses] == [str(i) for i in range(60)]
    assert max(r.timing.batch_size for r in responses) > 1
    forwarded = [t for a in engine.agents.values() for c in a.calls for t in c["tags"]]
    expected = [tag_for(s, j) for s in states for j in range(len(QUESTIONS))]
    assert sorted(forwarded) == sorted(expected)  # every row exactly once
