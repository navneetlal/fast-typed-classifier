import pytest

from fast_typed_classifier import client
from fast_typed_classifier.convert import (
    QuestionSetCache,
    RequestError,
    answers_from_proto,
    answers_to_proto,
    build_question_set,
    parse_state,
)
from fast_typed_classifier.v1 import classifier_pb2 as pb


def test_parse_state_text_and_json():
    assert parse_state(pb.ClassifyRequest(text="hello")) == "hello"
    assert parse_state(pb.ClassifyRequest(text="")) == ""
    assert parse_state(pb.ClassifyRequest(json='{"subject": "Hi", "body": "Refund"}')) == {"subject": "Hi", "body": "Refund"}
    assert parse_state(pb.ClassifyRequest(json='[{"role": "user", "content": "hi"}]')) == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("request_, message", [
    (pb.ClassifyRequest(), "either text or json"),
    (pb.ClassifyRequest(json="{not json"), "not valid JSON"),
    (pb.ClassifyRequest(json="42"), "object or an array"),
    (pb.ClassifyRequest(json='"a string"'), "object or an array"),
])
def test_parse_state_rejects(request_, message):
    with pytest.raises(RequestError, match=message):
        parse_state(request_)


def test_question_set_matches_laya_formats():
    qset = build_question_set(client.SUPPORT_TRIAGE + [
        client.choice("lang", "Which language?", ["en", "de"]),
        pb.Question(id="spam", instructions="Is this spam?", noul=pb.NoulQuestion(
            true_description="unsolicited ads", false_description="a real message", true_label="A", false_label="B")),
    ])
    assert qset.ids == ("department", "urgency", "churn_risk", "refund_requested", "lang", "spam")
    q = qset.questions
    assert q["department"] == {"type": "choice", "instructions": "Which department should handle this request?",
                               "criteria": {"billing": "invoices, payments, refunds",
                                            "technical": "bugs, outages, system errors",
                                            "sales": "pricing, new contracts", "other": "everything else"}}
    assert q["urgency"]["criteria"] == ["not urgent", "soon", "critical deadline or blocking issue"]
    assert q["churn_risk"] == {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"}
    assert q["lang"]["criteria"] == {"en": "", "de": ""}
    assert q["spam"]["criteria"] == {"false": "a real message", "true": "unsolicited ads"}
    assert q["spam"]["labels"] == {"false": "B", "true": "A"}
    assert qset.internal["department"]["t"] == "choice"
    assert list(qset.internal["department"]["crit"]) == ["billing", "technical", "sales", "other"]


@pytest.mark.parametrize("questions, message", [
    ([], "at least one question"),
    ([pb.Question(instructions="x", noul=pb.NoulQuestion())], "needs an id"),
    ([client.noul("a", "x"), client.noul("a", "y")], "used twice"),
    ([pb.Question(id="a", instructions="x")], "set one of choice, score or noul"),
    ([pb.Question(id="a", instructions="x", choice=pb.ChoiceQuestion())], "at least one criterion"),
    ([client.choice("a", "x", ["yes", "yes"])], "unique"),
    ([client.choice("a", "x", ["", "no"])], "non-empty"),
    ([pb.Question(id="a", instructions="x", score=pb.ScoreQuestion())], "at least one level"),
    ([pb.Question(id="a", instructions="x", noul=pb.NoulQuestion(true_label="A"))], "labels must map"),
    ([pb.Question(id="a", instructions="x", noul=pb.NoulQuestion(true_label="A", false_label="A"))], "labels must map"),
])
def test_question_set_rejects(questions, message):
    with pytest.raises(RequestError, match=message):
        build_question_set(questions)


def test_question_set_cache_reuses_and_distinguishes():
    cache = QuestionSetCache(maxsize=2)
    a = cache.get(client.SUPPORT_TRIAGE)
    assert cache.get(list(client.SUPPORT_TRIAGE)) is a
    # Same questions in a different order are a different set: option order is positional.
    b = cache.get(list(reversed(client.SUPPORT_TRIAGE)))
    assert b is not a and b.ids == tuple(reversed(a.ids))
    # Moving one question's boundary must not collide (length-prefixed key).
    c1 = cache.get([client.noul("ab", "x")])
    c2 = cache.get([client.noul("a", "bx")])
    assert c1.ids != c2.ids
    assert len(cache._items) == 2  # LRU bound


def test_invalid_questions_are_not_cached():
    cache = QuestionSetCache()
    with pytest.raises(RequestError):
        cache.get([client.noul("a", "x"), client.noul("a", "x")])
    assert not cache._items


def test_answers_round_trip():
    laya_answers = {
        "department": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "other": 0.1},
                       "confidence": 0.8, "answer_confidence": 0.9, "action": {"act_probability": 1.0}},
        "urgency": {"type": "score", "score": 1.44, "legend": {"0": "a", "1": "b", "2": "c"},
                    "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}, "confidence": 0.14,
                    "answer_confidence": 0.6, "action": {"act_probability": 0.9}},
        "churn": {"type": "noul", "noul": 0.8248, "confidence": 0.8248, "answer_confidence": 0.8248,
                  "action": {"act_probability": 1.0}},
    }
    response = pb.ClassifyResponse()
    answers_to_proto(laya_answers, response)
    assert response.answers["department"].type == pb.QUESTION_TYPE_CHOICE
    assert response.answers["urgency"].WhichOneof("value") == "score"
    out = answers_from_proto(response)
    assert out["department"]["choice"] == "billing"
    assert out["department"]["probabilities"] == pytest.approx([0.9, 0.1])
    assert out["urgency"]["score"] == pytest.approx(1.44)
    assert out["urgency"]["probabilities"] == pytest.approx([0.1, 0.3, 0.6])
    assert out["churn"]["noul"] == pytest.approx(0.8248)
    assert out["churn"]["probabilities"] == pytest.approx([0.1752, 0.8248])
    assert out["urgency"]["act_probability"] == pytest.approx(0.9)


def test_client_request_builder():
    r = client.request({"body": "héllo"}, [client.noul("a", "x")], request_id="r1", model="english", lang="fr",
                       max_len=64)
    assert r.WhichOneof("state") == "json" and '"héllo"' in r.json  # not \u-escaped
    assert (r.request_id, r.model, r.lang, r.max_len) == ("r1", "english", "fr", 64)
    assert client.request("plain").text == "plain"
