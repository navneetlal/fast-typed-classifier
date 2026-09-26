"""Conversion between the gRPC messages and Laya's dict formats."""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple, Union

from laya.agent import Agent

from fast_typed_classifier.v1 import classifier_pb2 as pb

State = Union[str, dict, list]

# The longest context any shipped checkpoint reads (laya-multilingual).
MAX_LEN_LIMIT = 8192

_TYPE_ENUM = {
    "choice": pb.QUESTION_TYPE_CHOICE,
    "score": pb.QUESTION_TYPE_SCORE,
    "noul": pb.QUESTION_TYPE_NOUL,
}


class RequestError(ValueError):
    """A request that cannot be answered as sent. Maps to INVALID_ARGUMENT."""


@dataclass(frozen=True)
class QuestionSet:
    """A validated question list, in both the forms Laya needs.

    `questions` is the dict `Router.predict` takes (and `Router.route` reads for workflow
    detection); `internal` is `Agent._to_internal` of each, which every checkpoint shares.
    """

    questions: Dict[str, Dict[str, Any]]
    ids: Tuple[str, ...]
    internal: Dict[str, Dict[str, Any]]


def parse_state(request: pb.ClassifyRequest) -> State:
    if request.max_len > MAX_LEN_LIMIT:
        raise RequestError("max_len %d is above the limit of %d" % (request.max_len, MAX_LEN_LIMIT))
    which = request.WhichOneof("state")
    if which == "text":
        return request.text
    if which == "json":
        try:
            state = json.loads(request.json)
        except ValueError as e:
            raise RequestError("state json is not valid JSON: %s" % e) from None
        if not isinstance(state, (dict, list)):
            raise RequestError("state json must be an object or an array, got %s" % type(state).__name__)
        return state
    raise RequestError("set the state: either text or json")


def _question_to_laya(q: pb.Question) -> Dict[str, Any]:
    kind = q.WhichOneof("kind")
    if kind is None:
        raise RequestError("question %r: set one of choice, score or noul" % q.id)
    qdef: Dict[str, Any] = {"type": kind, "instructions": q.instructions}
    if kind == "choice":
        labels = [o.label for o in q.choice.options]
        if len(set(labels)) != len(labels):
            raise RequestError("question %r: choice labels must be unique" % q.id)
        if any(not label for label in labels):
            raise RequestError("question %r: choice labels must be non-empty" % q.id)
        # Laya renders an empty description as the bare label, the same as the list form.
        qdef["criteria"] = {o.label: o.description for o in q.choice.options}
    elif kind == "score":
        qdef["criteria"] = list(q.score.levels)
    else:
        n = q.noul
        crit = {}
        if n.false_description:
            crit["false"] = n.false_description
        if n.true_description:
            crit["true"] = n.true_description
        if crit:
            qdef["criteria"] = crit
        if n.true_label or n.false_label:
            qdef["labels"] = {"false": n.false_label, "true": n.true_label}
    return qdef


def build_question_set(questions: Iterable[pb.Question]) -> QuestionSet:
    laya_questions: Dict[str, Dict[str, Any]] = {}
    for q in questions:
        if not q.id:
            raise RequestError("every question needs an id")
        if q.id in laya_questions:
            raise RequestError("question id %r is used twice" % q.id)
        laya_questions[q.id] = _question_to_laya(q)
    if not laya_questions:
        raise RequestError("send at least one question")
    try:
        for qid, qdef in laya_questions.items():
            Agent._check_question(qid, qdef)
        internal = {qid: Agent._to_internal(qdef) for qid, qdef in laya_questions.items()}
    except ValueError as e:
        raise RequestError(str(e)) from None
    return QuestionSet(questions=laya_questions, ids=tuple(laya_questions), internal=internal)


class QuestionSetCache:
    """LRU cache of validated question sets, keyed by their serialized bytes.

    Services usually send the same few question sets over and over, so this skips rebuilding
    and revalidating them on every request. Thread-safe.
    """

    def __init__(self, maxsize: int = 1024):
        self.maxsize = maxsize
        self._items: "OrderedDict[bytes, QuestionSet]" = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(questions: Iterable[pb.Question]) -> bytes:
        parts = []
        for q in questions:
            b = q.SerializeToString(deterministic=True)
            parts.append(len(b).to_bytes(4, "little"))
            parts.append(b)
        return b"".join(parts)

    def get(self, questions: Iterable[pb.Question]) -> QuestionSet:
        key = self._key(questions)
        with self._lock:
            qset = self._items.get(key)
            if qset is not None:
                self._items.move_to_end(key)
                return qset
        qset = build_question_set(questions)
        with self._lock:
            self._items[key] = qset
            if len(self._items) > self.maxsize:
                self._items.popitem(last=False)
        return qset


def routing_to_proto(decision: Dict[str, Any], out: pb.Routing) -> None:
    out.model = decision["model"]
    out.reason = decision.get("reason") or ""
    detection = decision.get("detection")
    if detection:
        out.detected_script = detection.get("script") or ""
        out.detected_language = detection.get("language") or ""


def answers_to_proto(answers: Dict[str, Dict[str, Any]], out: pb.ClassifyResponse) -> None:
    """Copy Laya's answer dicts into `out.answers`."""
    for qid, a in answers.items():
        msg = out.answers[qid]
        t = a["type"]
        msg.type = _TYPE_ENUM[t]
        if t == "choice":
            msg.choice = a["choice"]
            msg.probabilities.extend(a["probabilities"].values())
        elif t == "score":
            msg.score = a["score"]
            msg.probabilities.extend(a["probabilities"].values())
        else:
            p_true = a["noul"]
            msg.noul = p_true
            msg.probabilities.extend((round(1.0 - p_true, 4), p_true))
        msg.confidence = a["confidence"]
        msg.answer_confidence = a["answer_confidence"]
        msg.act_probability = a["action"]["act_probability"]


def answers_from_proto(response: pb.ClassifyResponse) -> Dict[str, Dict[str, Any]]:
    """The inverse of `answers_to_proto`, for clients and tests: plain dicts keyed by question id."""
    out: Dict[str, Dict[str, Any]] = {}
    for qid, a in response.answers.items():
        value = a.WhichOneof("value")
        out[qid] = {
            "type": value,
            value: getattr(a, value),
            "probabilities": list(a.probabilities),
            "confidence": a.confidence,
            "answer_confidence": a.answer_confidence,
            "act_probability": a.act_probability,
        }
    return out

