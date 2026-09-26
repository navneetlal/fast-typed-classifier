"""Client helpers, and a small CLI: `python -m fast_typed_classifier.client "text to classify"`.

    import grpc
    from fast_typed_classifier import client
    from fast_typed_classifier.v1 import classifier_pb2_grpc

    stub = classifier_pb2_grpc.ClassifierStub(grpc.insecure_channel("localhost:50051"))
    response = stub.Classify(client.request(
        {"subject": "Duplicate charge", "body": "We were billed twice, please refund."},
        [client.choice("department", "Which department should handle this?",
                       {"billing": "invoices, payments, refunds", "other": "everything else"}),
         client.noul("refund_requested", "Does the user ask for a refund?")]))
    response.answers["department"].choice  # "billing"
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Iterable, List, Union

from fast_typed_classifier.v1 import classifier_pb2 as pb


def choice(id: str, instructions: str, options: Union[Dict[str, str], Iterable[str]]) -> pb.Question:
    """A choice question. `options` maps label -> description, or is a list of labels."""
    items = options.items() if isinstance(options, dict) else ((label, "") for label in options)
    return pb.Question(id=id, instructions=instructions, choice=pb.ChoiceQuestion(
        options=[pb.ChoiceOption(label=label, description=desc or "") for label, desc in items]))


def score(id: str, instructions: str, levels: Iterable[str]) -> pb.Question:
    """A score question; `levels` are descriptions, level 0 first."""
    return pb.Question(id=id, instructions=instructions, score=pb.ScoreQuestion(levels=list(levels)))


def noul(id: str, instructions: str, true_description: str = "", false_description: str = "") -> pb.Question:
    """A yes/no question; the answer is P(true)."""
    return pb.Question(id=id, instructions=instructions, noul=pb.NoulQuestion(
        true_description=true_description, false_description=false_description))


def request(state: Union[str, dict, list], questions: Iterable[pb.Question] = (), *, request_id: str = "",
            model: str = "", lang: str = "", max_len: int = 0) -> pb.ClassifyRequest:
    """A ClassifyRequest for a text (str) or JSON (dict or list) state."""
    req = pb.ClassifyRequest(request_id=request_id, questions=list(questions), model=model, lang=lang,
                             max_len=max_len)
    if isinstance(state, str):
        req.text = state
    else:
        req.json = json.dumps(state, ensure_ascii=False)
    return req


SUPPORT_TRIAGE: List[pb.Question] = [
    choice("department", "Which department should handle this request?", {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "sales": "pricing, new contracts",
        "other": "everything else",
    }),
    score("urgency", "How urgent is this request?", ["not urgent", "soon", "critical deadline or blocking issue"]),
    noul("churn_risk", "Does the user threaten to cancel or leave?"),
    noul("refund_requested", "Does the user explicitly request a refund?"),
]


def main(argv=None) -> None:
    import grpc
    from google.protobuf.json_format import MessageToDict

    from fast_typed_classifier.v1 import classifier_pb2_grpc

    p = argparse.ArgumentParser(description="Classify a text with the support-triage questions.")
    p.add_argument("text", nargs="?", default="Hi, we were billed twice for March. Please refund the "
                                              "duplicate today or we will cancel our plan.")
    p.add_argument("--target", default="localhost:50051")
    p.add_argument("--model", default="")
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args(argv)

    with grpc.insecure_channel(args.target) as channel:
        stub = classifier_pb2_grpc.ClassifierStub(channel)
        try:
            response = stub.Classify(request(args.text, SUPPORT_TRIAGE, model=args.model), timeout=args.timeout)
        except grpc.RpcError as e:
            sys.exit("%s: %s" % (e.code().name, e.details()))
    print(json.dumps(MessageToDict(response, preserving_proto_field_name=True), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
