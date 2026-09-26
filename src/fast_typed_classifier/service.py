"""The Classifier gRPC service (grpc.aio)."""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Iterable, Optional, Tuple

import grpc
import laya
import torch

from fast_typed_classifier.convert import (
    QuestionSetCache,
    RequestError,
    answers_to_proto,
    parse_state,
    routing_to_proto,
)
from fast_typed_classifier.engine import Engine, ModelNotLoaded, Overloaded, ShuttingDown
from fast_typed_classifier.v1 import classifier_pb2 as pb
from fast_typed_classifier.v1 import classifier_pb2_grpc as pb_grpc

log = logging.getLogger(__name__)

_STATUS = {
    RequestError: grpc.StatusCode.INVALID_ARGUMENT,
    ModelNotLoaded: grpc.StatusCode.FAILED_PRECONDITION,
    Overloaded: grpc.StatusCode.RESOURCE_EXHAUSTED,
    ShuttingDown: grpc.StatusCode.UNAVAILABLE,
}


def _status_for(exc: BaseException) -> Tuple[grpc.StatusCode, str]:
    for cls, code in _STATUS.items():
        if isinstance(exc, cls):
            return code, str(exc)
    log.error("internal error", exc_info=exc)
    return grpc.StatusCode.INTERNAL, "internal error: %s: %s" % (type(exc).__name__, exc)


class ClassifierService(pb_grpc.ClassifierServicer):
    def __init__(self, engine: Engine):
        self.engine = engine
        self.questions = QuestionSetCache()

    async def _classify(self, request: pb.ClassifyRequest,
                        questions: Optional[Iterable[pb.Question]] = None) -> pb.ClassifyResponse:
        """Answer one request; engine and validation errors propagate."""
        self.engine.check_admission()
        state = parse_state(request)
        qset = self.questions.get(questions if questions is not None else request.questions)
        result = await self.engine.classify(state, qset, model=request.model, lang=request.lang,
                                            max_len=request.max_len)
        response = pb.ClassifyResponse(request_id=request.request_id, input_tokens=result.input_tokens)
        answers_to_proto(result.answers, response)
        routing_to_proto(result.routing, response.routing)
        response.timing.queue_ms = result.queue_ms
        response.timing.inference_ms = result.inference_ms
        response.timing.batch_size = result.batch_size
        return response

    async def _classify_item(self, request: pb.ClassifyRequest,
                             questions: Optional[Iterable[pb.Question]] = None) -> pb.ClassifyResponse:
        """Answer one request of a batch or stream; failures become the response's `error`."""
        try:
            return await self._classify(request, questions)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported to the caller per item
            code, message = _status_for(exc)
            return pb.ClassifyResponse(request_id=request.request_id,
                                       error=pb.Error(code=code.value[0], message=message))

    async def Classify(self, request, context):
        try:
            return await self._classify(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            code, message = _status_for(exc)
        await context.abort(code, message)

    async def ClassifyBatch(self, request, context):
        default = request.questions
        responses = await asyncio.gather(*(
            self._classify_item(item, item.questions if len(item.questions) else default)
            for item in request.requests))
        return pb.ClassifyBatchResponse(responses=responses)

    async def ClassifyStream(self, request_iterator, context) -> AsyncIterator[pb.ClassifyResponse]:
        done: asyncio.Queue = asyncio.Queue()
        pending: set = set()
        end_of_input = object()

        def on_done(task: asyncio.Task) -> None:
            pending.discard(task)
            done.put_nowait(task)

        async def read() -> None:
            try:
                async for request in request_iterator:
                    task = asyncio.create_task(self._classify_item(request))
                    pending.add(task)
                    task.add_done_callback(on_done)
            finally:
                done.put_nowait(end_of_input)

        reader = asyncio.create_task(read())
        input_open = True
        try:
            while input_open or pending or not done.empty():
                item = await done.get()
                if item is end_of_input:
                    input_open = False
                elif not item.cancelled():
                    yield item.result()
            if reader.done() and not reader.cancelled() and reader.exception() is not None:
                raise reader.exception()
        finally:
            reader.cancel()
            for task in list(pending):
                task.cancel()

    async def Route(self, request, context):
        try:
            state = parse_state(request)
            qset = self.questions.get(request.questions) if len(request.questions) else None
            decision = self.engine.route(state, qset.questions if qset else None,
                                         model=request.model, lang=request.lang)
        except RequestError as exc:
            code, message = _status_for(exc)
        else:
            routing = pb.Routing()
            routing_to_proto(decision, routing)
            return routing
        await context.abort(code, message)

    async def GetServerInfo(self, request, context):
        info = self.engine.info()
        limits = info["limits"]
        return pb.ServerInfo(
            device=info["device"],
            models=[pb.ModelInfo(**m) for m in info["models"]],
            max_batch_rows=limits.max_batch_rows,
            max_batch_tokens=limits.max_batch_tokens,
            max_inflight=limits.max_inflight,
            laya_version=laya.__version__,
            torch_version=torch.__version__,
        )
