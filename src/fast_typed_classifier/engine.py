"""Continuous batching over Laya checkpoints.

Each request goes through a three-stage pipeline, so the device is never idle waiting on Python:

1. prep pool: route (script and language detection) and tokenize the state against its questions
2. inference thread: collate and run the forward pass, batched with every other queued request
   for the same checkpoint -- whatever their question sets or languages, since each
   (state, question) pair is an independent row
3. decode thread: turn logits into typed answers, while the inference thread runs the next batch

The inference thread never waits for a batch to fill: when the device is free it takes whatever
is queued, so an idle server adds no latency and a busy one batches automatically.

Batching uses Laya's `Agent` internals (`_encode_state`, `_forward`, `_decode_answers`) because
`Agent.predict_batch` needs one question set and one language per call, and `Router.predict_batch`
drops the detected language that `Router.predict` uses for calibration. laya is pinned for this;
tests/test_real_models.py checks that answers match `Router.predict`.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from laya import Router
from laya.common import collate_items
from laya.router import normalise_name

from fast_typed_classifier.convert import QuestionSet, RequestError, State, build_question_set

log = logging.getLogger(__name__)


class ModelNotLoaded(Exception):
    """The request routed to a checkpoint this server did not load. Maps to FAILED_PRECONDITION."""


class Overloaded(Exception):
    """Too many requests in flight. Maps to RESOURCE_EXHAUSTED."""


class ShuttingDown(Exception):
    """The server is draining. Maps to UNAVAILABLE."""


@dataclass(frozen=True)
class Limits:
    # Most (state, question) rows in one forward pass.
    max_batch_rows: int
    # Most padded tokens (rows x longest row) in one forward pass.
    max_batch_tokens: int
    # Most requests admitted at once (tokenizing, queued or running); more get RESOURCE_EXHAUSTED.
    max_inflight: int = 1024
    # How long an idle inference thread waits for more requests before starting a batch.
    # 0 starts at once; requests that arrive during a forward pass still batch together.
    max_wait_ms: float = 0.0
    # Requests are merged into one forward pass only if the padding that adds costs at most this
    # fraction more tokens than running them in separate passes.
    max_padding: float = 0.25


# On CPU and Apple GPUs the forward pass is compute-bound even for one request (measured on an
# M1: time per request is flat from batch 1 to 32), so batching adds little throughput, makes each
# request wait for the whole batch, and every padding token costs as much as a real one. Small,
# tightly length-matched batches were fastest there. On CUDA, batching is several times faster per
# request and a GPU has headroom for some padding, so batches are bounded mainly by memory.
DEVICE_LIMITS = {
    "cpu": dict(max_batch_rows=8, max_batch_tokens=4096, max_padding=0.1),
    "mps": dict(max_batch_rows=8, max_batch_tokens=4096, max_padding=0.1),
    "cuda": dict(max_batch_rows=512, max_batch_tokens=32768, max_padding=0.5),
}


def default_limits(device_type: str, **overrides: Any) -> Limits:
    values = dict(DEVICE_LIMITS.get(device_type, DEVICE_LIMITS["cpu"]))
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Limits(**values)


@dataclass
class Result:
    answers: Dict[str, Dict[str, Any]]
    routing: Dict[str, Any]
    input_tokens: int
    queue_ms: float
    inference_ms: float
    batch_size: int


@dataclass(eq=False)
class _Job:
    model: str
    decision: Dict[str, Any]
    qset: QuestionSet
    items: List[Dict[str, Any]]  # one encoded row per question, from Agent._encode_state
    lang: Optional[str]          # language for calibration temperatures, as Router.predict passes it
    n_tokens: int
    arrived: float
    future: Optional[asyncio.Future] = None


class Engine:
    def __init__(self, router: Router, agents: Dict[str, Any], limits: Limits, prep_threads: int = 4):
        if not agents:
            raise ValueError("load at least one checkpoint")
        self.router = router
        self.agents = agents
        self.limits = limits
        self.device = str(next(iter(agents.values())).device)
        self._prep = ThreadPoolExecutor(prep_threads, thread_name_prefix="ftc-prep")
        self._infer = ThreadPoolExecutor(1, thread_name_prefix="ftc-infer")
        # Separate from prep, so finished results are not stuck behind a burst of new requests.
        self._decode = ThreadPoolExecutor(1, thread_name_prefix="ftc-decode")
        # Fast tokenizers are not safe to call from several threads at once.
        self._encode_locks = {name: threading.Lock() for name in agents}
        self._queue: "collections.deque[_Job]" = collections.deque()
        self._wake: Optional[asyncio.Event] = None
        self._worker: Optional[asyncio.Task] = None
        self._finishing: set = set()
        self._inflight = 0
        self._closing = False

    @classmethod
    def load(cls, models: Sequence[str], device: Optional[str] = None, *, default_route: str = "english",
             limits: Optional[Limits] = None, prep_threads: int = 4, **limit_overrides: Any) -> "Engine":
        """Load the named checkpoints ("english", "multilingual", "typed-decisions")."""
        router = Router(device=device, max_loaded=len(models), default=default_route)
        agents = {}
        for name in models:
            key = normalise_name(name)
            t0 = time.perf_counter()
            agents[key] = router.load(key)
            log.info("loaded %s on %s in %.1fs", key, agents[key].device, time.perf_counter() - t0)
        device_type = next(iter(agents.values())).device.type
        return cls(router, agents, limits or default_limits(device_type, **limit_overrides), prep_threads)

    # ------------------------------------------------------------------ public API

    def route(self, state: State, questions: Dict[str, Any], model: Optional[str] = None,
              lang: Optional[str] = None) -> Dict[str, Any]:
        try:
            decision = self.router.route(state, questions, model=model or None, lang=lang or None)
        except (ValueError, KeyError) as e:
            raise RequestError(str(e).strip("'\"")) from None
        if (decision["model"] == "english" and "english" not in self.agents and "multilingual" in self.agents
                and not model):
            # multilingual reads English too (the reverse is not true), so a multilingual-only
            # server -- smaller and faster -- still answers automatically routed English text.
            decision = dict(decision, model="multilingual",
                            repo=self.router.route("", None, model="multilingual")["repo"],
                            reason="%s; english is not loaded, so multilingual answers" % decision["reason"])
        return decision

    def check_admission(self) -> None:
        """Raise if a new request would be refused. Cheap: call it before parsing a request, so
        an overloaded server spends as little as possible on requests it will reject."""
        if self._closing or self._worker is None:
            raise ShuttingDown("server is not accepting requests")
        if self._inflight >= self.limits.max_inflight:
            raise Overloaded("server has %d requests in flight; retry with backoff" % self._inflight)

    async def classify(self, state: State, qset: QuestionSet, *, model: Optional[str] = None,
                       lang: Optional[str] = None, max_len: Optional[int] = None) -> Result:
        self.check_admission()
        self._inflight += 1
        try:
            loop = asyncio.get_running_loop()
            arrived = time.perf_counter()
            job = await loop.run_in_executor(self._prep, self._prepare, state, qset, model, lang,
                                             max_len or None, arrived)
            if self._worker is None:  # stopped while this request was tokenizing
                raise ShuttingDown("server stopped before this request ran")
            job.future = loop.create_future()
            self._queue.append(job)
            self._wake.set()
            return await job.future
        finally:
            self._inflight -= 1

    async def start(self) -> None:
        self._wake = asyncio.Event()
        self._worker = asyncio.create_task(self._run(), name="ftc-batcher")

    def begin_shutdown(self) -> None:
        """Refuse new requests; queued and running ones still complete."""
        self._closing = True

    async def stop(self) -> None:
        self._closing = True
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None
        if self._finishing:
            await asyncio.gather(*list(self._finishing), return_exceptions=True)
        while self._queue:
            job = self._queue.popleft()
            if not job.future.done():
                job.future.set_exception(ShuttingDown("server stopped before this request ran"))
        self._infer.shutdown(wait=True, cancel_futures=True)
        self._decode.shutdown(wait=True, cancel_futures=True)
        self._prep.shutdown(wait=True, cancel_futures=True)

    def warm_up(self) -> None:
        """Run each checkpoint once at a small and a full batch shape, so the first real requests
        do not pay one-off kernel selection and allocation costs (seconds on GPU)."""
        from fast_typed_classifier.v1 import classifier_pb2 as pb

        qset = build_question_set([
            pb.Question(id="c", instructions="Which team?", choice=pb.ChoiceQuestion(
                options=[pb.ChoiceOption(label="a"), pb.ChoiceOption(label="b")])),
            pb.Question(id="s", instructions="How urgent?", score=pb.ScoreQuestion(levels=["low", "high"])),
            pb.Question(id="n", instructions="Is this a question?", noul=pb.NoulQuestion()),
        ])
        per_request = len(qset.ids)
        for name, agent in self.agents.items():
            t0 = time.perf_counter()
            sizes = sorted({1, max(1, self.limits.max_batch_rows // per_request)})
            for size in sizes:
                jobs = [self._prepare("Warm-up request number %d." % i, qset, name, None, None, 0.0)
                        for i in range(size)]
                outputs = self._forward_jobs(agent, jobs)
                for result in self._decode_jobs(agent, jobs, outputs):
                    if isinstance(result, BaseException):
                        raise result
            log.info("warmed up %s (batches of %s requests) in %.1fs", name, sizes, time.perf_counter() - t0)

    def info(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "models": [{"name": name, "repo": self.router.route("", None, model=name)["repo"],
                        "max_len": int(agent.cfg.get("max_len", 512))} for name, agent in self.agents.items()],
            "limits": self.limits,
        }

    # ------------------------------------------------------------------ pipeline stages

    def _prepare(self, state: State, qset: QuestionSet, model: Optional[str], lang: Optional[str],
                 max_len: Optional[int], arrived: float) -> _Job:
        """Stage 1 (prep pool): route and tokenize."""
        decision = self.route(state, qset.questions, model=model, lang=lang)
        name = decision["model"]
        agent = self.agents.get(name)
        if agent is None:
            raise ModelNotLoaded("request routed to %r (%s), which this server did not load; loaded: %s"
                                 % (name, decision.get("reason"), ", ".join(self.agents)))
        detection = decision.get("detection") or {}
        try:
            with self._encode_locks[name]:
                items = agent._encode_state(state, list(qset.ids), qset.internal, max_len=max_len)
        except ValueError as e:
            raise RequestError(str(e)) from None
        return _Job(model=name, decision=decision, qset=qset, items=items,
                    lang=lang or detection.get("language"),
                    n_tokens=sum(len(it["ids"]) for it in items), arrived=arrived)

    def _take(self) -> Tuple[Optional[str], List[_Job]]:
        """Pop the next batch: the oldest request's checkpoint, and every queued request for the
        same checkpoint that fits the batch limits, oldest first."""
        q = self._queue
        while q and q[0].future.done():  # cancelled or timed out while queued
            q.popleft()
        if not q:
            return None, []
        name = q[0].model
        taken: List[_Job] = []
        skipped: List[_Job] = []
        rows = tokens = 0
        while q:
            job = q[0]
            if job.future.done():
                q.popleft()
                continue
            if job.model != name:
                skipped.append(q.popleft())
                continue
            if taken and (rows + len(job.items) > self.limits.max_batch_rows
                          or tokens + job.n_tokens > self.limits.max_batch_tokens):
                break
            taken.append(q.popleft())
            rows += len(job.items)
            tokens += job.n_tokens
        q.extendleft(reversed(skipped))
        return name, taken

    def _chunks(self, jobs: List[_Job]) -> List[List[int]]:
        """Group the rows of `jobs` (indexed as flattened in job order) into forward passes.

        A request's rows stay together, as in `Router.predict`: splitting them costs a whole extra
        pass (every pass streams all the weights), which measured slower than their padding.
        Requests are sorted by their longest row and merged into one pass unless that would exceed
        max_batch_rows, max_batch_tokens padded tokens, or cost more than max_padding above running
        them in separate passes. Only a request too big for the budgets alone is split by rows.
        """
        limits = self.limits
        units = []  # (longest row, first row index, row lengths)
        offset = 0
        for job in jobs:
            lengths = [len(item["ids"]) for item in job.items]
            units.append((max(lengths), offset, lengths))
            offset += len(lengths)
        units.sort(key=lambda u: u[0])

        chunks: List[List[int]] = []
        current: List[int] = []
        separate = 0  # padded tokens if the requests in `current` ran in separate passes
        for longest, first, lengths in units:
            n = len(lengths)
            alone = n * longest
            if n > limits.max_batch_rows or alone > limits.max_batch_tokens:
                if current:
                    chunks.append(current)
                    current, separate = [], 0
                chunks.extend(self._split_rows(first, lengths))
                continue
            merged = (len(current) + n) * longest  # sorted, so this request's longest row is the longest
            if current and (len(current) + n > limits.max_batch_rows or merged > limits.max_batch_tokens
                            or merged > (separate + alone) * (1.0 + limits.max_padding)):
                chunks.append(current)
                current, separate = [], 0
            current.extend(range(first, first + n))
            separate += alone
        if current:
            chunks.append(current)
        return chunks

    def _split_rows(self, first: int, lengths: List[int]) -> List[List[int]]:
        """One request too big for a single pass: its rows by length, within the row and token budgets."""
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        chunks: List[List[int]] = []
        current: List[int] = []
        for i in order:
            if current and (len(current) + 1 > self.limits.max_batch_rows
                            or (len(current) + 1) * lengths[i] > self.limits.max_batch_tokens):
                chunks.append(current)
                current = []
            current.append(first + i)
        if current:
            chunks.append(current)
        return chunks

    def _forward_jobs(self, agent: Any, jobs: List[_Job]) -> List[Tuple[list, list]]:
        """Stage 2 (inference thread): forward every row of `jobs`; per job, its logit and action rows."""
        rows = [item for job in jobs for item in job.items]
        logits_rows: List[Any] = [None] * len(rows)
        act_rows: List[Any] = [None] * len(rows)
        with torch.inference_mode():
            for chunk in self._chunks(jobs):
                batch = collate_items([[rows[r] for r in chunk]], agent.tok.pad_token_id)
                logits, act = agent._forward(batch)
                for i, r in enumerate(chunk):
                    logits_rows[r] = logits[i]
                    act_rows[r] = act[i]
        out = []
        pos = 0
        for job in jobs:
            n = len(job.items)
            out.append((logits_rows[pos:pos + n], act_rows[pos:pos + n]))
            pos += n
        return out

    def _decode_jobs(self, agent: Any, jobs: List[_Job], outputs: List[Any]) -> List[Any]:
        """Stage 3 (decode thread): typed answers per job, or the exception that job failed with."""
        results: List[Any] = []
        for job, out in zip(jobs, outputs):
            if isinstance(out, BaseException):
                results.append(out)
                continue
            try:
                logit_rows, act_rows = out
                # Rows from different forward passes can be padded to different option counts;
                # decoding reads only each row's own options.
                width = max(len(x) for x in logit_rows)
                logits = np.zeros((len(logit_rows), width), dtype=logit_rows[0].dtype)
                for i, x in enumerate(logit_rows):
                    logits[i, :len(x)] = x
                results.append(agent._decode_answers(logits, np.stack(act_rows), job.items, list(job.qset.ids),
                                                     job.qset.internal, 0, lang=job.lang))
            except Exception as e:  # noqa: BLE001 - reported per request
                results.append(e)
        return results

    # ------------------------------------------------------------------ batching loop

    async def _run(self) -> None:
        wait_s = self.limits.max_wait_ms / 1000.0
        while True:
            if not self._queue:
                self._wake.clear()
                await self._wake.wait()
                if wait_s:
                    await asyncio.sleep(wait_s)
            jobs: List[_Job] = []
            try:
                name, jobs = self._take()
                if jobs:
                    await self._step(self.agents[name], jobs)
            except asyncio.CancelledError:
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(ShuttingDown("server stopped during this request"))
                raise
            except Exception as exc:  # noqa: BLE001 - keep serving; fail only this batch
                log.exception("batching loop error")
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(exc)

    async def _step(self, agent: Any, jobs: List[_Job]) -> None:
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        try:
            outputs = await loop.run_in_executor(self._infer, self._forward_jobs, agent, jobs)
        except Exception as exc:  # noqa: BLE001
            if len(jobs) == 1:
                outputs = [exc]
            else:
                # Rerun each request alone, so one bad request cannot fail its batch-mates.
                log.warning("batch of %d failed (%s); retrying one by one", len(jobs), exc)
                outputs = []
                for job in jobs:
                    try:
                        outputs.extend(await loop.run_in_executor(self._infer, self._forward_jobs, agent, [job]))
                    except Exception as e:  # noqa: BLE001
                        outputs.append(e)
        # Decode on its own thread so the inference thread can start the next batch now.
        task = asyncio.create_task(self._finish(agent, jobs, outputs, started))
        self._finishing.add(task)
        task.add_done_callback(self._finishing.discard)

    async def _finish(self, agent: Any, jobs: List[_Job], outputs: List[Any], started: float) -> None:
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(self._decode, self._decode_jobs, agent, jobs, outputs)
        except Exception as exc:  # noqa: BLE001 - e.g. the pool shut down
            results = [exc] * len(jobs)
        ended = time.perf_counter()
        for job, result in zip(jobs, results):
            if job.future.done():
                continue
            if isinstance(result, BaseException):
                job.future.set_exception(result)
            else:
                job.future.set_result(Result(
                    answers=result, routing=job.decision, input_tokens=job.n_tokens,
                    queue_ms=(started - job.arrived) * 1000.0, inference_ms=(ended - started) * 1000.0,
                    batch_size=len(jobs)))
