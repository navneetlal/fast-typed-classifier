"""Shared test helpers: a fake Laya agent and an in-process gRPC server."""
from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
if (ROOT / ".hf-cache" / "hub" / "models--convaiinnovations--laya").is_dir():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

from laya import Router  # noqa: E402
from laya.common import serialize_state  # noqa: E402

from fast_typed_classifier.engine import Engine, Limits  # noqa: E402


def tag_for(state, question_index: int) -> int:
    """The tag FakeAgent gives the row for (state, question): lets tests check that every
    request got its own rows back, whatever batch it ran in."""
    return (zlib.crc32(serialize_state(state).encode()) % 100_000) * 100 + question_index


class _Tok:
    pad_token_id = 0


class FakeAgent:
    """Stands in for laya.Agent: one row per question, token count = characters of the state,
    and logits that carry the row's tag, so decoding reveals which row an answer came from."""

    def __init__(self, name: str, delay: float = 0.0):
        self.name = name
        self.device = torch.device("cpu")
        self.cfg = {"max_len": 512}
        self.tok = _Tok()
        self.delay = delay            # seconds per forward pass
        self.fail_tags: set = set()   # a forward pass containing one of these rows raises
        self.calls: List[Dict] = []   # one entry per forward pass
        self._lock = threading.Lock()

    def _encode_state(self, state, ids, internal, max_len=None, head_max_len=None):
        text = serialize_state(state)
        length = min(len(text), (max_len or self.cfg["max_len"]) - 1)
        items = []
        for j, qid in enumerate(ids):
            q = internal[qid]
            k = len(q["crit"]) if q["t"] in ("choice", "score") else 2
            items.append({"ids": [tag_for(state, j)] + [7] * max(length, 1),
                          "markers": list(range(1, k + 1)),
                          "qtype": {"choice": 0, "score": 1, "noul": 2}[q["t"]]})
        return items

    def _forward(self, b):
        tags = b["input_ids"][:, 0].tolist()
        with self._lock:
            self.calls.append({"rows": b["input_ids"].shape[0], "padded_len": b["input_ids"].shape[1],
                               "tags": tags, "thread": threading.current_thread().name})
        if self.delay:
            time.sleep(self.delay)
        if self.fail_tags.intersection(tags):
            raise RuntimeError("injected failure")
        n, kmax = b["marker_pos"].shape
        logits = np.zeros((n, kmax), dtype=np.float32)
        logits[:, 0] = np.array(tags, dtype=np.float32)
        return logits, np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (n, 1))

    def _decode_answers(self, logits, act, items, ids, internal, offset, lang=None):
        answers = {}
        for j, qid in enumerate(ids):
            r = offset + j
            q = internal[qid]
            tag = int(logits[r, 0])
            common = {"confidence": 0.5, "answer_confidence": 0.5, "action": {"act_probability": float(act[r, 0])},
                      "tag": tag, "lang": lang, "model": self.name}
            if q["t"] == "choice":
                keys = list(q["crit"])
                answers[qid] = dict(common, type="choice", choice=keys[0],
                                    probabilities={k: (1.0 if i == 0 else 0.0) for i, k in enumerate(keys)})
            elif q["t"] == "score":
                answers[qid] = dict(common, type="score", score=0.0,
                                    probabilities={str(i): (1.0 if i == 0 else 0.0) for i in range(len(q["crit"]))})
            else:
                answers[qid] = dict(common, type="noul", noul=0.25)
        return answers

    def forwarded_tags(self) -> set:
        return {t for call in self.calls for t in call["tags"]}


def fake_engine(models=("english", "multilingual"), delay: float = 0.0, **limits) -> Engine:
    values = dict(max_batch_rows=8, max_batch_tokens=4096, max_inflight=1024, max_wait_ms=0.0)
    values.update(limits)
    agents = {name: FakeAgent(name, delay) for name in models}
    return Engine(Router(max_loaded=len(models)), agents, Limits(**values), prep_threads=4)


class ServerThread:
    """Runs `serve()` on its own event loop in a background thread."""

    def __init__(self, engine: Engine, grace: float = 5.0):
        from fast_typed_classifier.server import serve

        self.engine = engine
        self._serve = serve
        self._grace = grace
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._main, daemon=True)
        self.port: Optional[int] = None

    def _main(self):
        asyncio.set_event_loop(self._loop)
        self._stop = asyncio.Event()

        def on_ready(port: int) -> None:
            self.port = port
            self._ready.set()

        async def run():
            await self._serve(self.engine, "127.0.0.1", 0, self._grace, on_ready, self._stop)

        try:
            self._loop.run_until_complete(run())
        except BaseException as e:  # noqa: BLE001
            self._error = e
            self._ready.set()

    def start(self) -> "ServerThread":
        self._thread.start()
        self._ready.wait(60)
        if self._error:
            raise self._error
        return self

    @property
    def target(self) -> str:
        return "127.0.0.1:%d" % self.port

    def stop(self, timeout: float = 30.0) -> None:
        if self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._stop.set)
            self._thread.join(timeout)
        if self._error:
            raise self._error


@contextlib.contextmanager
def running_server(engine: Engine, grace: float = 5.0):
    server = ServerThread(engine, grace).start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def server_factory():
    servers = []

    def make(engine: Engine, grace: float = 5.0) -> ServerThread:
        s = ServerThread(engine, grace).start()
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.stop()
