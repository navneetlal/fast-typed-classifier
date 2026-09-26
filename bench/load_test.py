"""Load test: throughput and latency percentiles against a running server.

    .venv/bin/python bench/load_test.py --concurrency 8 --duration 30
    .venv/bin/python bench/load_test.py --mode stream --concurrency 32 --mix multilingual
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import itertools
import statistics
import sys
import time
from pathlib import Path

import grpc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fast_typed_classifier import client  # noqa: E402
from fast_typed_classifier.v1 import classifier_pb2 as pb  # noqa: E402
from fast_typed_classifier.v1 import classifier_pb2_grpc as pb_grpc  # noqa: E402

TEXTS = {
    "english": [
        {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
         "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."},
        "The export button has returned a 500 error since this morning and my whole team is blocked.",
        "Could you send me pricing for the enterprise plan with 200 seats?",
        "Thanks for the quick fix yesterday, everything works now.",
    ],
    "multilingual": [
        "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।",
        "La aplicación se cierra cada vez que abro la configuración.",
        "Der Kunde wurde zweimal belastet und möchte eine Rückerstattung.",
        "Je veux annuler mon abonnement immédiatement.",
    ],
}


def percentile(sorted_values, p):
    if not sorted_values:
        return float("nan")
    k = (len(sorted_values) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def build_requests(mix: str):
    texts = TEXTS["english"] + TEXTS["multilingual"] if mix == "mixed" else TEXTS[mix]
    return [client.request(t, client.SUPPORT_TRIAGE) for t in texts]


class Stats:
    def __init__(self):
        self.latencies = []
        self.queue_ms = []
        self.inference_ms = []
        self.batch_sizes = collections.Counter()
        self.errors = collections.Counter()

    def ok(self, latency_ms, response):
        if response.HasField("error"):
            self.errors[grpc.StatusCode(self._code(response.error.code)).name] += 1
            return
        self.latencies.append(latency_ms)
        self.queue_ms.append(response.timing.queue_ms)
        self.inference_ms.append(response.timing.inference_ms)
        self.batch_sizes[response.timing.batch_size] += 1

    @staticmethod
    def _code(value):
        return next(c for c in grpc.StatusCode if c.value[0] == value)


async def run_unary(stub, requests, concurrency, deadline, stats, timeout, backoff):
    counter = itertools.count()

    async def worker():
        while time.perf_counter() < deadline:
            req = requests[next(counter) % len(requests)]
            t0 = time.perf_counter()
            try:
                response = await stub.Classify(req, timeout=timeout)
            except grpc.aio.AioRpcError as e:
                stats.errors[e.code().name] += 1
                if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                    await asyncio.sleep(backoff)
                continue
            stats.ok((time.perf_counter() - t0) * 1000, response)

    await asyncio.gather(*(worker() for _ in range(concurrency)))


async def run_stream(stub, requests, concurrency, deadline, stats, timeout, backoff):
    """One bidirectional stream, keeping `concurrency` requests in flight."""
    window = asyncio.Semaphore(concurrency)
    sent = {}
    counter = itertools.count()

    async def produce():
        while time.perf_counter() < deadline:
            await window.acquire()
            i = next(counter)
            req = requests[i % len(requests)].__deepcopy__()
            req.request_id = str(i)
            sent[req.request_id] = time.perf_counter()
            yield req

    call = stub.ClassifyStream(produce(), timeout=timeout + (deadline - time.perf_counter()))
    async def release_after(delay):
        await asyncio.sleep(delay)
        window.release()

    async for response in call:
        stats.ok((time.perf_counter() - sent.pop(response.request_id)) * 1000, response)
        if response.error.code == grpc.StatusCode.RESOURCE_EXHAUSTED.value[0]:
            asyncio.ensure_future(release_after(backoff))  # back off before refilling the slot
        else:
            window.release()


async def main_async(args):
    requests = build_requests(args.mix)
    async with grpc.aio.insecure_channel(args.target) as channel:
        stub = pb_grpc.ClassifierStub(channel)
        info = await stub.GetServerInfo(pb.GetServerInfoRequest())
        # warm connection and server
        await stub.Classify(requests[0], timeout=args.timeout)
        stats = Stats()
        t0 = time.perf_counter()
        deadline = t0 + args.duration
        runner = run_unary if args.mode == "unary" else run_stream
        await runner(stub, requests, args.concurrency, deadline, stats, args.timeout, args.backoff_ms / 1000)
        elapsed = time.perf_counter() - t0

    lat = sorted(stats.latencies)
    n = len(lat)
    print("server: device=%s models=%s max_batch_rows=%d max_batch_tokens=%d" % (
        info.device, ",".join(m.name for m in info.models), info.max_batch_rows, info.max_batch_tokens))
    print("load:   mode=%s concurrency=%d mix=%s duration=%.1fs (4 questions per request)" % (
        args.mode, args.concurrency, args.mix, elapsed))
    print("result: %d ok, %.2f req/s (%.1f questions/s)%s" % (
        n, n / elapsed, 4 * n / elapsed,
        "  errors: %s" % dict(stats.errors) if stats.errors else ""))
    if n:
        print("latency ms: p50 %.0f  p90 %.0f  p99 %.0f  max %.0f  mean %.0f" % (
            percentile(lat, 50), percentile(lat, 90), percentile(lat, 99), lat[-1], statistics.mean(lat)))
        print("server ms:  queue mean %.0f  inference mean %.0f  batch sizes %s" % (
            statistics.mean(stats.queue_ms), statistics.mean(stats.inference_ms),
            dict(sorted(stats.batch_sizes.items()))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="localhost:50051")
    p.add_argument("--mode", choices=["unary", "stream"], default="unary")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--mix", choices=["english", "multilingual", "mixed"], default="mixed")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--backoff-ms", type=float, default=100.0, help="wait after RESOURCE_EXHAUSTED before retrying")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
