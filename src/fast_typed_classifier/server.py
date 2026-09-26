"""Server entry point: `fast-typed-classifier` or `python -m fast_typed_classifier`."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional

log = logging.getLogger("fast_typed_classifier")

MODELS = ("english", "multilingual", "typed-decisions")

GRPC_OPTIONS = [
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ("grpc.max_send_message_length", 64 * 1024 * 1024),
    # Let clients keep idle connections alive with pings instead of reconnecting.
    ("grpc.keepalive_time_ms", 30_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.min_recv_ping_interval_without_data_ms", 10_000),
    ("grpc.http2.max_pings_without_data", 0),
]


def _env(name: str, default: str) -> str:
    return os.environ.get("FTC_" + name, default)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="fast-typed-classifier",
        description="gRPC typed-decision classifier backed by Laya. Every flag can also be set with "
                    "an FTC_<FLAG> environment variable, e.g. FTC_PORT=50051.")
    p.add_argument("--host", default=_env("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(_env("PORT", "50051")))
    p.add_argument("--device", default=_env("DEVICE", "auto"), choices=["auto", "cuda", "cpu", "mps"],
                   help="auto: cuda if available, else cpu (default: %(default)s)")
    p.add_argument("--models", default=_env("MODELS", "english,multilingual"),
                   help="comma-separated checkpoints to load: %s, or 'all' (default: %%(default)s). "
                        "Requests routed to a checkpoint that is not loaded fail with FAILED_PRECONDITION."
                        % ", ".join(MODELS))
    p.add_argument("--default-route", default=_env("DEFAULT_ROUTE", "english"), choices=["english", "multilingual"],
                   help="checkpoint for text whose language cannot be told (default: %(default)s)")
    p.add_argument("--max-batch-rows", type=int, default=_int_env("MAX_BATCH_ROWS"),
                   help="most (input, question) rows per forward pass (default: 512 on cuda, 8 on cpu)")
    p.add_argument("--max-batch-tokens", type=int, default=_int_env("MAX_BATCH_TOKENS"),
                   help="most padded tokens per forward pass (default: 32768 on cuda, 4096 on cpu)")
    p.add_argument("--max-padding", type=float, default=_float_env("MAX_PADDING"),
                   help="most padding per forward pass, as a fraction of real tokens (default: 0.5 on "
                        "cuda, 0.1 on cpu and mps)")
    p.add_argument("--max-inflight", type=int, default=int(_env("MAX_INFLIGHT", "1024")),
                   help="most requests admitted at once; more get RESOURCE_EXHAUSTED (default: %(default)s)")
    p.add_argument("--max-wait-ms", type=float, default=float(_env("MAX_WAIT_MS", "0")),
                   help="how long an idle server waits to fill a batch (default: %(default)s)")
    p.add_argument("--prep-threads", type=int, default=int(_env("PREP_THREADS", "4")),
                   help="threads for tokenizing and decoding (default: %(default)s)")
    p.add_argument("--torch-threads", type=int, default=int(_env("TORCH_THREADS", "0")),
                   help="CPU threads per forward pass; 0: performance cores on Apple Silicon, "
                        "else the PyTorch default")
    p.add_argument("--no-warm-up", action="store_true", default=_env("NO_WARM_UP", "") == "1",
                   help="skip the startup warm-up (the first requests will be slow)")
    p.add_argument("--grace", type=float, default=float(_env("GRACE", "10")),
                   help="seconds to finish in-flight requests on SIGTERM (default: %(default)s)")
    p.add_argument("--log-level", default=_env("LOG_LEVEL", "INFO"))
    return p.parse_args(argv)


def _int_env(name: str) -> Optional[int]:
    value = _env(name, "")
    return int(value) if value else None


def _float_env(name: str) -> Optional[float]:
    value = _env(name, "")
    return float(value) if value else None


def configure_model_cache() -> None:
    """Use ./.hf-cache when it exists and HF_HOME is not set, and skip the Hub update check once
    the checkpoints are there. Must run before laya / huggingface_hub are imported."""
    local = Path.cwd() / ".hf-cache"
    if "HF_HOME" not in os.environ and local.is_dir():
        os.environ["HF_HOME"] = str(local)
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    if (hub / "models--convaiinnovations--laya").is_dir():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _performance_cores() -> Optional[int]:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return None
    try:
        out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True, check=True)
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def resolve_models(spec: str) -> List[str]:
    names = list(MODELS) if spec.strip() == "all" else [m.strip() for m in spec.split(",") if m.strip()]
    if not names:
        raise SystemExit("--models: name at least one checkpoint")
    return names


def resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


async def serve(engine, host: str, port: int, grace: float, on_ready: Optional[Callable[[int], None]] = None,
                stop: Optional[asyncio.Event] = None) -> None:
    """Run the gRPC server until SIGINT/SIGTERM (or `stop` is set), then drain and exit.

    `on_ready` is called with the bound port once the server is listening (`port` 0 picks a free one).
    """
    import grpc
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc
    from grpc_reflection.v1alpha import reflection

    from fast_typed_classifier.service import ClassifierService
    from fast_typed_classifier.v1 import classifier_pb2, classifier_pb2_grpc

    server = grpc.aio.server(options=GRPC_OPTIONS)
    classifier_pb2_grpc.add_ClassifierServicer_to_server(ClassifierService(engine), server)
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    service_name = classifier_pb2.DESCRIPTOR.services_by_name["Classifier"].full_name
    reflection.enable_server_reflection((service_name, health.SERVICE_NAME, reflection.SERVICE_NAME), server)
    bound = server.add_insecure_port("%s:%d" % (host, port))

    await engine.start()
    await server.start()
    for name in ("", service_name):
        await health_servicer.set(name, health_pb2.HealthCheckResponse.SERVING)
    log.info("serving %s on %s:%d (device=%s, models=%s)", service_name, host, bound, engine.device,
             ",".join(engine.agents))
    if on_ready is not None:
        on_ready(bound)

    stop = stop or asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):  # not the main thread, or Windows
            pass
    await stop.wait()

    log.info("shutting down: finishing in-flight requests (up to %.0fs)", grace)
    # Tell load balancers first, then refuse new work, then let in-flight requests finish.
    await health_servicer.enter_graceful_shutdown()
    engine.begin_shutdown()
    await server.stop(grace)
    await engine.stop()
    log.info("stopped")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    configure_model_cache()

    import torch

    from fast_typed_classifier.engine import Engine

    threads = args.torch_threads or _performance_cores()
    if threads:
        torch.set_num_threads(threads)
    device = resolve_device(args.device)
    log.info("loading %s on %s (torch threads: %d)", args.models, device, torch.get_num_threads())
    engine = Engine.load(resolve_models(args.models), device=device, default_route=args.default_route,
                         prep_threads=args.prep_threads, max_batch_rows=args.max_batch_rows,
                         max_batch_tokens=args.max_batch_tokens, max_padding=args.max_padding,
                         max_inflight=args.max_inflight, max_wait_ms=args.max_wait_ms)
    log.info("limits: %s", engine.limits)
    if not args.no_warm_up:
        engine.warm_up()
    asyncio.run(serve(engine, args.host, args.port, args.grace))


if __name__ == "__main__":
    main()
