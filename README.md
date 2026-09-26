# fast-typed-classifier

An internal gRPC microservice for typed-decision classification, built on
[Laya](https://github.com/NandhaKishorM/laya). Laya is a non-autoregressive "decision engine": it answers
typed questions (`choice`, `score`, `noul`) about a piece of text or JSON in a single forward pass.
Nothing is generated, so there is nothing to parse and nothing to hallucinate.

Callers send an input and a list of questions; the service routes the input to the right
checkpoint by script and language (Laya's **Route Mode**), answers every question in one forward pass,
and returns typed answers with calibrated confidence.

| Checkpoint | Encoder | Used for |
|---|---|---|
| `english` (`convaiinnovations/laya`) | ModernBERT-large (421M) | English |
| `multilingual` (`laya-multilingual`) | mmBERT-base (322M) | 100+ languages, including English |
| `typed-decisions` (`laya-typed-decisions`) | ModernBERT-large (421M) | only on explicit request (`model: "typed-decisions"`) |

## Quick start

Everything stays inside this folder: Python packages in `.venv/`, model checkpoints in `.hf-cache/`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"

# Start the server (CPU here; on a machine with CUDA, the default --device auto picks the GPU)
.venv/bin/fast-typed-classifier --device cpu

# In another terminal: classify a text with the example support-triage questions
.venv/bin/python -m fast_typed_classifier.client "We were billed twice, please refund the duplicate."
```

The server downloads the checkpoints into `.hf-cache/` on first start (about 2.2 GB for all three),
then runs offline. Startup (load and warm-up) takes about 10 s on CPU. The service is ready when it
logs `serving fast_typed_classifier.v1.Classifier on 0.0.0.0:50051`, and its gRPC health check
reports `SERVING`.

Requires Python 3.10 or newer. Tested with Python 3.14, `laya` 0.3.20, `torch` 2.14 and `grpcio` 1.84.

## API

The contract is [`proto/fast_typed_classifier/v1/classifier.proto`](proto/fast_typed_classifier/v1/classifier.proto).
Service `fast_typed_classifier.v1.Classifier`:

| RPC | Use it for |
|---|---|
| `Classify(ClassifyRequest) → ClassifyResponse` | One input. Concurrent calls are batched together on the server. |
| `ClassifyBatch(ClassifyBatchRequest) → ClassifyBatchResponse` | Many inputs in one call. Responses keep request order; one bad item does not fail the call. Batch-level `questions` apply to items that have none. |
| `ClassifyStream(stream ClassifyRequest) → stream ClassifyResponse` | High-throughput pipelines. Responses arrive as each is ready, not in request order; match them by `request_id`. |
| `Route(ClassifyRequest) → Routing` | Which checkpoint would answer, and why. No forward pass. |
| `GetServerInfo` | Loaded checkpoints, device, batching limits, versions. |

The server also serves the standard gRPC health check (`grpc.health.v1.Health`) and server
reflection, so tools such as grpcurl and Postman can discover the API.

### Request

```text
ClassifyRequest
  request_id   echoed back; needed to match stream responses
  text | json  the input: plain text, or a JSON object (e.g. an email) / array (a conversation)
  questions    repeated Question, answered in one forward pass
  model        optional: force "english", "multilingual" or "typed-decisions"
  lang         optional language hint, e.g. "de"
  max_len      optional token budget (0: checkpoint default; up to 8192 for multilingual)

Question { id, instructions, one of:
  choice { options: [{label, description}] }   pick one label
  score  { levels: [description, ...] }        expected level, 0 to n-1
  noul   { true_description, false_description, true_label, false_label }   P(true), all optional
}
```

### Response

`answers` maps each question id to an `Answer`: its `type`, the value (`choice` label, `score` or
`noul` probability), `probabilities` (per option or level, in request order; `[P(false), P(true)]` for
noul), `confidence`, `answer_confidence` (calibrated, comparable across types) and `act_probability`.
It also carries `routing` (checkpoint, reason, detected script and language), `input_tokens`, and
`timing` (`queue_ms`, `inference_ms`, `batch_size`) for observability.

### Errors

| Status | When |
|---|---|
| `INVALID_ARGUMENT` | Missing or invalid state or questions, duplicate ids, unknown `model`, `max_len` above 8192. |
| `FAILED_PRECONDITION` | The request routed to a checkpoint this server did not load (see `--models`). |
| `RESOURCE_EXHAUSTED` | More than `--max-inflight` requests in flight. Retry with backoff. |
| `UNAVAILABLE` | The server is shutting down. Retry on another replica. |
| `DEADLINE_EXCEEDED` | The client's deadline passed. A request that times out while queued is dropped before it reaches the model. |

In `ClassifyBatch` and `ClassifyStream`, a failed item gets these codes in its response's `error`
field instead, and the other items are unaffected.

### Calling from Python

```python
import grpc
from fast_typed_classifier import client
from fast_typed_classifier.v1 import classifier_pb2_grpc

stub = classifier_pb2_grpc.ClassifierStub(grpc.insecure_channel("localhost:50051"))
response = stub.Classify(client.request(
    {"subject": "Duplicate charge", "body": "We were billed twice, please refund."},
    [client.choice("department", "Which department should handle this?",
                   {"billing": "invoices, payments, refunds", "other": "everything else"}),
     client.score("urgency", "How urgent is this?", ["not urgent", "soon", "blocking"]),
     client.noul("refund_requested", "Does the user ask for a refund?")]),
    timeout=5)

response.answers["department"].choice         # "billing"
response.answers["refund_requested"].noul     # P(yes)
response.routing.model                        # "english"
```

Other languages: generate stubs from the `.proto` with your language's gRPC tooling. After editing
the `.proto`, regenerate the Python stubs with `./scripts/gen_proto.sh`.

**Always set a deadline** (`timeout=`) on calls, and use a gRPC retry policy with backoff for
`RESOURCE_EXHAUSTED` and `UNAVAILABLE`. Under overload the server rejects requests immediately rather than
queueing them without bound. A client that retries without backoff still costs the server CPU
(measured: throughput fell from 5.1 to 3.6 req/s under about 11,000 rejections per second).

## Configuration

Every flag can also be set with an environment variable: `FTC_` followed by the flag name in upper case,
e.g. `FTC_PORT=50051 FTC_MODELS=multilingual`.

| Flag | Default | Meaning |
|---|---|---|
| `--host`, `--port` | `0.0.0.0`, `50051` | Listen address. |
| `--device` | `auto` | `auto` picks `cuda` if available, else `cpu`. `mps` (Apple GPU) only when named. |
| `--models` | `english,multilingual` | Checkpoints to load, or `all`. Everything auto-routed is covered by these two. |
| `--default-route` | `english` | Checkpoint for text whose language cannot be determined. |
| `--max-batch-rows` | 512 on CUDA, 8 on CPU and MPS | Most (input, question) rows per forward pass. |
| `--max-batch-tokens` | 32768 on CUDA, 4096 on CPU and MPS | Most padded tokens per forward pass (bounds memory). |
| `--max-padding` | 0.5 on CUDA, 0.1 on CPU and MPS | Requests share a pass only if the padding that adds costs at most this fraction more than separate passes. |
| `--max-inflight` | `1024` | Most requests admitted at once; more get `RESOURCE_EXHAUSTED`. |
| `--max-wait-ms` | `0` | How long an idle server waits to fill a batch. 0 adds no latency. |
| `--prep-threads` | `4` | Threads for routing and tokenization. |
| `--torch-threads` | `0` | CPU threads per forward pass. 0: the performance cores on Apple Silicon, else the PyTorch default. |
| `--no-warm-up` | off | Skip the startup warm-up (the first requests will then be slow). |
| `--grace` | `10` | Seconds to finish in-flight requests on SIGTERM. |

**Multilingual-only deployment.** `--models multilingual` needs about half the memory, and on CPU it
answers about 3x faster than `english` (97 ms vs 285 ms per 4-question request). English text is
then answered by `multilingual` automatically (the routing reason says so). An explicit
`model: "english"` still fails with `FAILED_PRECONDITION`. Check accuracy on your own English
data before choosing this: the English checkpoint is the larger model.

## How it gets high throughput and low latency

Each request goes through a three-stage pipeline, so the device never waits on Python:

```text
 gRPC handlers (asyncio)
   │  validate (question sets are cached), admission check
   ▼
 prep threads ── route (script/language detection) + tokenize
   ▼
 queue ── continuous batching
   ▼
 inference thread ── collate + forward pass on the device
   ▼
 decode thread ── logits → typed answers (while the next forward pass runs)
```

- **Continuous batching, no added wait.** When the device is free, the inference thread takes
  everything queued for the oldest request's checkpoint (up to the batch limits) and runs it at once.
  An idle server serves a request immediately. Under load, requests that arrive during a forward pass
  are batched into the next one.
- **Batches across question sets and languages.** Laya turns each (input, question) pair into an
  independent row, so requests with different questions or languages share one forward pass. Each
  row is decoded with its own language's calibration, exactly as `Router.predict` does. (Laya's own
  `Router.predict_batch` batches only identical question sets and drops the detected language, which
  changes confidence values for non-English text.)
- **Length-matched batches.** Requests are sorted by length, and a short request shares a pass with a
  longer one only if the padding costs at most `--max-padding` more than running them separately.
  Without this, a 30-word request batched with a 250-word one is padded to about 8x its length, and
  on CPU and MPS every padding token costs as much as a real one. A request's own rows always stay in
  one pass (as in `Router.predict`), because splitting them costs a whole extra pass: each pass
  streams all the model weights, about 1.7 GB for `english`. Only a request too big for
  `--max-batch-rows` / `--max-batch-tokens` on its own is split.
- **Load shedding.** Admission is checked before any parsing. Requests cancelled or timed out
  while queued are skipped before they reach the model, and one failing request is retried alone so
  it cannot fail its batch-mates.
- **Warm-up at startup.** Each checkpoint runs a forward pass at a small and a full batch shape
  before the server reports `SERVING`, so no user request pays the multi-second first-call cost.
- **Graceful shutdown.** On SIGTERM the health check switches to `NOT_SERVING`, new requests get
  `UNAVAILABLE`, and in-flight requests finish (up to `--grace` seconds).

The batching code calls Laya's `Agent` internals (`_encode_state`, `_forward`, `_decode_answers`), so
`laya` is pinned to `0.3.20` exactly. `tests/test_real_models.py` checks that answers match
`Router.predict`; run it before upgrading laya.

### Measured on CPU (Apple M1, 8 GB, `--device cpu`)

4 questions per request (the support-triage set), via `bench/load_test.py`:

| Load | Throughput | Latency p50 / p99 | Notes |
|---|---|---|---|
| 1 client, English | 3.0 req/s | 285 / 579 ms | the forward pass itself; about 1 ms queueing |
| 1 client, multilingual | 10.0 req/s | 97 / 136 ms | |
| 8 clients, mixed | 4.9 req/s | 1633 / 2454 ms | CPU saturated; latency is queueing |
| 32 clients, mixed | 4.6 req/s | 6274 / 9032 ms | same throughput, longer queue |

The service adds about 1.6 ms per request on top of the forward pass (routing, tokenization and
gRPC; see `test_single_request_latency_is_model_bound`).

On CPU (and Apple GPUs) the model is compute-bound even for one request: time per request is flat from
batch 1 to 32. Batching therefore adds almost no throughput there. The small CPU batch default
(8 rows) was the best measured: 4.9 req/s against 4.1 for 4 rows and 4.2 for 32 rows. CPU throughput scales
by adding replicas, not bigger batches.

### Measured on the Apple GPU (M1, `--device mps`)

Same load test and defaults (8 rows, 4096 tokens, 10% padding):

| Load | MPS | CPU (from above) |
|---|---|---|
| 1 client, English, p50 | 188 ms | 285 ms |
| 1 client, multilingual, p50 | 85 ms | 97 ms |
| 8 clients mixed, throughput | 6.9 to 7.1 req/s (p50 about 1.1 s) | 4.9 req/s |

- **fp16 only when batched.** Laya runs a forward pass in fp16 on MPS only when it has 5 or more rows. So a lone
  4-question request runs in fp32, and two batched requests (8 rows) run in fp16. In back-to-back
  runs this was faster than fp32 without batching (`--max-batch-rows 4`): 6.9 vs 5.9 req/s at 8
  clients. An alternating re-test to rule out heat was cut short when the Mac went to sleep.
  Batched answers differ from a lone request's by at most 0.006, with the same labels.
- **No first-time cost for new input lengths.** 40 requests with lengths the server had never seen
  ran as fast as the same requests repeated, on MPS and on CPU.
- **Length-matched batching matters here.** With 40 English requests of 3 to 250 words at 8 clients, matching
  requests by length raised MPS throughput from 1.53 to 1.65 req/s. Splitting a single request's
  rows by length instead made one client 20% slower, which is why a request's rows stay together.

These numbers come from a fanless MacBook Air: sustained runs slow down as it heats up, and it
sleeps when idle on battery, so repeated measurements vary by about 15%. Compare settings in
alternating runs with cool-downs between them, and keep the machine awake (`caffeinate -i`).

### On CUDA

Production is meant to run on CUDA, where batching pays off: Laya's own numbers on a T4 are 33 ms for one
question versus 7.2 ms per question batched. Nothing in the code is CPU-specific. `--device auto`
selects the GPU, Laya enables fp16/bf16 autocast, and the batch limits switch to 512 rows / 32768
tokens. **This has not been run on a CUDA GPU yet.** Before production:

- Run the test suite on the GPU host: `FTC_TEST_DEVICE=cuda pytest -m models` compares answers with
  `Router.predict`. On GPUs, batched requests may run in fp16 where a lone request runs in fp32, so
  that test allows differences up to 0.02 for batched requests (measured on MPS: up to 0.006), but
  chosen labels must match exactly.
- Run `bench/load_test.py` at rising `--concurrency` and tune `--max-batch-rows` / `--max-batch-tokens`
  for the GPU's memory and your latency target.
- At high request rates, Python work per request (about 1.5 ms of tokenization) can become the limit
  before the GPU does. Run two server processes per GPU, or more replicas behind the load balancer.
- Laya's optional TileLang fast path (`laya[fast]`, `Agent.accelerate()`) is not wired in yet.

## Testing

```bash
.venv/bin/python -m pytest -m "not models"   # 57 tests, about 5 s: conversion, batching engine, gRPC surface
.venv/bin/python -m pytest                   # all 61, adds the real checkpoints on CPU (about 30 s)
FTC_TEST_DEVICE=mps .venv/bin/python -m pytest -m models   # the real checkpoints on the Apple GPU
```

- `tests/test_convert.py`: proto ↔ Laya conversion and every validation rule.
- `tests/test_engine.py`: the batcher, with a fake model that tags every row. It checks that no request ever
  gets another's answers (120 concurrent mixed requests), and it covers the row and token budgets,
  length sorting, cancellation, error isolation, overload, shutdown, language calibration and the
  multilingual fallback.
- `tests/test_service.py`: every RPC over a real socket, error codes, batch and stream per-item errors,
  deadlines (a timed-out request never reaches the model), health, reflection, and graceful shutdown.
- `tests/test_real_models.py`: the real checkpoints (CPU by default, or `FTC_TEST_DEVICE`), compared with
  `Router.predict` across English, Hindi, German, Spanish, a conversation, `lang` and `model` overrides,
  truncation and every question type. On CPU the answers match exactly, including when 30 requests
  run concurrently in shared batches. On MPS single requests match exactly; batched ones are within
  0.006, with the same labels.

Load test against a running server:

```bash
.venv/bin/python bench/load_test.py --concurrency 8 --duration 30 --mix mixed
.venv/bin/python bench/load_test.py --mode stream --concurrency 32 --mix multilingual
```

## Project layout

```text
proto/fast_typed_classifier/v1/classifier.proto   the API contract
src/fast_typed_classifier/
  server.py      CLI, gRPC server, health, reflection, graceful shutdown
  service.py     the Classifier RPCs and error mapping
  engine.py      routing, tokenization, continuous batching, inference
  convert.py     proto <-> Laya formats, validation, question-set cache
  client.py      request and question builders, example CLI
  v1/            generated stubs (scripts/gen_proto.sh)
tests/           see Testing
bench/           load_test.py
```

## Notes

- Laya logs a `RuntimeWarning` when loading one checkpoint: a shipped calibration temperature is out of
  range and gets clamped. Confidence from the affected entries is uncalibrated; answers are not affected.
- References: [Laya repository](https://github.com/NandhaKishorM/laya),
  [Laya documentation](https://nandhakishorm.github.io/laya/).
