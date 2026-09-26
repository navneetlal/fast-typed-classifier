#!/usr/bin/env bash
# Regenerate the Python gRPC stubs from proto/. Run after editing a .proto file.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
"$PY" -m grpc_tools.protoc \
  -I proto \
  --python_out=src --pyi_out=src --grpc_python_out=src \
  proto/fast_typed_classifier/v1/classifier.proto
touch src/fast_typed_classifier/v1/__init__.py
echo "generated src/fast_typed_classifier/v1/"
