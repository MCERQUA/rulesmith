"""Clef, Cloudflare's decision model, served from its own release: `POST /v1/systemone`
answered as Jev answers it, on an NVIDIA GPU, Apple's, or a CPU, for `--backend clef`."""

import importlib
import sys
import threading
import traceback
from collections.abc import Callable
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path

from rulesmith.serve import JSONHandler

PATH = "/v1/systemone"

Answer = Callable[[dict], dict]


def load(weights: str, device: str | None) -> tuple[Answer, str]:
    """The release's own `systemone` bound to its model, and the device it runs on: `device`,
    or else the best this machine has. `weights` is a Hugging Face repo or a downloaded copy."""
    # The clef extra's packages, imported only here so the rest of rulesmith installs without.
    import torch
    from huggingface_hub import snapshot_download

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            print("warning: no GPU found; Clef will run on the CPU, slowly", file=sys.stderr)
    path = Path(weights) if Path(weights).is_dir() else Path(snapshot_download(weights))
    # The model's code ships beside its weights, as the release's model card loads it.
    sys.path.insert(0, str(path))
    release = importlib.import_module("joint_schema_model")
    model, processor = release.load_release_model(path, device=device)
    return (lambda request: release.systemone(model, processor, request)), device


def make_server(answer: Answer, host: str, port: int) -> ThreadingHTTPServer:
    """A server that answers each systemone request with `answer`, one at a time."""

    class Handler(JSONHandler):
        def post(self, request):
            if self.path != PATH:
                return HTTPStatus.NOT_FOUND, {"error": f"nothing at {self.path}; ask {PATH}"}
            if not isinstance(request, dict):
                return HTTPStatus.BAD_REQUEST, {"error": "a systemone request is a JSON object"}
            try:
                with self.server.lock:
                    return HTTPStatus.OK, answer(request)
            # The release checks a request as it reads it, so a malformed one fails here.
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                return HTTPStatus.BAD_REQUEST, {"error": f"not a systemone request: {error}"}
            except RuntimeError as error:
                # Out of memory, most often: the caller hears why, the log keeps where.
                traceback.print_exc()
                return HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"the model failed: {error}"}

    server = ThreadingHTTPServer((host, port), Handler)
    # One model on one device, which the release does not make safe to run from two threads.
    server.lock = threading.Lock()
    return server
