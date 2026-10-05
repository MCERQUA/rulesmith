"""Decisions over HTTP: each served graph answers `POST /decisions/<name>` with what
`rulesmith run` prints, so a workflow in any language can ask one at its branch points."""

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pydantic import ValidationError

from rulesmith.runtime import DecisionProgram, GraphExecutionError

PREFIX = "/decisions"


def make_server(programs: dict[str, DecisionProgram], host: str, port: int) -> ThreadingHTTPServer:
    """A server for these decisions, by name. Each request is decided on its own thread; a
    program keeps no state between decisions, and its trace appends a whole line at a time."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == PREFIX:
                return self.answer(HTTPStatus.OK, {"decisions": sorted(programs)})
            return self.answer(HTTPStatus.NOT_FOUND, {"error": f"nothing at {self.path}"})

        def do_POST(self):
            name = self.path.removeprefix(f"{PREFIX}/")
            if not self.path.startswith(f"{PREFIX}/") or name not in programs:
                return self.answer(
                    HTTPStatus.NOT_FOUND,
                    {"error": f"no decision named {name!r}; serving {sorted(programs)}"},
                )
            length = int(self.headers.get("Content-Length") or 0)
            try:
                state = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as error:
                return self.answer(
                    HTTPStatus.BAD_REQUEST, {"error": f"the body is not JSON: {error}"}
                )
            try:
                decided = programs[name](state=state)
            except ValidationError as error:
                return self.answer(HTTPStatus.BAD_REQUEST, {"error": f"not an input: {error}"})
            except GraphExecutionError as failure:
                # The graph ran and could not answer: the caller decides what that means.
                return self.answer(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {"error": str(failure), "result": dict(failure.prediction)},
                )
            return self.answer(HTTPStatus.OK, dict(decided))

        def answer(self, status: HTTPStatus, body: dict) -> None:
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            return  # the trace is the record of what was decided

    return ThreadingHTTPServer((host, port), Handler)
