from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlparse

from data.audio_catalog import AudioCatalog
from providers.annotation_store import AnnotationStore
from providers.audio_visualization import (
    compute_spectrum_analysis,
    inspect_audio,
    read_audio_segment,
    render_frequency_profile_svg,
    render_spectrogram_png,
    render_wav_bytes,
    render_waveform_svg,
)


class AudioEdaHttpServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        catalog: AudioCatalog,
        annotation_store: AnnotationStore,
        page_html: bytes,
    ) -> None:
        super().__init__(server_address, AudioEdaRequestHandler)
        self.catalog = catalog
        self.annotation_store = annotation_store
        self.page_html = page_html


class AudioEdaRequestHandler(BaseHTTPRequestHandler):
    server: AudioEdaHttpServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/":
                self._send_bytes(HTTPStatus.OK, self.server.page_html, "text/html; charset=utf-8")
                return
            if path == "/api/summary":
                self._send_json(self.server.catalog.summary)
                return
            if path == "/api/annotations":
                recording_id = _query_optional_text(query, "recording_id")
                annotation_items = self.server.annotation_store.list_annotations(
                    recording_id=recording_id
                )
                payload = {"items": annotation_items}
                self._send_json(payload)
                return
            if path == "/api/recordings":
                dataset = _query_value(query, "dataset", "all")
                limit = _query_int(query, "limit", 50, minimum=1, maximum=200)
                offset = _query_int(query, "offset", 0, minimum=0, maximum=1_000_000)
                search_payload = self.server.catalog.search(
                    query=_query_value(query, "query", ""),
                    dataset=dataset,
                    limit=limit,
                    offset=offset,
                )
                self._send_json(search_payload)
                return
            if path.startswith("/api/recordings/"):
                recording_id = path.removeprefix("/api/recordings/")
                entry = self.server.catalog.get_entry(recording_id)
                detail = self.server.catalog.serialize_detail(entry)
                audio_info = inspect_audio(self.server.catalog.resolve_path(recording_id))
                detail["audio_info"] = {
                    "sample_rate": audio_info.sample_rate,
                    "channels": audio_info.channels,
                    "frames": audio_info.frames,
                    "duration_sec": audio_info.duration_sec,
                }
                self._send_json(detail)
                return
            if path == "/api/species":
                labels = _query_label_list(query)
                species_payload = {"items": self.server.catalog.serialize_species_batch(labels)}
                self._send_json(species_payload)
                return
            if path.startswith("/audio/") and path.endswith(".wav"):
                recording_id = path.removeprefix("/audio/").removesuffix(".wav")
                audio, sample_rate = read_audio_segment(
                    self.server.catalog.resolve_path(recording_id),
                    start_sec=_query_float(query, "start_sec", 0.0),
                    duration_sec=_query_optional_float(query, "duration_sec"),
                )
                self._send_bytes(HTTPStatus.OK, render_wav_bytes(audio, sample_rate), "audio/wav")
                return
            if path.startswith("/api/waveform/") and path.endswith(".svg"):
                recording_id = path.removeprefix("/api/waveform/").removesuffix(".svg")
                audio, _ = read_audio_segment(
                    self.server.catalog.resolve_path(recording_id),
                    start_sec=_query_float(query, "start_sec", 0.0),
                    duration_sec=_query_optional_float(query, "duration_sec"),
                )
                image_payload = render_waveform_svg(audio)
                self._send_bytes(HTTPStatus.OK, image_payload, "image/svg+xml")
                return
            if path.startswith("/api/frequency-profile/") and path.endswith(".svg"):
                recording_id = path.removeprefix("/api/frequency-profile/").removesuffix(".svg")
                audio, sample_rate = read_audio_segment(
                    self.server.catalog.resolve_path(recording_id),
                    start_sec=_query_float(query, "start_sec", 0.0),
                    duration_sec=_query_optional_float(query, "duration_sec"),
                )
                image_payload = render_frequency_profile_svg(audio, sample_rate)
                self._send_bytes(HTTPStatus.OK, image_payload, "image/svg+xml")
                return
            if path.startswith("/api/spectrogram/") and path.endswith(".png"):
                recording_id = path.removeprefix("/api/spectrogram/").removesuffix(".png")
                audio, sample_rate = read_audio_segment(
                    self.server.catalog.resolve_path(recording_id),
                    start_sec=_query_float(query, "start_sec", 0.0),
                    duration_sec=_query_optional_float(query, "duration_sec"),
                )
                image_payload = render_spectrogram_png(audio, sample_rate)
                self._send_bytes(HTTPStatus.OK, image_payload, "image/png")
                return
            if path.startswith("/api/spectrum-analysis/"):
                recording_id = path.removeprefix("/api/spectrum-analysis/")
                audio, sample_rate = read_audio_segment(
                    self.server.catalog.resolve_path(recording_id),
                    start_sec=_query_float(query, "start_sec", 0.0),
                    duration_sec=_query_optional_float(query, "duration_sec"),
                )
                analysis_payload = compute_spectrum_analysis(audio, sample_rate)
                self._send_json(analysis_payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
        except KeyError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path != "/api/annotations":
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            payload = self._read_json_body()
            recording_id = str(payload["recording_id"])
            entry = self.server.catalog.get_entry(recording_id)
            annotation = self.server.annotation_store.add_annotation(
                recording_id=recording_id,
                dataset=entry.dataset,
                filename=entry.filename,
                relative_path=entry.relative_path,
                start_sec=_payload_float(payload, "start_sec", 0.0),
                duration_sec=_payload_float(payload, "duration_sec", 0.0),
                category=str(payload.get("category", "other")),
                labels=_payload_labels(payload),
                note=str(payload.get("note", "")),
            )
            self._send_json({"annotation": annotation})
        except KeyError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        message = format % args
        sys.stderr.write(f"[audio-eda] {message}\n")

    def _send_json(self, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(HTTPStatus.OK, body, "application/json; charset=utf-8")

    def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return cast(dict[str, object], payload)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local BirdCLEF audio EDA server")
    parser.add_argument(
        "--competition-root",
        default="data/input/BirdCLEF+ 2026",
        help="directory that contains train.csv and audio folders",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--annotation-path",
        default="data/eda_annotations/audio_eda_annotations.jsonl",
        help="path to JSONL file that stores EDA annotations",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    competition_root = Path(args.competition_root).resolve()
    catalog = AudioCatalog.from_competition_root(competition_root)
    annotation_store = AnnotationStore(Path(args.annotation_path).resolve())
    page_html = _page_path().read_bytes()
    server = AudioEdaHttpServer(
        (args.host, args.port),
        catalog=catalog,
        annotation_store=annotation_store,
        page_html=page_html,
    )
    dataset_counts = cast(dict[str, int], catalog.summary["datasets"])
    print(f"Audio EDA server: http://{args.host}:{args.port}")
    print(f"Competition root: {competition_root}")
    print(
        "Indexed recordings: "
        f"{catalog.summary['recordings']} "
        f"(train_audio={dataset_counts.get('train_audio', 0)}, "
        f"train_soundscapes={dataset_counts.get('train_soundscapes', 0)}, "
        f"test_soundscapes={dataset_counts.get('test_soundscapes', 0)})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAudio EDA server stopped.")
    finally:
        server.server_close()
    return 0


def _page_path() -> Path:
    return Path(__file__).with_name("static").joinpath("audio_eda.html")


def _query_value(params: dict[str, list[str]], name: str, default: str) -> str:
    values = params.get(name)
    if not values:
        return default
    return values[0]


def _query_int(
    params: dict[str, list[str]],
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = int(_query_value(params, name, str(default)))
    return max(minimum, min(value, maximum))


def _query_float(params: dict[str, list[str]], name: str, default: float) -> float:
    return float(_query_value(params, name, str(default)))


def _query_optional_float(params: dict[str, list[str]], name: str) -> float | None:
    value = _query_value(params, name, "")
    if value == "":
        return None
    return float(value)


def _query_optional_text(params: dict[str, list[str]], name: str) -> str | None:
    value = _query_value(params, name, "")
    if value == "":
        return None
    return value


def _query_label_list(params: dict[str, list[str]]) -> list[str]:
    raw = _query_value(params, "labels", "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _payload_float(payload: dict[str, object], key: str, default: float) -> float:
    raw_value = payload.get(key, default)
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    if isinstance(raw_value, str):
        return float(raw_value)
    raise ValueError(f"{key} must be numeric")


def _payload_labels(payload: dict[str, object]) -> list[str]:
    raw_value = payload.get("labels", [])
    if not isinstance(raw_value, list):
        raise ValueError("labels must be a list")
    return [str(label) for label in raw_value if str(label).strip()]


if __name__ == "__main__":
    raise SystemExit(main())
