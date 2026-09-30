import itertools
import json
import os
import threading
import time
from contextlib import contextmanager

from ..device import get_torch_device

_RUN_DIR = os.environ.get("RUN_DIR") or None
_lock = threading.Lock()
_files = {}
_sids = itertools.count(1)
_step = None


def enabled():
    return _RUN_DIR is not None


def set_step(step):
    """
    이벤트에 붙일 trainer step 설정
    """
    global _step
    _step = int(step) if step is not None else None


def _cmdline():
    """
    파일을 쓴 프로세스 (ex. trainer, actor, vLLM) 를 확인
    """
    try:
        with open("/proc/self/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()[:200]

    except OSError:
        return ""


def _file(name: str):
    """
    다른 프로세서와 겹치지 않는 파일 반환
    """
    key = (name, os.getpid())
    f = _files.get(key)
    if f is None:
        os.makedirs(_RUN_DIR, exist_ok=True)
        f = open(os.path.join(_RUN_DIR, f"{name}.{os.getpid()}.jsonl"), "a", buffering=1)

        f.write(
            json.dumps(
                {
                    "meta": {
                        "pid": os.getpid(),
                        "cmdline": _cmdline(),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "ts_ns": time.time_ns(),
                    }
                }
            )
            + "\n"
        )

        _files[key] = f

    return f


def _write(name: str, record: dict):
    """
    name 문서에 record 를 json str 형태로 기록
    """
    line = json.dumps(record, separators=(",", ":"), default=str)

    # 동시 쓰기 방지
    with _lock:
        _file(name).write(line + "\n")


def event(name: str, phase: str, sid: int = None, **extra):
    """
    구간에 대해 phase 와 시간을 기록
    """
    if _RUN_DIR is None:
        return

    # record
    record = {"ts_ns": time.time_ns(), "name": name, "phase": phase, "step": _step}
    if sid is not None:
        record["id"] = sid
    if extra:
        record["extra"] = extra

    _write("events", record)


@contextmanager
def span(name: str, sync: bool = False, **extra):
    """
    phase B(시작) 와 E(끝) 에 대한 event 기록
    """
    if _RUN_DIR is None:
        yield
        return

    # gpu 의 경우 동기화 필요
    device = get_torch_device() if sync else None
    if device is not None and not (device.is_available() and device.is_initialized()):
        device = None
    if device is not None:
        device.synchronize()

    # span
    sid = next(_sids)
    event(name, "B", sid, **extra)
    try:
        yield
    finally:
        if device is not None:
            device.synchronize()
        event(name, "E", sid)


def trace(record: dict):
    """
    rollout 요청 하나를 trace 파일에 기록
    """
    if _RUN_DIR is None:
        return
    _write("trace", record)
