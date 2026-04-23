"""Small vLLM subprocess manager for the RunPod worker."""

from __future__ import annotations

import logging
import os
import select
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def _pause_group(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGSTOP)
    except ProcessLookupError:
        pass


def _resume_group(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGCONT)
    except ProcessLookupError:
        pass


class ManagedVllmServer:
    """Launch and supervise one local vLLM API server."""

    _GPU_PHASE_LOCKS: dict[str, threading.Lock] = {}
    _GPU_PHASE_LOCKS_MUTEX = threading.Lock()

    @classmethod
    def _gpu_lock(cls, cuda_devices: str) -> threading.Lock:
        with cls._GPU_PHASE_LOCKS_MUTEX:
            lock = cls._GPU_PHASE_LOCKS.get(cuda_devices)
            if lock is None:
                lock = threading.Lock()
                cls._GPU_PHASE_LOCKS[cuda_devices] = lock
            return lock

    def __init__(
        self,
        *,
        name: str,
        model: str,
        port: int,
        io_processor_plugin: str | None = None,
        runner: str = "pooling",
        gpu_memory_utilization: float = 0.30,
        max_model_len: int | None = None,
        max_num_seqs: int = 128,
        max_num_batched_tokens: int | None = None,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        enforce_eager: bool | None = None,
        enable_prefix_caching: bool = False,
        enable_chunked_prefill: bool = False,
        startup_timeout: int = 600,
        cuda_devices: str = "0",
        plugins: list[str] | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.port = int(port)
        self.io_processor_plugin = io_processor_plugin
        self.runner = runner
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.max_model_len = max_model_len
        self.max_num_seqs = int(max_num_seqs)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.dtype = dtype
        self.trust_remote_code = bool(trust_remote_code)
        self.enforce_eager = enforce_eager
        self.enable_prefix_caching = bool(enable_prefix_caching)
        self.enable_chunked_prefill = bool(enable_chunked_prefill)
        self.startup_timeout = int(startup_timeout)
        self.cuda_devices = cuda_devices
        self.plugins = list(plugins or [])
        self.extra_args = list(extra_args or [])

        self.process: subprocess.Popen[str] | None = None
        self._gpu_phase_acquired = False
        self._gpu_lock_instance = self._gpu_lock(cuda_devices)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _build_command(self) -> list[str]:
        cmd = [
            "vllm",
            "serve",
            self.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--runner",
            self.runner,
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--dtype",
            self.dtype,
            "--uvicorn-log-level",
            "warning",
        ]
        if self.io_processor_plugin:
            cmd.extend(["--io-processor-plugin", self.io_processor_plugin])
        if self.max_model_len:
            cmd.extend(["--max-model-len", str(self.max_model_len)])
        if self.max_num_batched_tokens:
            cmd.extend(["--max-num-batched-tokens", str(self.max_num_batched_tokens)])
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.enforce_eager is True:
            cmd.append("--enforce-eager")
        elif self.enforce_eager is False:
            cmd.append("--no-enforce-eager")
        if not self.enable_prefix_caching:
            cmd.append("--no-enable-prefix-caching")
        if not self.enable_chunked_prefill:
            cmd.append("--no-enable-chunked-prefill")
        cmd.extend(self.extra_args)
        return cmd

    def _assert_port_available(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            in_use = sock.connect_ex(("127.0.0.1", self.port)) == 0
        if in_use:
            raise RuntimeError(
                f"{self.name} cannot bind 127.0.0.1:{self.port}; the port is already in use"
            )

    def _release_gpu_lock_if_held(self) -> None:
        if self._gpu_phase_acquired:
            try:
                self._gpu_lock_instance.release()
            except RuntimeError:
                pass
            self._gpu_phase_acquired = False

    def _handle_gpu_phase_marker(self, line: str) -> None:
        if self._gpu_phase_acquired or "Starting to load model" not in line:
            return
        pid = self.process.pid if self.process else None
        if pid is None:
            return
        logger.info("[%s] GPU phase detected; acquiring lock", self.name)
        _pause_group(pid)
        self._gpu_lock_instance.acquire()
        self._gpu_phase_acquired = True
        _resume_group(pid)

    def _health_request_sync(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=3) as response:
                return response.status == 200
        except Exception:
            return False

    def _start_log_reader(self) -> None:
        def _reader() -> None:
            proc = self.process
            if proc is None or proc.stdout is None:
                return
            try:
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        logger.info("[%s] %s", self.name, line)
            except (ValueError, OSError):
                return

        threading.Thread(
            target=_reader,
            name=f"vllm-log-{self.name}",
            daemon=True,
        ).start()

    def _wait_for_ready(self) -> None:
        start = time.time()
        last_error: Exception | None = None
        while time.time() - start < self.startup_timeout:
            if self.process is not None and self.process.stdout is not None:
                while True:
                    try:
                        ready, _, _ = select.select([self.process.stdout], [], [], 0)
                    except Exception:
                        break
                    if not ready:
                        break
                    line = self.process.stdout.readline()
                    if not line:
                        break
                    line = line.rstrip()
                    if not line:
                        continue
                    logger.info("[%s] %s", self.name, line)
                    self._handle_gpu_phase_marker(line)

            if self.process is not None and self.process.poll() is not None:
                self._release_gpu_lock_if_held()
                raise RuntimeError(f"{self.name} exited with code {self.process.returncode}")

            try:
                if self._health_request_sync():
                    self._release_gpu_lock_if_held()
                    return
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc

            time.sleep(0.2)

        self._release_gpu_lock_if_held()
        raise TimeoutError(f"{self.name} failed to start within {self.startup_timeout}s: {last_error}")

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return

        self._assert_port_available()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.cuda_devices
        env.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
        if self.plugins:
            env["VLLM_PLUGINS"] = ",".join(self.plugins)

        cmd = self._build_command()
        logger.info("[%s] starting: %s", self.name, " ".join(cmd))
        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            self._wait_for_ready()
            self._start_log_reader()
            logger.info("[%s] ready on %s", self.name, self.base_url)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        proc = self.process
        if proc is None:
            return
        pid = proc.pid
        try:
            _resume_group(pid)
        except Exception:
            pass
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        finally:
            self._release_gpu_lock_if_held()
            self.process = None

    def health(self) -> dict[str, Any]:
        if self.process is None or self.process.poll() is not None:
            return {
                "status": "stopped",
                "name": self.name,
                "pid": None,
                "exit_code": self.process.returncode if self.process else None,
            }
        return {
            "status": "healthy" if self._health_request_sync() else "unhealthy",
            "name": self.name,
            "pid": self.process.pid,
            "url": self.base_url,
        }
