"""Main method runner with resource monitoring."""

import fnmatch
import os
import sys
import json
import time
import shlex
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from .method import injected_params
from .monitor import ResourceMonitor
from .results import ResultWriter
from .serialization import json_default
from .subprocess_utils import run_with_realtime_output, save_output_to_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



def _parse_monitor_interval(raw, default: float = 1.0) -> float:
    """Validate MONITOR_INTERVAL.

    It was parsed with a bare float(): a non-numeric value raised before
    run() had written anything, 0 produced a busy loop forking nvidia-smi as
    fast as it could, a negative value raised inside the daemon monitor thread
    (which then died silently), and anything above 2 outlived the 2-second
    join and raced the final sample.
    """
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            f"[WARN] MONITOR_INTERVAL={raw!r} is not a number; using {default}s"
        )
        return default
    if not (value == value) or value in (float("inf"), float("-inf")):
        logger.warning(f"[WARN] MONITOR_INTERVAL={raw!r} is not finite; using {default}s")
        return default
    if value < 0.05:
        logger.warning(f"[WARN] MONITOR_INTERVAL={value} is too small; using 0.05s")
        return 0.05
    if value > 2.0:
        logger.warning(
            f"[WARN] MONITOR_INTERVAL={value} exceeds the 2s join timeout; using 2.0s"
        )
        return 2.0
    return value


class MethodRunner:
    """
    Wrapper for executing graph anomaly detection methods.
    
    Features:
    - Automatic resource monitoring (CPU, memory, GPU)
    - Execution timing
    - Result standardization
    - Error handling
    
    Usage in Dockerfile:
        CMD ["python", "-m", "graflag_runner.runner"]
        
    Environment variables required:
        - DATA: Input dataset path
        - EXP: Experiment output path
        - METHOD_NAME: Method name
        - COMMAND: Command to execute (e.g., "python main.py --dataset uci")
    """
    
    def __init__(
        self,
        data_dir: str,
        exp_dir: str,
        method_name: str,
        command: str,
        monitor_interval: float = 1.0,
        pass_env_args: bool = False,
        **kwargs
    ):
        """
        Initialize method runner.
        
        Args:
            data_dir: Input dataset directory
            exp_dir: Experiment output directory
            method_name: Name of the method
            command: Command to execute
            monitor_interval: Resource monitoring interval in seconds (default: 1.0)
            pass_env_args: Whether to extract env vars starting with _ and pass as CLI args (default: False)
            **kwargs: Additional configuration
        """
        self.data_dir = Path(data_dir)
        self.exp_dir = Path(exp_dir)
        self.method_name = method_name
        self.command = command
        self.monitor_interval = monitor_interval
        self.pass_env_args = pass_env_args
        self.config = kwargs
        
        # Extract environment variables starting with _ if requested
        if self.pass_env_args:
            self.command = self._build_command_with_env_args()
        
        # Create experiment directory
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize resource monitor (uses spot() method now)
        self.monitor = ResourceMonitor()
        
        logger.info("=" * 60)
        logger.info(f"GraFlag Runner - {self.method_name}")
        logger.info("=" * 60)
        logger.info(f"[INFO] Data: {self.data_dir}")
        logger.info(f"[INFO] Output: {self.exp_dir}")
        logger.info(f"[INFO] Command: {self.command}")
        logger.info(f"[INFO] Monitor interval: {self.monitor_interval}s")
        logger.info(f"[INFO] Pass env args: {self.pass_env_args}")
        logger.info("")
    
    def _build_command_with_env_args(self) -> str:
        """
        Extract environment variables starting with _ and append them as CLI arguments.

        Example:
            _BATCH_SIZE=128 -> --batch_size 128
            _LEARNING_RATE=0.001 -> --learning_rate 0.001

        Returns:
            Command string with appended arguments
        """
        env_args = []

        # injected_params() is the filter, not a bare scan of os.environ:
        # GRAFLAG_PARAMS names what GraFlag actually set, so a shell's own
        # `_P9K_TTY` or conda's empty `_CE_CONDA` no longer becomes a
        # `--ce_conda` the method's argparse rejects.
        for key, value in injected_params().items():
            # Remove leading underscore and convert to lowercase
            arg_name = key[1:].lower()
            if value == "":
                # Documented convention: an empty value means a bare boolean
                # flag (`_USE_MEMORY=` -> `--use_memory`). Quoting it would
                # emit `--use_memory ''` and make a store_true argument fail.
                # No in-tree method spells a flag this way any more -- gady was
                # the last, and its .env now says `true` -- but the convention
                # stays for anything still passing --pass-env-args.
                env_args.append(f"--{arg_name}")
                continue
            # The command is run through a shell, so values with spaces or
            # metacharacters must be quoted (e.g. _HIDDEN_DIMS="64 128").
            env_args.append(f"--{arg_name} {shlex.quote(value)}")

        if env_args:
            args_str = " ".join(env_args)
            logger.info(f"[INFO] Extracted env args: {args_str}")
            return f"{self.command} {args_str}"

        return self.command
    
    def _merge_runtime_metadata(self, exec_time_ms: float, resources: dict):
        """Record the runner's own measurements in results.json metadata.

        RESULTS_STANDARD.md says resource metrics are "also set automatically
        by graflag_runner", but nothing called add_resource_metrics -- the
        runner measured execution time and peak memory, wrote them only to
        status.json, and results.json carried whatever the method happened to
        record for itself (often nothing).

        The runner's numbers win. Methods that track their own do it with a
        handful of point samples of their own process, while the monitor
        samples the whole process tree continuously -- on a real dynwalk run
        the method reported 409 MB against the monitor's 907 MB, a 2.2x
        understatement in a field people compare across methods. Anything the
        method reported is preserved under method_reported_* so nothing is lost.

        Failures here are logged and ignored -- the run already succeeded.
        """
        results_file = self.exp_dir / "results.json"
        try:
            with open(results_file) as fh:
                data = json.load(fh)

            metadata = data.setdefault("metadata", {})
            measured = {"exec_time_ms": round(exec_time_ms, 2)}
            for key in ("peak_memory_mb", "peak_gpu_mb"):
                if resources and resources.get(key) is not None:
                    measured[key] = resources[key]

            for key, value in measured.items():
                reported = metadata.get(key)
                if reported is not None and reported != value:
                    metadata[f"method_reported_{key}"] = reported
                metadata[key] = value

            tmp = results_file.with_name(results_file.name + ".meta.tmp")
            with open(tmp, "w") as fh:
                json.dump(data, fh, indent=2, default=json_default)
            os.replace(tmp, results_file)
        except Exception as exc:
            logger.warning(f"[WARN] Could not record resource metrics: {exc}")

    def _results_problem(self) -> Optional[str]:
        """Describe what is wrong with results.json, or None if it is fine."""
        results_file = self.exp_dir / "results.json"
        if not results_file.is_file():
            return f"wrote no results.json in {self.exp_dir}"
        try:
            with open(results_file) as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            return f"results.json is not valid JSON ({exc})"
        except OSError as exc:
            return f"results.json could not be read ({exc})"
        if not isinstance(data, dict) or data.get("result_type") is None:
            return "results.json has no result_type"
        if "scores" not in data and "scores_file" not in data:
            return "results.json has no scores"
        return None

    def _save_status(self, status: str, exec_time_ms: float = None,
                     resources: dict = None, exit_code: int = None,
                     error: str = None):
        """Save execution status to status.json in the experiment directory."""
        status_data = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method_name": self.method_name,
        }
        if exec_time_ms is not None:
            status_data["exec_time_ms"] = round(exec_time_ms, 2)
        if resources is not None:
            status_data["resources"] = resources
        if exit_code is not None:
            status_data["exit_code"] = exit_code
        if error is not None:
            status_data["error"] = str(error)

        status_file = self.exp_dir / "status.json"
        try:
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write status.json: {e}")

    def run(self) -> Dict[str, Any]:
        """
        Execute method with monitoring.

        Returns:
            Dictionary with execution summary
        """
        # Save initial running status
        self._save_status("running")

        # Start monitoring in background thread
        # Arm from this thread so a fast-failing method cannot stop the
        # monitor before the worker thread has set its own flag.
        self.monitor.arm()
        monitor_thread = threading.Thread(
            target=self.monitor.start_monitoring,
            args=(self.monitor_interval,),
            daemon=True
        )
        monitor_thread.start()

        # Record start time
        start_time = time.time()

        try:
            logger.info("[INFO] Starting method execution...")

            # Execute command with real-time output using utility function
            return_code, captured_output = run_with_realtime_output(
                command=self.command,
                shell=True,
                cwd=os.getcwd()
            )

            # Record end time
            end_time = time.time()
            exec_time_ms = (end_time - start_time) * 1000

            # Stop monitoring
            self.monitor.stop_monitoring()
            monitor_thread.join(timeout=2)

            # Get resource summary
            resources = self.monitor.get_summary()

            # Log results
            logger.info("")
            logger.info("[INFO] Execution Summary:")
            logger.info(f"   [INFO] Execution time: {exec_time_ms:.2f}ms")
            logger.info(f"   [INFO] Peak memory: {resources['peak_memory_mb']:.2f}MB")
            if resources['peak_gpu_mb'] is not None:
                logger.info(f"   [INFO] Peak GPU memory: {resources['peak_gpu_mb']:.2f}MB")
            logger.info("")

            # Save captured output to file using utility function
            output_file = self.exp_dir / "method_output.txt"
            save_output_to_file(
                output_lines=captured_output,
                output_file=str(output_file),
                header="=== METHOD OUTPUT ===\n"
            )
            logger.info(f"[INFO] Full output saved to: {output_file}")

            if return_code == 0:
                # Exit code 0 is not proof the method produced anything. A
                # method that swallows an exception, ignores finalize()'s
                # "No scores saved" error, or writes outside $EXP exits clean
                # and used to be recorded as completed with an empty
                # experiment directory.
                problem = self._results_problem()
                if problem:
                    logger.error(f"[FAIL] Method exited 0 but {problem}")
                    self._save_status(
                        "failed", exec_time_ms, resources, exit_code=0,
                        error=f"Method exited 0 but {problem}",
                    )
                    raise RuntimeError(f"Method exited 0 but {problem}")
                self._merge_runtime_metadata(exec_time_ms, resources)
                logger.info("[OK] Method execution completed successfully")
                self._save_status("completed", exec_time_ms, resources, exit_code=0)
            else:
                logger.error(f"[FAIL] Method execution failed with exit code {return_code}")
                logger.error(f"[INFO] Check {output_file} for details")
                self._save_status("failed", exec_time_ms, resources, exit_code=return_code)
                raise RuntimeError(f"Method execution failed with exit code {return_code}")

            return {
                "success": True,
                "exec_time_ms": exec_time_ms,
                "resources": resources,
                "output_file": str(output_file)
            }

        except Exception as e:
            # Stop monitoring on error
            self.monitor.stop_monitoring()
            # Always record the failure. The old guard only wrote when the
            # existing status was "running", so if _save_status("running") had
            # failed earlier (read-only mount, disk full -- it only warns) a
            # previous attempt's "completed" survived the crash and the
            # experiment advertised somebody else's success.
            self._save_status(
                "failed",
                (time.time() - start_time) * 1000 if start_time else None,
                error=str(e),
            )
            logger.error(f"[FAIL] Execution error: {e}")
            raise
    
    @classmethod
    def from_env(cls, pass_env_args: bool = False):
        """
        Create runner from environment variables.

        Args:
            pass_env_args: Whether to pass _* env vars as CLI args (default: False)

        Environment variables:
            - DATA: Input dataset path
            - EXP: Experiment output path
            - METHOD_NAME: Method name
            - COMMAND: Command to execute
            - MONITOR_INTERVAL: Resource monitoring interval in seconds (optional, default: 1.0)
            - SUPPORTED_DATASETS: Comma-separated list of compatible dataset patterns (optional)
        """
        data_dir = os.environ.get("DATA")
        exp_dir = os.environ.get("EXP")
        method_name = os.environ.get("METHOD_NAME", "Unknown")
        command = os.environ.get("COMMAND")
        monitor_interval = _parse_monitor_interval(
            os.environ.get("MONITOR_INTERVAL")
        )
        supported_datasets = os.environ.get("SUPPORTED_DATASETS", "")

        if not all([data_dir, exp_dir, command]):
            raise ValueError(
                "Missing required environment variables: DATA, EXP, COMMAND"
            )

        # Validate dataset compatibility if SUPPORTED_DATASETS is specified
        if supported_datasets:
            dataset_name = os.path.basename(data_dir.rstrip('/'))
            patterns = [p.strip() for p in supported_datasets.split(',') if p.strip()]

            # fnmatch handles a wildcard anywhere in the pattern. The previous
            # hand-rolled matcher only understood a trailing '*', so a leading
            # one such as '*_snapshot' -- which methods/example advertises and
            # addgraph, dynwalk and strgnn all declare -- matched nothing and
            # those methods rejected every dataset they were given.
            is_compatible = any(
                fnmatch.fnmatchcase(dataset_name, pattern) for pattern in patterns
            )

            if not is_compatible:
                logger.error(f"[FAIL] Dataset '{dataset_name}' is not compatible with method '{method_name}'")
                logger.error(f"   Supported datasets: {', '.join(patterns)}")
                raise ValueError(
                    f"Dataset '{dataset_name}' is not compatible with method '{method_name}'. "
                    f"Supported datasets: {', '.join(patterns)}"
                )
        
        return cls(
            data_dir=data_dir,
            exp_dir=exp_dir,
            method_name=method_name,
            command=command,
            monitor_interval=monitor_interval,
            pass_env_args=pass_env_args
        )



def _write_early_failure(exc: BaseException) -> None:
    """Record a failure that happened before MethodRunner could start."""
    exp_dir = os.environ.get("EXP")
    if not exp_dir:
        return
    try:
        path = Path(exp_dir)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method_name": os.environ.get("METHOD_NAME", "unknown"),
            "error": f"{type(exc).__name__}: {exc}",
            "stage": "startup",
        }
        (path / "status.json").write_text(json.dumps(payload, indent=2))
    except Exception as write_error:      # never mask the original failure
        logger.warning(f"Could not record startup failure: {write_error}")


def main():
    """CLI entry point for running as module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GraFlag Method Runner")
    parser.add_argument(
        "--pass-env-args",
        action="store_true",
        help="Extract environment variables starting with _ and pass as CLI arguments"
    )
    
    args = parser.parse_args()
    
    try:
        runner = MethodRunner.from_env(pass_env_args=args.pass_env_args)
        summary = runner.run()

        logger.info("[OK] Runner completed successfully")
        sys.exit(0)

    except Exception as e:
        logger.error(f"[FAIL] Runner failed: {e}")
        # from_env() and the constructor run before run() writes anything, so
        # a failure here (an unusable MONITOR_INTERVAL, an unsupported
        # dataset, a ResourceMonitor that cannot start) left $EXP completely
        # empty. The container log is discarded once the Swarm task is reaped,
        # so the orchestrator had nothing to report but "unknown".
        _write_early_failure(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
