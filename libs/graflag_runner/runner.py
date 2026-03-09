"""Main method runner with resource monitoring."""

import os
import sys
import json
import time
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from .monitor import ResourceMonitor
from .results import ResultWriter
from .subprocess_utils import run_with_realtime_output, save_output_to_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

        for key, value in os.environ.items():
            if key.startswith("_"):
                # Remove leading underscore and convert to lowercase
                arg_name = key[1:].lower()
                env_args.append(f"--{arg_name} {value}")

        if env_args:
            args_str = " ".join(env_args)
            logger.info(f"[INFO] Extracted env args: {args_str}")
            return f"{self.command} {args_str}"

        return self.command
    
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
            # Save failed status (only if not already saved by return_code check)
            status_file = self.exp_dir / "status.json"
            try:
                existing = json.loads(status_file.read_text())
                if existing.get("status") == "running":
                    end_time = time.time()
                    self._save_status("failed", (end_time - start_time) * 1000, error=str(e))
            except Exception:
                self._save_status("failed", error=str(e))
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
        monitor_interval = float(os.environ.get("MONITOR_INTERVAL", "1.0"))
        supported_datasets = os.environ.get("SUPPORTED_DATASETS", "")

        if not all([data_dir, exp_dir, command]):
            raise ValueError(
                "Missing required environment variables: DATA, EXP, COMMAND"
            )

        # Validate dataset compatibility if SUPPORTED_DATASETS is specified
        if supported_datasets:
            dataset_name = os.path.basename(data_dir.rstrip('/'))
            patterns = [p.strip() for p in supported_datasets.split(',') if p.strip()]

            is_compatible = False
            for pattern in patterns:
                # Support wildcard patterns (e.g., "generaldyg_*", "btc_*")
                if pattern.endswith('*'):
                    prefix = pattern[:-1]
                    if dataset_name.startswith(prefix):
                        is_compatible = True
                        break
                elif pattern == dataset_name:
                    is_compatible = True
                    break

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
        sys.exit(1)


if __name__ == "__main__":
    main()
