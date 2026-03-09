"""Subprocess utilities for real-time output streaming."""

import os
import sys
import subprocess
from typing import List, Tuple, Optional, Dict


def run_with_realtime_output(
    command: str,
    shell: bool = True,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    stdin_passthrough: bool = True
) -> Tuple[int, List[str]]:
    """
    Run a command with real-time output streaming while capturing all output.
    
    This function streams subprocess output in real-time to stdout while also
    capturing it for later use. Optionally forwards stdin from parent to subprocess,
    enabling interactive processes and piped input.
    
    Args:
        command: Command to execute (string if shell=True, list if shell=False)
        shell: Whether to execute through shell (default: True)
        cwd: Working directory for the command (default: current directory)
        env: Environment variables dict (default: copy of current environment)
        stdin_passthrough: Forward stdin from parent to subprocess (default: True)
                          Set to False to disable stdin (subprocess gets None)
    
    Returns:
        Tuple of (return_code, captured_output_lines)
        - return_code: Exit code of the process
        - captured_output_lines: List of output lines (includes newlines)
    
    Example:
        >>> # Basic usage
        >>> return_code, output = run_with_realtime_output("python train.py")
        >>> if return_code == 0:
        >>>     with open("log.txt", "w") as f:
        >>>         f.writelines(output)
        
        >>> # With piped input
        >>> return_code, output = run_with_realtime_output(
        >>>     "python process.py",
        >>>     stdin_passthrough=True  # Allows: echo "data" | python wrapper.py
        >>> )
        
        >>> # Interactive process
        >>> return_code, output = run_with_realtime_output(
        >>>     "python interactive.py",
        >>>     stdin_passthrough=True  # User can type input
        >>> )
    """
    # Set environment variables to disable Python output buffering
    if env is None:
        env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    # Determine stdin handling
    # If stdin_passthrough=True, inherit parent's stdin (allows piped input & interactive)
    # If stdin_passthrough=False, subprocess gets no input (stdin=None)
    stdin_config = None if stdin_passthrough else subprocess.DEVNULL
    
    # Start subprocess with piped output and configurable stdin
    process = subprocess.Popen(
        command,
        shell=shell,
        stdin=stdin_config,  # Inherit stdin or disable it
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True,
        env=env,
        cwd=cwd
    )
    
    # Capture output while streaming it in real-time
    captured_output = []
    for line in process.stdout:
        # Print to console (forwarded to parent)
        print(line, end='', flush=True)
        sys.stdout.flush()  # Force flush
        # Capture for later use
        captured_output.append(line)
    
    # Wait for process to complete
    return_code = process.wait()
    
    return return_code, captured_output


def run_command_list(
    commands: List[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    stop_on_error: bool = True,
    stdin_passthrough: bool = False
) -> List[Tuple[str, int, List[str]]]:
    """
    Run multiple commands sequentially with real-time output.
    
    Args:
        commands: List of commands to execute
        cwd: Working directory for all commands
        env: Environment variables dict
        stop_on_error: If True, stop executing after first failure
        stdin_passthrough: Forward stdin to commands (default: False for batch jobs)
    
    Returns:
        List of tuples: [(command, return_code, output_lines), ...]
    
    Example:
        >>> results = run_command_list([
        >>>     "python prepare_data.py",
        >>>     "python train_model.py"
        >>> ])
        >>> for cmd, code, output in results:
        >>>     if code != 0:
        >>>         print(f"Failed: {cmd}")
    """
    results = []
    
    for command in commands:
        print(f"\n{'='*60}")
        print(f"Running: {command}")
        print(f"{'='*60}\n")
        
        return_code, output = run_with_realtime_output(
            command=command,
            cwd=cwd,
            env=env,
            stdin_passthrough=stdin_passthrough
        )
        
        results.append((command, return_code, output))
        
        if stop_on_error and return_code != 0:
            print(f"\n[ERROR] Command failed with exit code {return_code}")
            break
    
    return results


def save_output_to_file(
    output_lines: List[str],
    output_file: str,
    header: str = "=== OUTPUT ===\n"
) -> None:
    """
    Save captured output to a file.
    
    Args:
        output_lines: List of output lines to save
        output_file: Path to output file
        header: Optional header to prepend to the file
    
    Example:
        >>> return_code, output = run_with_realtime_output("python train.py")
        >>> save_output_to_file(output, "training.log", "=== TRAINING LOG ===\n")
    """
    with open(output_file, 'w') as f:
        if header:
            f.write(header)
        f.writelines(output_lines)
