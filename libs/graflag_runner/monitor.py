"""Resource monitoring for method execution."""

import os
import time
import psutil
import subprocess
from typing import Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Import ResultWriter for spot() functionality
from .results import ResultWriter


class ResourceMonitor:
    """Monitor CPU, memory, and GPU usage during method execution."""
    
    def __init__(self, pid: Optional[int] = None):
        """
        Initialize resource monitor.
        
        Args:
            pid: Process ID to monitor (None = current process)
        """
        self.pid = pid or os.getpid()
        self.monitoring = False
        self.peak_memory_mb = 0
        self.peak_gpu_mb = 0
        self.has_gpu = self._check_gpu()
        
        # System totals
        self.total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        self.total_gpu_mb = self._get_total_gpu_memory() if self.has_gpu else 0
        
        # ResultWriter for spot() functionality
        self.result_writer = ResultWriter()
        
    def _check_gpu(self) -> bool:
        """Check if NVIDIA GPU is available."""
        try:
            subprocess.run(
                ["nvidia-smi"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                timeout=2
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _get_total_gpu_memory(self) -> float:
        """Get total GPU memory in MB."""
        if not self.has_gpu:
            return 0
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2
            )
            return float(result.stdout.strip().split('\n')[0])
        except Exception:
            return 0
    
    def _get_process_memory(self) -> float:
        """Get current process memory usage in MB (including children)."""
        try:
            process = psutil.Process(self.pid)
            # Get memory for main process
            memory = process.memory_info().rss / (1024 * 1024)
            # Add memory for all child processes
            for child in process.children(recursive=True):
                try:
                    memory += child.memory_info().rss / (1024 * 1024)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return memory
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0
    
    def _get_gpu_memory(self) -> float:
        """Get GPU memory usage in MB (all processes in container)."""
        if not self.has_gpu:
            return 0
        try:
            # In a container, we're the only workload, so track total GPU usage
            # This is more reliable than trying to track specific PIDs since
            # PyTorch/CUDA may spawn processes that aren't direct children
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip().split('\n')[0])
            
            return 0.0
        except Exception as e:
            logger.debug(f"Failed to get GPU memory: {e}")
            return 0
    
    def start_monitoring(self, interval: float = 1.0):
        """
        Start background monitoring loop.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.monitoring = True
        
        logger.info(f"[INFO] Resource monitoring started (PID: {self.pid})")
        logger.info(f"   Total memory: {self.total_memory_mb:.0f}MB")
        if self.has_gpu:
            logger.info(f"   Total GPU memory: {self.total_gpu_mb:.0f}MB")
        else:
            logger.info("   GPU: Not available")
        
        while self.monitoring:
            current_memory = self._get_process_memory()
            current_gpu = self._get_gpu_memory()
            
            # Update peaks
            self.peak_memory_mb = max(self.peak_memory_mb, current_memory)
            self.peak_gpu_mb = max(self.peak_gpu_mb, current_gpu)
            
            # Log to CSV using spot() method
            if self.result_writer:
                self.result_writer.spot(
                    "resources",
                    memory_mb=round(current_memory, 2),
                    gpu_mb=round(current_gpu, 2)
                )
            
            time.sleep(interval)
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get resource usage summary.
        
        Returns:
            Dictionary with peak memory and GPU usage
        """
        return {
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "peak_gpu_mb": round(self.peak_gpu_mb, 2) if self.has_gpu else None,
            "total_memory_mb": round(self.total_memory_mb, 2),
            "total_gpu_mb": round(self.total_gpu_mb, 2) if self.has_gpu else None,
        }
