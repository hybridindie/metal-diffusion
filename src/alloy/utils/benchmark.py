"""Benchmarking utilities for measuring real performance"""
import time
import psutil
import json
from statistics import mean, median, stdev
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

class Benchmark:
    """Performance benchmark tracker"""
    def __init__(self, name="Benchmark"):
        self.name = name
        self.runs = []
        self.current_run = {}
        self.start_time = None
        self.process = psutil.Process()
        
    def start_run(self):
        """Start a new benchmark run"""
        self.current_run = {
            "steps": {},
            "total_time": 0,
            "peak_memory_mb": 0,
            "start_memory_mb": self.process.memory_info().rss / 1024 / 1024
        }
        self.start_time = time.time()
        
    def step(self, name):
        """Start timing a step"""
        return BenchmarkStep(self, name)
        
    def end_run(self):
        """End current run and record results"""
        self.current_run["total_time"] = time.time() - self.start_time
        self.current_run["end_memory_mb"] = self.process.memory_info().rss / 1024 / 1024
        self.current_run["peak_memory_mb"] = max(
            self.current_run["peak_memory_mb"],
            self.current_run["end_memory_mb"]
        )
        self.runs.append(self.current_run)
        self.current_run = {}
        
    def update_memory(self):
        """Update peak memory usage"""
        if self.current_run:
            current_mem = self.process.memory_info().rss / 1024 / 1024
            self.current_run["peak_memory_mb"] = max(
                self.current_run.get("peak_memory_mb", 0),
                current_mem
            )
    
    def get_stats(self):
        """Calculate statistics across all runs"""
        if not self.runs:
            return None
            
        stats = {
            "num_runs": len(self.runs),
            "total_time": {
                "mean": mean(r["total_time"] for r in self.runs),
                "median": median(r["total_time"] for r in self.runs),
                "min": min(r["total_time"] for r in self.runs),
                "max": max(r["total_time"] for r in self.runs),
                "stdev": stdev(r["total_time"] for r in self.runs) if len(self.runs) > 1 else 0
            },
            "peak_memory": {
                "mean": mean(r["peak_memory_mb"] for r in self.runs),
                "median": median(r["peak_memory_mb"] for r in self.runs),
                "min": min(r["peak_memory_mb"] for r in self.runs),
                "max": max(r["peak_memory_mb"] for r in self.runs)
            }
        }
        
        # Calculate step statistics
        step_names = set()
        for run in self.runs:
            step_names.update(run["steps"].keys())
            
        stats["steps"] = {}
        for step_name in step_names:
            times = [r["steps"][step_name] for r in self.runs if step_name in r["steps"]]
            if times:
                stats["steps"][step_name] = {
                    "mean": mean(times),
                    "median": median(times),
                    "min": min(times),
                    "max": max(times)
                }
        
        return stats
    
    def print_results(self):
        """Print formatted benchmark results"""
        stats = self.get_stats()
        if not stats:
            console.print("[red]No benchmark data collected[/red]")
            return
            
        # Summary table
        table = Table(title=f"Benchmark Results: {self.name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", style="magenta")
        table.add_column("Median", style="green")
        table.add_column("Min/Max", style="yellow")
        
        # Total time
        t = stats["total_time"]
        table.add_row(
            "Total Time",
            f"{t['mean']:.2f}s",
            f"{t['median']:.2f}s",
            f"{t['min']:.2f}s / {t['max']:.2f}s"
        )
        
        # Memory
        m = stats["peak_memory"]
        table.add_row(
            "Peak Memory",
            f"{m['mean']:.1f} MB",
            f"{m['median']:.1f} MB",
            f"{m['min']:.1f} MB / {m['max']:.1f} MB"
        )
        
        console.print(table)
        
        # Step breakdown
        if stats["steps"]:
            console.print("\\n[bold cyan]Step Breakdown:[/bold cyan]")
            step_table = Table()
            step_table.add_column("Step", style="cyan")
            step_table.add_column("Mean", style="magenta")
            step_table.add_column("% of Total", style="yellow")
            
            total_mean = stats["total_time"]["mean"]
            for step_name, step_stats in sorted(stats["steps"].items()):
                percentage = (step_stats["mean"] / total_mean) * 100
                step_table.add_row(
                    step_name,
                    f"{step_stats['mean']:.3f}s",
                    f"{percentage:.1f}%"
                )
            
            console.print(step_table)
        
        console.print(f"\\n[green]Benchmark complete![/green] ({stats['num_runs']} runs)")
        
    def save_json(self, filename):
        """Save benchmark results to JSON"""
        stats = self.get_stats()
        stats["name"] = self.name
        stats["raw_runs"] = self.runs
        
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        console.print(f"[green]Saved benchmark data to {filename}[/green]")


class BenchmarkStep:
    """Context manager for timing a step"""
    def __init__(self, benchmark, name):
        self.benchmark = benchmark
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        self.benchmark.current_run["steps"][self.name] = elapsed
        self.benchmark.update_memory()
