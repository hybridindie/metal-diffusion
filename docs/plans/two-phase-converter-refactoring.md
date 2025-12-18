# TwoPhaseConverter Base Class Refactoring

## Overview

Refactor the duplicate 2-phase subprocess conversion pattern from 5 converters (Flux, Hunyuan, LTX, Wan, Lumina) into a reusable `TwoPhaseConverter` base class.

## Problem

Each of the 5 converters duplicates ~80 lines of identical logic:
- Part 1/Part 2 intermediate validation
- Subprocess spawning with `multiprocessing.Process`
- Pipeline assembly with `ct.utils.make_pipeline`
- Cleanup and metadata

Total: ~500 lines of duplicated code.

## Solution

Create `TwoPhaseConverter(ModelConverter)` base class with:
- Abstract methods for model-specific workers
- Template method pattern for the conversion flow
- Helper methods for common operations

## Design

### Base Class Interface

```python
class TwoPhaseConverter(ModelConverter):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name for filenames and messages (e.g., 'Flux', 'Hunyuan')."""
        pass

    @abstractmethod
    def get_part1_worker(self) -> Callable:
        """Return the worker function for Part 1 conversion."""
        pass

    @abstractmethod
    def get_part2_worker(self) -> Callable:
        """Return the worker function for Part 2 conversion."""
        pass

    @property
    def should_download_source(self) -> bool:
        """Override to disable automatic source download. Default: True."""
        return True

    @property
    def part1_description(self) -> str:
        """Override for custom Part 1 description. Default: 'Part 1'."""
        return "Part 1"

    @property
    def part2_description(self) -> str:
        """Override for custom Part 2 description. Default: 'Part 2'."""
        return "Part 2"
```

### Derived Paths

```python
@property
def output_filename(self) -> str:
    return f"{self.model_name}_Transformer_{self.quantization}.mlpackage"

@property
def part1_filename(self) -> str:
    return f"{self.model_name}Part1.mlpackage"

@property
def part2_filename(self) -> str:
    return f"{self.model_name}Part2.mlpackage"
```

### Convert Method (Template)

```python
def convert(self):
    os.makedirs(self.output_dir, exist_ok=True)
    ml_model_path = os.path.join(self.output_dir, self.output_filename)

    if os.path.exists(ml_model_path):
        console.print(f"[yellow]Model exists, skipping:[/yellow] {ml_model_path}")
        return

    intermediates_dir = os.path.join(self.output_dir, "intermediates")
    os.makedirs(intermediates_dir, exist_ok=True)

    if self.should_download_source:
        self.model_id = self.download_source_weights(self.model_id, self.output_dir)

    try:
        part1_path = os.path.join(intermediates_dir, self.part1_filename)
        part2_path = os.path.join(intermediates_dir, self.part2_filename)

        self._convert_part(1, part1_path, self.get_part1_worker(), intermediates_dir)
        self._convert_part(2, part2_path, self.get_part2_worker(), intermediates_dir)
        self._assemble_pipeline(part1_path, part2_path, ml_model_path, intermediates_dir)

    except Exception:
        console.print(f"[yellow]Note: Intermediate files left in {intermediates_dir}[/yellow]")
        raise
```

### Helper Methods

```python
def _convert_part(self, part_num: int, output_path: str, worker_fn: Callable, intermediates_dir: str) -> None:
    """Convert a single part with validation and subprocess isolation."""
    # Validate existing intermediate or re-convert
    # Spawn subprocess with worker_fn
    # Raise on failure

def _assemble_pipeline(self, part1_path: str, part2_path: str, output_path: str, intermediates_dir: str) -> None:
    """Load parts, assemble pipeline, cleanup, and save."""
    # Load with CPU_ONLY
    # Make pipeline
    # Add metadata
    # Cleanup intermediates
    # Save final model
```

## Subclass Examples

### FluxConverter (minimal)

```python
class FluxConverter(TwoPhaseConverter):
    @property
    def model_name(self) -> str:
        return "Flux"

    @property
    def should_download_source(self) -> bool:
        return False  # Flux handles downloads differently

    def get_part1_worker(self) -> Callable:
        return convert_flux_part1

    def get_part2_worker(self) -> Callable:
        return convert_flux_part2
```

### HunyuanConverter (with custom descriptions)

```python
class HunyuanConverter(TwoPhaseConverter):
    @property
    def model_name(self) -> str:
        return "Hunyuan"

    @property
    def part1_description(self) -> str:
        return "Part 1 (Dual-Stream Blocks)"

    @property
    def part2_description(self) -> str:
        return "Part 2 (Single-Stream Blocks)"

    def get_part1_worker(self) -> Callable:
        return convert_hunyuan_part1

    def get_part2_worker(self) -> Callable:
        return convert_hunyuan_part2
```

## Migration Plan

1. Add `TwoPhaseConverter` to `base.py`
2. Refactor each converter one at a time:
   - FluxConverter
   - HunyuanConverter
   - LTXConverter
   - WanConverter
   - LuminaConverter
3. Run tests after each refactor
4. Remove duplicate code

## Expected Results

- ~500 lines removed across 5 files
- Each converter reduced from ~100-150 lines to ~30-50 lines
- Consistent behavior across all converters
- Easier to add new 2-phase converters
