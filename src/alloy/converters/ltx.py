import coremltools as ct
import multiprocessing
import os
import shutil
import gc

from rich.console import Console

from .base import ModelConverter
from alloy.converters.ltx_workers import convert_ltx_part1, convert_ltx_part2

console = Console()


class LTXConverter(ModelConverter):
    """
    Converter for LTX Video models.
    Uses 2-phase subprocess isolation (split at midpoint of blocks) to prevent OOM.
    """

    def __init__(self, model_id, output_dir, quantization):
        # Allow user to specify Lightricks or other repo
        if "/" not in model_id and not os.path.isfile(model_id):
            model_id = "Lightricks/LTX-Video"
        super().__init__(model_id, output_dir, quantization)

    def convert(self):
        """Main conversion entry point using 2-phase subprocess pattern."""
        os.makedirs(self.output_dir, exist_ok=True)
        ml_model_dir = os.path.join(self.output_dir, f"LTXVideo_Transformer_{self.quantization}.mlpackage")

        if os.path.exists(ml_model_dir):
            console.print(f"[yellow]Model exists, skipping:[/yellow] {ml_model_dir}")
            return

        # Use persistent intermediate directory
        intermediates_dir = os.path.join(self.output_dir, "intermediates")
        os.makedirs(intermediates_dir, exist_ok=True)

        # Download source weights if needed
        self.model_id = self.download_source_weights(self.model_id, self.output_dir)

        try:
            # Paths for intermediate parts
            part1_path = os.path.join(intermediates_dir, "LTXPart1.mlpackage")
            part2_path = os.path.join(intermediates_dir, "LTXPart2.mlpackage")

            # --- Part 1: First half of blocks ---
            skip_p1 = False
            if os.path.exists(part1_path):
                try:
                    console.print(f"[dim]Checking existing Part 1 at {part1_path}...[/dim]")
                    ct.models.MLModel(part1_path, compute_units=ct.ComputeUnit.CPU_ONLY)
                    console.print("[green]Found valid Part 1 intermediate. Resuming...[/green]")
                    skip_p1 = True
                except Exception:
                    console.print("[yellow]Found invalid/incomplete Part 1. Re-converting...[/yellow]")
                    shutil.rmtree(part1_path)

            if not skip_p1:
                console.print("\n[bold]Spawning Part 1 Conversion Process (First Half of Blocks)...[/bold]")
                p1 = multiprocessing.Process(
                    target=convert_ltx_part1,
                    args=(self.model_id, part1_path, self.quantization),
                    kwargs={"intermediates_dir": intermediates_dir}
                )
                p1.start()
                p1.join()

                if p1.exitcode != 0:
                    raise RuntimeError("LTX Part 1 Worker Failed")

            # --- Part 2: Second half of blocks ---
            skip_p2 = False
            if os.path.exists(part2_path):
                try:
                    console.print(f"[dim]Checking existing Part 2 at {part2_path}...[/dim]")
                    ct.models.MLModel(part2_path, compute_units=ct.ComputeUnit.CPU_ONLY)
                    console.print("[green]Found valid Part 2 intermediate. Resuming...[/green]")
                    skip_p2 = True
                except Exception:
                    console.print("[yellow]Found invalid/incomplete Part 2. Re-converting...[/yellow]")
                    shutil.rmtree(part2_path)

            if not skip_p2:
                console.print("\n[bold]Spawning Part 2 Conversion Process (Second Half of Blocks)...[/bold]")
                p2 = multiprocessing.Process(
                    target=convert_ltx_part2,
                    args=(self.model_id, part2_path, self.quantization),
                    kwargs={"intermediates_dir": intermediates_dir}
                )
                p2.start()
                p2.join()

                if p2.exitcode != 0:
                    raise RuntimeError("LTX Part 2 Worker Failed")

            # --- Assemble Pipeline ---
            console.print("\n[bold]Assembling Pipeline...[/bold]")

            # Load lazily from disk with CPU_ONLY
            m1 = ct.models.MLModel(part1_path, compute_units=ct.ComputeUnit.CPU_ONLY)
            m2 = ct.models.MLModel(part2_path, compute_units=ct.ComputeUnit.CPU_ONLY)

            pipeline_model = ct.utils.make_pipeline(m1, m2)

            # Add metadata
            pipeline_model.author = "Alloy"
            pipeline_model.license = "Apache 2.0"
            pipeline_model.short_description = f"LTX Video Transformer (Split Pipeline) {self.quantization}"

            # Cleanup intermediates BEFORE saving final pipeline
            console.print("[dim]Releasing intermediate memory/disk for final save...[/dim]")
            del m1, m2
            gc.collect()

            try:
                shutil.rmtree(intermediates_dir)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not clear intermediates: {e}[/yellow]")

            console.print(f"[dim]Saving final pipeline to {ml_model_dir}...[/dim]")
            pipeline_model.save(ml_model_dir)

            console.print(f"[bold green]✓ LTX Video conversion complete![/bold green] Saved to {self.output_dir}")

        except Exception:
            console.print(f"[yellow]Note: Intermediate files left in {intermediates_dir} for inspection/cleanup.[/yellow]")
            raise
