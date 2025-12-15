"""Utility functions for model validation and inspection"""
import os
import coremltools as ct
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

def validate_model(model_path):
    """Validate a Core ML model package"""
    try:
        if not os.path.exists(model_path):
            console.print(f"[red]Error: Model not found at {model_path}[/red]")
            return False
        
        if not model_path.endswith('.mlpackage'):
            console.print(f"[yellow]Warning: Expected .mlpackage, got {model_path}[/yellow]")
        
        console.print(f"[cyan]Validating model:[/cyan] {model_path}")
        
        # Load model
        model = ct.models.MLModel(model_path)
        
        # Basic checks
        spec = model.get_spec()
        console.print(f"[green]✓[/green] Model loads successfully")
        console.print(f"[green]✓[/green] Model type: {spec.WhichOneof('Type')}")
        
        # Input/Output validation
        if hasattr(spec.description, 'input'):
            console.print(f"[green]✓[/green] Inputs: {len(spec.description.input)} tensors")
        if hasattr(spec.description, 'output'):
            console.print(f"[green]✓[/green] Outputs: {len(spec.description.output)} tensors")
        
        console.print(f"[bold green]✓ Model validation passed![/bold green]")
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Model validation failed:[/red] {str(e)}")
        return False


def show_model_info(model_path):
    """Display detailed model information"""
    try:
        model = ct.models.MLModel(model_path)
        spec = model.get_spec()
        
        # Create info table
        table = Table(title=f"Model Info: {Path(model_path).name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Model Type", spec.WhichOneof('Type'))
        table.add_row("Path", str(model_path))
        
        # File size
        size_mb = sum(f.stat().st_size for f in Path(model_path).rglob('*') if f.is_file()) / (1024 * 1024)
        table.add_row("Size", f"{size_mb:.2f} MB")
        
        # Inputs
        if hasattr(spec.description, 'input'):
            input_info = []
            for inp in spec.description.input:
                shape_str = f"{inp.name}: {inp.type.WhichOneof('Type')}"
                input_info.append(shape_str)
            table.add_row("Inputs", "\\n".join(input_info[:5]))  # Show first 5
        
        # Outputs
        if hasattr(spec.description, 'output'):
            output_info = []
            for out in spec.description.output:
                shape_str = f"{out.name}: {out.type.WhichOneof('Type')}"
                output_info.append(shape_str)
            table.add_row("Outputs", "\\n".join(output_info[:5]))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error reading model info:[/red] {str(e)}")


def list_models(directory):
    """List all Core ML models in a directory"""
    try:
        directory = Path(directory)
        if not directory.exists():
            console.print(f"[red]Directory not found:[/red] {directory}")
            return
        
        # Find all .mlpackage files
        models = list(directory.rglob("*.mlpackage"))
        
        if not models:
            console.print(f"[yellow]No models found in {directory}[/yellow]")
            return
        
        # Create table
        table = Table(title=f"Converted Models in {directory}")
        table.add_column("Model", style="cyan")
        table.add_column("Size", style="magenta")
        table.add_column("Modified", style="green")
        
        for model_path in sorted(models):
            size_mb = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024 * 1024)
            modified = model_path.stat().st_mtime
            from datetime import datetime
            modified_str = datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M')
            
            rel_path = model_path.relative_to(directory)
            table.add_row(str(rel_path), f"{size_mb:.1f} MB", modified_str)
        
        console.print(table)
        console.print(f"\\n[green]Found {len(models)} model(s)[/green]")
        
    except Exception as e:
        console.print(f"[red]Error listing models:[/red] {str(e)}")
