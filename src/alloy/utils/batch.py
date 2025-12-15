"""Batch conversion utilities"""
import json
import yaml
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

console = Console()

def parse_batch_file(batch_file):
    """
    Parse batch conversion file (JSON or YAML).
    
    Expected format:
    [
      {
        "model": "black-forest-labs/FLUX.1-schnell",
        "type": "flux",
        "output_dir": "converted_models/flux_schnell",
        "quantization": "int4"
      },
      ...
    ]
    """
    path = Path(batch_file)
    if not path.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_file}")
    
    content = path.read_text()
    
    # Try JSON first
    if path.suffix in ['.json', '.jsonl']:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    # Try YAML
    elif path.suffix in ['.yaml', '.yml']:
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}")
    else:
        # Try both
        try:
            data = json.loads(content)
        except:
            try:
                data = yaml.safe_load(content)
            except:
                raise ValueError("File must be JSON or YAML format")
    
    if not isinstance(data, list):
        raise ValueError("Batch file must contain a list of model configs")
    
    return data


def validate_batch_config(config):
    """Validate a single batch config entry"""
    required = ['model', 'type']
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    # Set defaults
    config.setdefault('output_dir', f"converted_models/{config['model'].split('/')[-1]}")
    config.setdefault('quantization', 'float16')
    
    return config


def run_batch_conversion(batch_file, dry_run=False, parallel=False):
    """
    Run batch conversion from file.
    
    Args:
        batch_file: Path to JSON/YAML file with model configs
        dry_run: If True, only print what would be converted
        parallel: If True, run conversions in parallel (experimental)
    """
    console.print(f"[cyan]Loading batch file:[/cyan] {batch_file}")
    
    try:
        configs = parse_batch_file(batch_file)
    except Exception as e:
        console.print(f"[red]Error parsing batch file:[/red] {e}")
        return False
    
    console.print(f"[green]Found {len(configs)} model(s) to convert[/green]")
    
    # Validate all configs first
    validated_configs = []
    for i, config in enumerate(configs, 1):
        try:
            validated = validate_batch_config(config)
            validated_configs.append(validated)
            console.print(f"  [{i}] {validated['model']} → {validated['output_dir']}")
        except Exception as e:
            console.print(f"[red]  [{i}] Invalid config:[/red] {e}")
            return False
    
    if dry_run:
        console.print("\n[yellow]Dry run - no conversions performed[/yellow]")
        return True
    
    # Run conversions
    if parallel:
        console.print("\n[yellow]⚠ Parallel conversion is experimental and may cause high memory usage[/yellow]")
        # TODO: Implement parallel conversion with multiprocessing
        console.print("[red]Parallel mode not yet implemented, falling back to sequential[/red]")
    
    # Sequential conversion
    console.print(f"\n[cyan]Starting sequential conversion...[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        overall = progress.add_task("[cyan]Overall Progress", total=len(validated_configs))
        
        success_count = 0
        failed = []
        
        for i, config in enumerate(validated_configs, 1):
            model = config['model']
            progress.update(overall, description=f"Converting {i}/{len(validated_configs)}: {model}")
            
            try:
                # Import here to avoid circular imports
                from .converter import SDConverter
                from .flux_converter import FluxConverter
                from .ltx_converter import LTXConverter
                from .wan_converter import WanConverter
                from .hunyuan_converter import HunyuanConverter
                
                # Create appropriate converter
                converter_map = {
                    'sd': SDConverter,
                    'flux': FluxConverter,
                    'ltx': LTXConverter,
                    'wan': WanConverter,
                    'hunyuan': HunyuanConverter
                }
                
                converter_class = converter_map.get(config['type'])
                if not converter_class:
                    raise ValueError(f"Unknown model type: {config['type']}")
                
                converter = converter_class(
                    config['model'],
                    config['output_dir'],
                    config['quantization']
                )
                
                console.print(f"\n[bold cyan]Converting {model}...[/bold cyan]")
                converter.convert()
                console.print(f"[green]✓ Completed: {model}[/green]")
                success_count += 1
                
            except Exception as e:
                console.print(f"[red]✗ Failed: {model}[/red]")
                console.print(f"[red]  Error: {e}[/red]")
                failed.append((model, str(e)))
            
            progress.update(overall, advance=1)
    
    # Summary
    console.print(f"\n[bold]Batch Conversion Summary:[/bold]")
    console.print(f"  [green]Success:[/green] {success_count}/{len(validated_configs)}")
    if failed:
        console.print(f"  [red]Failed:[/red] {len(failed)}")
        for model, error in failed:
            console.print(f"    - {model}: {error}")
    
    return len(failed) == 0
