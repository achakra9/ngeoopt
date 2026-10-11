"""Command line interface: ``ngeoopt run input.yaml`` etc."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import __version__

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Parallel finite-difference geometry optimizer.")


@app.command()
def run(config: Path = typer.Argument(..., exists=True, help="YAML input file"),
        task: Optional[str] = typer.Option(None, help="override task: optimize|gradient|energy"),
        workers: Optional[str] = typer.Option(None, help="override parallel.workers"),
        device: Optional[str] = typer.Option(None, help="override parallel.device"),
        no_restart: bool = typer.Option(False, "--no-restart", help="ignore checkpoint")):
    """Run a calculation described by a YAML file."""
    import yaml

    from .config import config_from_dict
    from .driver import run as _run

    data = yaml.safe_load(config.read_text()) or {}
    if task:
        data["task"] = task
    par = data.setdefault("parallel", {}) or {}
    data["parallel"] = par
    if workers:
        par["workers"] = int(workers) if workers.isdigit() else workers
    if device:
        par["device"] = device
    if no_restart:
        data.setdefault("output", {})["restart"] = False
    cfg = config_from_dict(data, base_dir=config.parent.resolve())
    res = _run(cfg)
    if res.optimization is not None and not res.optimization.converged:
        raise typer.Exit(code=2)


@app.command()
def init(path: Path = typer.Argument(Path("ngeoopt.yaml"), help="where to write the template")):
    """Write a fully commented example input file."""
    from .config import EXAMPLE_CONFIG
    if path.exists():
        typer.echo(f"{path} exists; not overwriting", err=True)
        raise typer.Exit(code=1)
    path.write_text(EXAMPLE_CONFIG)
    typer.echo(f"wrote {path}")


@app.command()
def devices():
    """Show which accelerators and optional packages ngeoopt can see."""
    from .device import summary
    typer.echo(summary())


@app.command()
def status(workdir: Path = typer.Argument(Path("ngeoopt_run"))):
    """Summarise the checkpoint of a (possibly running) optimization."""
    side = workdir / "ngeoopt_state.json"
    if not side.exists():
        typer.echo(f"no checkpoint in {workdir}", err=True)
        raise typer.Exit(code=1)
    meta = json.loads(side.read_text())
    typer.echo(f"cycle {meta['cycle']}  converged={meta['converged']}  "
               f"trust={meta['trust']:.3f}")
    typer.echo(f"{'cyc':>4} {'energy':>18} {'maxF':>9} {'rmsF':>9} {'step':>9} "
               f"{'root':>4} {'jobs':>5} {'ok':>3}")
    for h in meta.get("history", []):
        typer.echo(f"{h['cycle']:4d} {h['energy']:18.10f} {h['max_force']:9.2e} "
                   f"{h['rms_force']:9.2e} {h['step_norm']:9.2e} {h['root']:4d} "
                   f"{h['njobs']:5d} {'y' if h['accepted'] else 'n':>3}")


@app.command()
def version():
    typer.echo(__version__)


def main():
    app()


if __name__ == "__main__":
    main()
