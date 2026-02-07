"""Web app serving command"""
import click
import subprocess
from pathlib import Path

@click.command()
@click.option('--port', default=8501, help='Port number')
@click.option('--config', type=click.Path(), default='configs/default.yaml',
              help='Configuration file')
def serve(port, config):
    """Launch the Streamlit web dashboard"""
    click.echo(f"🚀 Starting dashboard on port {port}...")
    
    app_path = Path(__file__).parent.parent / "ui" / "app.py"
    
    if not app_path.exists():
        click.echo(f"❌ App not found: {app_path}", err=True)
        return 1
    
    subprocess.run([
        "streamlit", "run", str(app_path),
        "--server.port", str(port),
        "--", "--config", config
    ])
