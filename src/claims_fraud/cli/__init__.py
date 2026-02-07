"""CLI module for claims-fraud"""
import click
import subprocess
import sys
from pathlib import Path


@click.group()
@click.version_option(version='0.1.0')
def main():
    """Claims Fraud Detection - Fraud detection system for insurance claims"""
    pass


@click.command()
@click.option('--port', default=8501, help='Port to run on')
def serve(port):
    """Launch the Streamlit dashboard"""
    
    click.echo(f"🚀 Starting dashboard on port {port}...")
    
    # Use the __main__.py launcher which sets up imports correctly
    launcher_path = Path(__file__).parent.parent / "ui" / "__main__.py"
    
    if not launcher_path.exists():
        click.echo(f"❌ Dashboard launcher not found at: {launcher_path}")
        sys.exit(1)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(launcher_path),
            "--server.port", str(port)
        ])
    except KeyboardInterrupt:
        click.echo("\n👋 Dashboard stopped")
    except Exception as e:
        click.echo(f"❌ Error launching dashboard: {e}")
        sys.exit(1)


main.add_command(serve)


if __name__ == '__main__':
    main()
