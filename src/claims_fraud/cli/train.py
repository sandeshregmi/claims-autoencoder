"""Training CLI command"""
import click
from pathlib import Path

@click.command()
@click.option('--config', type=click.Path(exists=True), required=True,
              help='Configuration file')
@click.option('--data', type=click.Path(exists=True), required=True,
              help='Training data (parquet/csv)')
@click.option('--output', type=click.Path(), default='models/model.pkl',
              help='Output model path')
@click.option('--model-type', type=click.Choice(['catboost', 'xgboost']), 
              default='catboost', help='Model type')
def train(config, data, output, model_type):
    """Train a fraud detection model"""
    from claims_fraud.config.manager import ConfigManager
    from claims_fraud.ml.training import train_model
    
    click.echo(f"🎓 Training {model_type} model...")
    click.echo(f"   Config: {config}")
    click.echo(f"   Data: {data}")
    
    # Load config
    config_mgr = ConfigManager(config)
    cfg = config_mgr.get_config()
    
    # Train
    model = train_model(cfg, data_path=data, model_type=model_type)
    
    # Save
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    
    click.echo(f"✅ Model saved to: {output}")
