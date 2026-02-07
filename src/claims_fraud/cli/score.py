"""Scoring CLI command"""
import click
import pandas as pd

@click.command()
@click.option('--model', type=click.Path(exists=True), required=True,
              help='Trained model file')
@click.option('--input', 'input_file', type=click.Path(exists=True), required=True,
              help='Input data file')
@click.option('--output', type=click.Path(), required=True,
              help='Output scores file')
@click.option('--threshold', type=float, default=0.95,
              help='Fraud threshold percentile')
def score(model, input_file, output, threshold):
    """Score claims for fraud"""
    from claims_fraud.core.scoring import FraudDetector
    
    click.echo(f"🎯 Scoring claims...")
    click.echo(f"   Model: {model}")
    click.echo(f"   Input: {input_file}")
    
    # Load
    detector = FraudDetector.load(model)
    data = pd.read_parquet(input_file)
    
    # Score
    scores = detector.predict(data)
    
    # Save
    results = pd.DataFrame({
        'fraud_score': scores,
        'is_fraud': scores > scores.quantile(threshold)
    })
    results.to_csv(output, index=False)
    
    click.echo(f"✅ Scores saved to: {output}")
    click.echo(f"   Flagged: {results['is_fraud'].sum()} / {len(results)}")
