"""Evaluation CLI command"""
import click

@click.command()
@click.option('--model', type=click.Path(exists=True), required=True,
              help='Trained model file')
@click.option('--test-data', type=click.Path(exists=True), required=True,
              help='Test data file')
@click.option('--output', type=click.Path(), default='evaluation_report.txt',
              help='Output report file')
def evaluate(model, test_data, output):
    """Evaluate model performance"""
    from claims_fraud.core.scoring import FraudDetector
    from claims_fraud.analysis.evaluation import evaluate_model
    import pandas as pd
    
    click.echo(f"📊 Evaluating model...")
    
    # Load
    detector = FraudDetector.load(model)
    data = pd.read_parquet(test_data)
    
    # Evaluate
    results = evaluate_model(detector, data)
    
    # Save report
    with open(output, 'w') as f:
        f.write(results)
    
    click.echo(f"✅ Report saved to: {output}")
