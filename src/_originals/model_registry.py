"""
Model Registry Module
Manages model versioning and deployment with MLflow.
"""

import mlflow
import mlflow.pytorch
import torch
from pathlib import Path
from typing import Optional, Dict, List
import logging
import json


logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manages model lifecycle with MLflow Model Registry.
    
    Features:
    - Model registration
    - Version management
    - Stage transitions (Staging, Production, Archived)
    - Model loading and deployment
    """
    
    def __init__(self, tracking_uri: str = "mlruns"):
        """
        Initialize model registry.
        
        Args:
            tracking_uri: MLflow tracking URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        description: Optional[str] = None
    ) -> str:
        """
        Register a model from an MLflow run.
        
        Args:
            run_id: MLflow run ID
            model_name: Name to register model under
            description: Model description
        
        Returns:
            Model version
        """
        model_uri = f"runs:/{run_id}/model"
        
        try:
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            version = model_version.version
            
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=version,
                    description=description
                )
            
            logger.info(f"Registered model '{model_name}' version {version}")
            return version
        
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str
    ):
        """
        Transition model to a different stage.
        
        Args:
            model_name: Model name
            version: Model version
            stage: Target stage ('Staging', 'Production', 'Archived')
        """
        valid_stages = ['None', 'Staging', 'Production', 'Archived']
        if stage not in valid_stages:
            raise ValueError(f"Stage must be one of {valid_stages}")
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            logger.info(f"Transitioned model '{model_name}' v{version} to {stage}")
        
        except Exception as e:
            logger.error(f"Failed to transition model: {e}")
            raise
    
    def get_model_version(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Dict:
        """
        Get model version details.
        
        Args:
            model_name: Model name
            version: Specific version (optional)
            stage: Stage to get latest from (optional)
        
        Returns:
            Dictionary with model version info
        """
        if version:
            model_version = self.client.get_model_version(model_name, version)
        elif stage:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                raise ValueError(f"No model found in stage '{stage}'")
            model_version = versions[0]
        else:
            raise ValueError("Either version or stage must be specified")
        
        return {
            'name': model_version.name,
            'version': model_version.version,
            'stage': model_version.current_stage,
            'description': model_version.description,
            'run_id': model_version.run_id,
            'status': model_version.status,
        }
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = 'Production',
        device: str = 'cpu'
    ) -> torch.nn.Module:
        """
        Load model from registry.
        
        Args:
            model_name: Model name
            version: Specific version (optional)
            stage: Stage to load from (optional)
            device: Device to load model on
        
        Returns:
            Loaded PyTorch model
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            raise ValueError("Either version or stage must be specified")
        
        try:
            model = mlflow.pytorch.load_model(
                model_uri=model_uri,
                map_location=torch.device(device)
            )
            
            logger.info(f"Loaded model '{model_name}' from {model_uri}")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def list_models(self) -> List[Dict]:
        """
        List all registered models.
        
        Returns:
            List of model dictionaries
        """
        registered_models = self.client.search_registered_models()
        
        models = []
        for rm in registered_models:
            models.append({
                'name': rm.name,
                'creation_timestamp': rm.creation_timestamp,
                'last_updated_timestamp': rm.last_updated_timestamp,
                'description': rm.description,
            })
        
        return models
    
    def list_model_versions(self, model_name: str) -> List[Dict]:
        """
        List all versions of a model.
        
        Args:
            model_name: Model name
        
        Returns:
            List of version dictionaries
        """
        versions = self.client.search_model_versions(f"name='{model_name}'")
        
        version_list = []
        for mv in versions:
            version_list.append({
                'version': mv.version,
                'stage': mv.current_stage,
                'status': mv.status,
                'run_id': mv.run_id,
                'description': mv.description,
            })
        
        return sorted(version_list, key=lambda x: int(x['version']), reverse=True)
    
    def delete_model_version(self, model_name: str, version: str):
        """
        Delete a specific model version.
        
        Args:
            model_name: Model name
            version: Version to delete
        """
        try:
            self.client.delete_model_version(model_name, version)
            logger.info(f"Deleted model '{model_name}' version {version}")
        
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            raise
    
    def delete_model(self, model_name: str):
        """
        Delete entire model and all versions.
        
        Args:
            model_name: Model name
        """
        try:
            self.client.delete_registered_model(model_name)
            logger.info(f"Deleted model '{model_name}'")
        
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            raise
    
    def add_model_tag(
        self,
        model_name: str,
        version: str,
        key: str,
        value: str
    ):
        """
        Add tag to model version.
        
        Args:
            model_name: Model name
            version: Model version
            key: Tag key
            value: Tag value
        """
        self.client.set_model_version_tag(model_name, version, key, value)
        logger.info(f"Added tag {key}={value} to model '{model_name}' v{version}")
    
    def get_production_model(self, model_name: str) -> torch.nn.Module:
        """
        Get the current production model.
        
        Args:
            model_name: Model name
        
        Returns:
            Production model
        """
        return self.load_model(model_name, stage='Production')
    
    def promote_to_production(
        self,
        model_name: str,
        version: str,
        archive_existing: bool = True
    ):
        """
        Promote a model version to production.
        
        Args:
            model_name: Model name
            version: Version to promote
            archive_existing: Whether to archive existing production model
        """
        # Archive existing production models if requested
        if archive_existing:
            prod_versions = self.client.get_latest_versions(
                model_name, stages=['Production']
            )
            for pv in prod_versions:
                self.transition_model_stage(
                    model_name, pv.version, 'Archived'
                )
        
        # Promote new version to production
        self.transition_model_stage(model_name, version, 'Production')
        
        logger.info(f"Promoted model '{model_name}' v{version} to Production")


def get_best_run_from_experiment(
    experiment_name: str,
    metric: str = 'val_loss',
    ascending: bool = True
) -> str:
    """
    Get the best run from an experiment based on a metric.
    
    Args:
        experiment_name: MLflow experiment name
        metric: Metric to optimize
        ascending: Whether lower is better
    
    Returns:
        Run ID of best run
    """
    client = mlflow.tracking.MlflowClient()
    
    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    # Search runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1
    )
    
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")
    
    best_run = runs[0]
    logger.info(f"Best run: {best_run.info.run_id} with {metric}={best_run.data.metrics.get(metric)}")
    
    return best_run.info.run_id


if __name__ == "__main__":
    # Example usage
    registry = ModelRegistry()
    
    # List models
    models = registry.list_models()
    print(f"Registered models: {models}")
    
    # Register a new model (example)
    # run_id = "some_run_id"
    # version = registry.register_model(run_id, "claims_autoencoder", "Production model")
    
    # Load production model
    # model = registry.get_production_model("claims_autoencoder")
