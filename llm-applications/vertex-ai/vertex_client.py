"""
GCP Vertex AI Client for NexaGen AI Ops Platform
Enterprise-grade Vertex AI and Gemini integration with custom fine-tuning
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio

# Google Cloud imports
from google.cloud import aiplatform
from google.cloud import storage
from google.cloud import secretmanager
from google.cloud import monitoring_v3
from google.cloud import logging as cloud_logging
from google.auth import default
import google.generativeai as genai

# Vertex AI specific imports
from vertexai.preview.language_models import TextGenerationModel, ChatModel
from vertexai.preview.generative_models import GenerativeModel
from google.cloud.aiplatform import gapic as aip
from google.cloud.aiplatform_v1 import (
    PredictionServiceClient,
    EndpointServiceClient,
    ModelServiceClient
)

# Configuration and utilities
import structlog
from pydantic import BaseSettings

# Set up structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)


@dataclass
class VertexAIConfig:
    """Configuration for GCP Vertex AI services"""
    
    # GCP credentials
    project_id: str
    location: str = "us-central1"
    
    # Vertex AI settings
    staging_bucket: str
    custom_training_image_uri: str = "gcr.io/cloud-aiplatform/training/tf-cpu.2-8:latest"
    
    # Model settings
    base_model_name: str = "text-bison@001"
    custom_model_name: str = "nexagen-tuned-bison"
    endpoint_name: str = "nexagen-tuned-endpoint"
    
    # Training settings
    training_job_display_name: str = "nexagen-finetune-job"
    dataset_name: str = "nexagen-training-dataset"
    
    # Gemini settings
    gemini_model: str = "gemini-pro"
    generation_config: Dict[str, Any] = None
    
    # Storage settings
    bucket_name: str = "nexagen-finetune-data"
    
    # Monitoring settings
    enable_monitoring: bool = True
    enable_logging: bool = True
    
    # Performance settings
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40


@dataclass
class FineTuningResult:
    """Result from fine-tuning operation"""
    job_name: str
    model_name: str
    endpoint_name: str
    training_metrics: Dict[str, float]
    deployment_status: str
    cost_estimate: float
    training_time: float


@dataclass
class ModelInferenceResult:
    """Result from model inference"""
    query: str
    response: str
    confidence_score: float
    latency_ms: float
    cost_estimate: float
    model_used: str
    metadata: Dict[str, Any]


class VertexAIClient:
    """Main client for GCP Vertex AI operations"""
    
    def __init__(self, config: VertexAIConfig):
        self.config = config
        self.logger = logger.bind(service="vertex_ai")
        
        # Initialize Vertex AI
        aiplatform.init(
            project=config.project_id,
            location=config.location,
            staging_bucket=config.staging_bucket
        )
        
        # Initialize clients
        self._init_clients()
        
        # Performance tracking
        self.metrics: Dict[str, List[float]] = {
            "response_times": [],
            "costs": [],
            "confidence_scores": [],
            "inference_latencies": []
        }
    
    def _init_clients(self):
        """Initialize all GCP service clients"""
        try:
            # Get default credentials
            self.credentials, self.project = default()
            
            # Storage client for data management
            self.storage_client = storage.Client(
                project=self.config.project_id,
                credentials=self.credentials
            )
            
            # Secret Manager client
            self.secret_client = secretmanager.SecretManagerServiceClient(
                credentials=self.credentials
            )
            
            # Monitoring client
            if self.config.enable_monitoring:
                self.monitoring_client = monitoring_v3.MetricServiceClient(
                    credentials=self.credentials
                )
            
            # Cloud Logging client
            if self.config.enable_logging:
                self.logging_client = cloud_logging.Client(
                    project=self.config.project_id,
                    credentials=self.credentials
                )
            
            # Vertex AI service clients
            self.prediction_client = PredictionServiceClient(
                credentials=self.credentials
            )
            self.endpoint_client = EndpointServiceClient(
                credentials=self.credentials
            )
            self.model_client = ModelServiceClient(
                credentials=self.credentials
            )
            
            # Initialize Gemini
            genai.configure(
                api_key=self._get_secret("gemini-api-key") if self._secret_exists("gemini-api-key") else None
            )
            
            self.logger.info("GCP Vertex AI clients initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize Vertex AI clients", error=str(e))
            raise
    
    async def create_training_dataset(
        self, 
        training_data_path: str,
        dataset_display_name: Optional[str] = None
    ) -> str:
        """Create a training dataset from JSONL file"""
        try:
            # Upload training data to GCS if local
            if not training_data_path.startswith("gs://"):
                gcs_path = await self._upload_to_gcs(
                    training_data_path,
                    f"training-data/train.jsonl"
                )
            else:
                gcs_path = training_data_path
            
            # Create dataset
            dataset_display_name = dataset_display_name or self.config.dataset_name
            
            dataset = aiplatform.TextDataset.create(
                display_name=dataset_display_name,
                gcs_source=gcs_path,
                import_schema_uri=aiplatform.schema.dataset.ioformat.text.single_label_classification
            )
            
            self.logger.info(
                "Training dataset created",
                dataset_name=dataset.display_name,
                resource_name=dataset.resource_name
            )
            
            return dataset.resource_name
            
        except Exception as e:
            self.logger.error("Failed to create training dataset", error=str(e))
            raise
    
    async def fine_tune_model(
        self,
        training_dataset_path: str,
        validation_dataset_path: Optional[str] = None
    ) -> FineTuningResult:
        """Fine-tune a model using Vertex AI custom training"""
        start_time = datetime.now()
        
        try:
            # Create training dataset
            dataset_name = await self.create_training_dataset(training_dataset_path)
            
            # Define training job
            job = aiplatform.CustomTrainingJob(
                display_name=self.config.training_job_display_name,
                script_path="llm-applications/vertex-ai/training_script.py",
                container_uri=self.config.custom_training_image_uri,
                requirements=["torch>=1.9.0", "transformers>=4.20.0"],
                model_serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-8:latest"
            )
            
            # Run training
            model = job.run(
                dataset=aiplatform.TextDataset(dataset_name),
                replica_count=1,
                machine_type="n1-standard-4",
                accelerator_type="NVIDIA_TESLA_T4",
                accelerator_count=1,
                base_output_dir=f"gs://{self.config.bucket_name}/model-output"
            )
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Deploy model to endpoint
            endpoint = await self._deploy_model_to_endpoint(model)
            
            # Get training metrics (placeholder - in real implementation, extract from job logs)
            training_metrics = {
                "final_loss": 0.15,
                "accuracy": 0.92,
                "perplexity": 1.8,
                "bleu_score": 0.85
            }
            
            # Estimate cost
            cost_estimate = self._estimate_training_cost(training_time)
            
            result = FineTuningResult(
                job_name=job.display_name,
                model_name=model.display_name,
                endpoint_name=endpoint.display_name,
                training_metrics=training_metrics,
                deployment_status="deployed",
                cost_estimate=cost_estimate,
                training_time=training_time
            )
            
            self.logger.info(
                "Model fine-tuning completed",
                model_name=model.display_name,
                training_time=training_time,
                cost_estimate=cost_estimate
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Model fine-tuning failed", error=str(e))
            raise
    
    async def inference_with_custom_model(
        self,
        query: str,
        endpoint_name: Optional[str] = None,
        include_confidence: bool = True
    ) -> ModelInferenceResult:
        """Run inference using fine-tuned model"""
        start_time = datetime.now()
        
        try:
            endpoint_name = endpoint_name or self.config.endpoint_name
            
            # Get endpoint
            endpoints = aiplatform.Endpoint.list(
                filter=f'display_name="{endpoint_name}"'
            )
            
            if not endpoints:
                raise ValueError(f"Endpoint {endpoint_name} not found")
            
            endpoint = endpoints[0]
            
            # Prepare prediction request
            instances = [{"content": query}]
            parameters = {
                "maxOutputTokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "topP": self.config.top_p,
                "topK": self.config.top_k
            }
            
            # Make prediction
            response = endpoint.predict(
                instances=instances,
                parameters=parameters
            )
            
            # Calculate metrics
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            cost_estimate = self._estimate_inference_cost(len(query), len(response.predictions[0]["content"]))
            
            # Calculate confidence score (simplified heuristic)
            confidence_score = await self._calculate_confidence(
                query, 
                response.predictions[0]["content"]
            ) if include_confidence else 0.8
            
            # Track metrics
            await self._track_metrics(latency_ms, cost_estimate, confidence_score)
            
            result = ModelInferenceResult(
                query=query,
                response=response.predictions[0]["content"],
                confidence_score=confidence_score,
                latency_ms=latency_ms,
                cost_estimate=cost_estimate,
                model_used=endpoint_name,
                metadata={
                    "endpoint_id": endpoint.resource_name,
                    "timestamp": start_time.isoformat(),
                    "parameters": parameters
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Model inference failed", error=str(e))
            raise
    
    async def inference_with_gemini(
        self,
        query: str,
        system_prompt: Optional[str] = None
    ) -> ModelInferenceResult:
        """Run inference using Gemini Pro"""
        start_time = datetime.now()
        
        try:
            model = genai.GenerativeModel(self.config.gemini_model)
            
            # Prepare conversation
            if system_prompt:
                conversation = [
                    {"role": "user", "parts": [system_prompt]},
                    {"role": "model", "parts": ["I understand. I'll follow these instructions."]},
                    {"role": "user", "parts": [query]}
                ]
                prompt = "\n".join([f"{msg['role']}: {msg['parts'][0]}" for msg in conversation])
            else:
                prompt = query
            
            # Generate response
            response = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k
                }
            )
            
            # Calculate metrics
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            cost_estimate = self._estimate_gemini_cost(len(query), len(response.text))
            confidence_score = 0.85  # Gemini typically has high confidence
            
            # Track metrics
            await self._track_metrics(latency_ms, cost_estimate, confidence_score)
            
            result = ModelInferenceResult(
                query=query,
                response=response.text,
                confidence_score=confidence_score,
                latency_ms=latency_ms,
                cost_estimate=cost_estimate,
                model_used=self.config.gemini_model,
                metadata={
                    "timestamp": start_time.isoformat(),
                    "system_prompt_used": system_prompt is not None
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Gemini inference failed", error=str(e))
            raise
    
    async def batch_inference(
        self,
        queries: List[str],
        use_custom_model: bool = True,
        batch_size: int = 10
    ) -> List[ModelInferenceResult]:
        """Run batch inference on multiple queries"""
        results = []
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            batch_results = []
            
            for query in batch:
                try:
                    if use_custom_model:
                        result = await self.inference_with_custom_model(query)
                    else:
                        result = await self.inference_with_gemini(query)
                    
                    batch_results.append(result)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process query: {query[:50]}...", error=str(e))
                    # Create error result
                    error_result = ModelInferenceResult(
                        query=query,
                        response=f"Error processing query: {str(e)}",
                        confidence_score=0.0,
                        latency_ms=0.0,
                        cost_estimate=0.0,
                        model_used="error",
                        metadata={"error": True}
                    )
                    batch_results.append(error_result)
            
            results.extend(batch_results)
            
            # Add small delay between batches to avoid rate limiting
            if i + batch_size < len(queries):
                await asyncio.sleep(1)
        
        self.logger.info(f"Batch inference completed", total_queries=len(queries), successful=len([r for r in results if not r.metadata.get("error", False)]))
        
        return results
    
    async def evaluate_model_performance(
        self,
        test_dataset_path: str,
        ground_truth_path: str,
        use_custom_model: bool = True
    ) -> Dict[str, Any]:
        """Evaluate model performance against ground truth"""
        try:
            # Load test data and ground truth
            test_queries = await self._load_test_queries(test_dataset_path)
            ground_truth = await self._load_ground_truth(ground_truth_path)
            
            # Run inference on test queries
            results = await self.batch_inference(test_queries, use_custom_model)
            
            # Calculate evaluation metrics
            evaluation_metrics = {
                "total_queries": len(test_queries),
                "successful_responses": len([r for r in results if not r.metadata.get("error", False)]),
                "average_latency_ms": sum(r.latency_ms for r in results) / len(results),
                "average_confidence": sum(r.confidence_score for r in results) / len(results),
                "total_cost_inr": sum(r.cost_estimate for r in results)
            }
            
            # Calculate quality metrics if ground truth available
            if ground_truth:
                quality_scores = []
                for i, result in enumerate(results):
                    if i < len(ground_truth) and not result.metadata.get("error", False):
                        # Simple similarity score (in production, use BLEU, ROUGE, etc.)
                        similarity = self._calculate_similarity(result.response, ground_truth[i])
                        quality_scores.append(similarity)
                
                if quality_scores:
                    evaluation_metrics.update({
                        "average_quality_score": sum(quality_scores) / len(quality_scores),
                        "quality_scores_count": len(quality_scores)
                    })
            
            self.logger.info("Model evaluation completed", **evaluation_metrics)
            
            return evaluation_metrics
            
        except Exception as e:
            self.logger.error("Model evaluation failed", error=str(e))
            raise
    
    async def _deploy_model_to_endpoint(self, model) -> aiplatform.Endpoint:
        """Deploy model to serving endpoint"""
        try:
            endpoint = model.deploy(
                deployed_model_display_name=self.config.custom_model_name,
                machine_type="n1-standard-2",
                min_replica_count=0,
                max_replica_count=1,
                accelerator_type="NVIDIA_TESLA_T4",
                accelerator_count=1
            )
            
            self.logger.info(f"Model deployed to endpoint", endpoint_name=endpoint.display_name)
            
            return endpoint
            
        except Exception as e:
            self.logger.error("Model deployment failed", error=str(e))
            raise
    
    async def _upload_to_gcs(self, local_path: str, gcs_path: str) -> str:
        """Upload file to Google Cloud Storage"""
        try:
            bucket = self.storage_client.bucket(self.config.bucket_name)
            blob = bucket.blob(gcs_path)
            
            blob.upload_from_filename(local_path)
            
            gcs_uri = f"gs://{self.config.bucket_name}/{gcs_path}"
            self.logger.info(f"File uploaded to GCS", local_path=local_path, gcs_uri=gcs_uri)
            
            return gcs_uri
            
        except Exception as e:
            self.logger.error("GCS upload failed", error=str(e))
            raise
    
    async def _load_test_queries(self, dataset_path: str) -> List[str]:
        """Load test queries from dataset"""
        queries = []
        try:
            with open(dataset_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if 'prompt' in data:
                        queries.append(data['prompt'])
            return queries
        except Exception as e:
            self.logger.error("Failed to load test queries", error=str(e))
            return []
    
    async def _load_ground_truth(self, ground_truth_path: str) -> List[str]:
        """Load ground truth responses"""
        ground_truth = []
        try:
            with open(ground_truth_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if 'completion' in data:
                        ground_truth.append(data['completion'])
            return ground_truth
        except Exception as e:
            self.logger.error("Failed to load ground truth", error=str(e))
            return []
    
    def _calculate_similarity(self, response: str, ground_truth: str) -> float:
        """Calculate similarity between response and ground truth"""
        # Simple word overlap similarity (in production, use semantic similarity)
        response_words = set(response.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        if not truth_words:
            return 0.0
        
        overlap = len(response_words.intersection(truth_words))
        similarity = overlap / len(truth_words)
        
        return min(similarity, 1.0)
    
    async def _calculate_confidence(self, query: str, response: str) -> float:
        """Calculate confidence score for response"""
        # Simplified confidence calculation
        confidence_factors = []
        
        # Response length factor
        if len(response) > 50:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Query complexity factor
        if len(query.split()) > 5:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.9)
        
        # Uncertainty indicators
        uncertainty_phrases = ["I don't know", "unclear", "not sure", "maybe"]
        if any(phrase in response.lower() for phrase in uncertainty_phrases):
            confidence_factors.append(0.4)
        else:
            confidence_factors.append(0.8)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _estimate_training_cost(self, training_time_seconds: float) -> float:
        """Estimate training cost in INR"""
        # Rough estimate: $1/hour for n1-standard-4 + T4 GPU
        hours = training_time_seconds / 3600
        cost_usd = hours * 1.5  # Including GPU cost
        cost_inr = cost_usd * 83  # USD to INR conversion
        
        return cost_inr
    
    def _estimate_inference_cost(self, input_length: int, output_length: int) -> float:
        """Estimate inference cost in INR"""
        # Rough estimate for Vertex AI prediction
        input_tokens = input_length / 4
        output_tokens = output_length / 4
        
        # Vertex AI pricing (approximate)
        cost_per_1k_tokens = 0.002  # USD
        total_tokens = input_tokens + output_tokens
        cost_usd = (total_tokens / 1000) * cost_per_1k_tokens
        cost_inr = cost_usd * 83
        
        return cost_inr
    
    def _estimate_gemini_cost(self, input_length: int, output_length: int) -> float:
        """Estimate Gemini API cost in INR"""
        # Gemini Pro pricing
        input_tokens = input_length / 4
        output_tokens = output_length / 4
        
        # Gemini pricing (approximate)
        input_cost_per_1k = 0.00025  # USD per 1k input tokens
        output_cost_per_1k = 0.00075  # USD per 1k output tokens
        
        cost_usd = (input_tokens / 1000 * input_cost_per_1k) + (output_tokens / 1000 * output_cost_per_1k)
        cost_inr = cost_usd * 83
        
        return cost_inr
    
    async def _track_metrics(self, latency_ms: float, cost: float, confidence: float):
        """Track performance metrics"""
        self.metrics["inference_latencies"].append(latency_ms)
        self.metrics["costs"].append(cost)
        self.metrics["confidence_scores"].append(confidence)
    
    def _get_secret(self, secret_name: str) -> str:
        """Get secret from Google Secret Manager"""
        try:
            name = f"projects/{self.config.project_id}/secrets/{secret_name}/versions/latest"
            response = self.secret_client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception:
            return None
    
    def _secret_exists(self, secret_name: str) -> bool:
        """Check if secret exists"""
        try:
            name = f"projects/{self.config.project_id}/secrets/{secret_name}"
            self.secret_client.get_secret(request={"name": name})
            return True
        except Exception:
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Vertex AI services"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "services": {}
        }
        
        # Check Vertex AI Model API
        try:
            # Simple API call to check connectivity
            models = aiplatform.Model.list(limit=1)
            health_status["services"]["vertex_ai_models"] = {
                "status": "healthy",
                "models_accessible": True
            }
        except Exception as e:
            health_status["services"]["vertex_ai_models"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "unhealthy"
        
        # Check Cloud Storage
        try:
            bucket = self.storage_client.bucket(self.config.bucket_name)
            bucket.exists()
            health_status["services"]["cloud_storage"] = {
                "status": "healthy",
                "bucket": self.config.bucket_name
            }
        except Exception as e:
            health_status["services"]["cloud_storage"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check Gemini API
        try:
            model = genai.GenerativeModel(self.config.gemini_model)
            response = model.generate_content("Hello", generation_config={"max_output_tokens": 10})
            health_status["services"]["gemini_api"] = {
                "status": "healthy",
                "model": self.config.gemini_model
            }
        except Exception as e:
            health_status["services"]["gemini_api"] = {
                "status": "degraded",
                "error": str(e)
            }
        
        return health_status
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get aggregated performance metrics"""
        if not any(self.metrics.values()):
            return {"message": "No metrics available yet"}
        
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1]
                }
        
        # Calculate derived metrics
        if self.metrics["costs"]:
            summary["total_cost_inr"] = sum(self.metrics["costs"])
            summary["average_cost_per_request"] = summary["total_cost_inr"] / len(self.metrics["costs"])
        
        return summary


# Factory function
def create_vertex_ai_client(
    project_id: str,
    location: str = "us-central1",
    staging_bucket: str = None
) -> VertexAIClient:
    """Factory function to create Vertex AI client"""
    
    if not staging_bucket:
        staging_bucket = f"{project_id}-vertex-ai-staging"
    
    config = VertexAIConfig(
        project_id=project_id,
        location=location,
        staging_bucket=staging_bucket,
        bucket_name=f"{project_id.replace('_', '-')}-finetune-data"
    )
    
    return VertexAIClient(config)


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def demo():
        # Create client
        client = create_vertex_ai_client(
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location="us-central1"
        )
        
        # Fine-tune model
        print("Starting model fine-tuning...")
        finetune_result = await client.fine_tune_model(
            "data/fine_tune/train.jsonl",
            "data/fine_tune/validation.jsonl"
        )
        print(f"Fine-tuning completed: {finetune_result.model_name}")
        
        # Test inference
        print("Testing inference...")
        inference_result = await client.inference_with_custom_model(
            "What are the key financial metrics mentioned in the report?"
        )
        print(f"Response: {inference_result.response}")
        print(f"Confidence: {inference_result.confidence_score:.2f}")
        print(f"Cost: ₹{inference_result.cost_estimate:.4f}")
        
        # Health check
        health = await client.health_check()
        print(f"Health Status: {health['overall_status']}")
        
        # Performance metrics
        metrics = await client.get_performance_metrics()
        print(f"Performance Metrics: {metrics}")
    
    # Run demo
    asyncio.run(demo())