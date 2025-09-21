"""
Azure AI Foundry Client for NexaGen AI Ops Platform
Enterprise-grade Azure OpenAI and AI Foundry integration
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio
import json

# Azure imports
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, Environment, Job
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

# Azure OpenAI imports
from openai import AzureOpenAI

# Azure Cognitive Services
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.contentsafety import ContentSafetyClient
from azure.cognitiveservices.contentsafety.models import TextAnalyzeRequest
from msrest.authentication import CognitiveServicesCredentials

# Configuration and utilities
from pydantic import BaseSettings, Field
import structlog

# Set up structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)


@dataclass
class AzureAIConfig:
    """Configuration for Azure AI services"""
    
    # Azure credentials
    subscription_id: str
    tenant_id: str
    resource_group: str
    
    # AI Foundry settings
    ai_foundry_hub: str = "nexagen-ai-hub"
    ml_workspace: str = "nexagen-ml-workspace"
    
    # Azure OpenAI settings
    openai_endpoint: str
    openai_api_key: str
    openai_api_version: str = "2024-02-01"
    gpt_deployment: str = "nexagen-gpt-flow"
    embedding_deployment: str = "nexagen-embed"
    
    # Azure Search settings
    search_endpoint: str
    search_api_key: str
    search_index: str = "nexagen-documents"
    
    # Computer Vision settings
    cv_endpoint: str
    cv_api_key: str
    
    # Content Safety settings
    safety_endpoint: str
    safety_api_key: str
    
    # Storage settings
    storage_account: str
    storage_key: str
    
    # Key Vault settings
    key_vault_url: str
    
    # Monitoring settings
    app_insights_key: str
    enable_telemetry: bool = True
    
    # Model settings
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class DocumentAnalysisResult:
    """Result from document analysis pipeline"""
    query: str
    response: str
    confidence_score: float
    safety_score: float
    retrieved_contexts: List[str]
    processing_time: float
    cost_estimate: float
    metadata: Dict[str, Any]


class AzureAIFoundryClient:
    """Main client for Azure AI Foundry operations"""
    
    def __init__(self, config: AzureAIConfig):
        self.config = config
        self.logger = logger.bind(service="azure_ai_foundry")
        
        # Initialize credentials
        self.credential = DefaultAzureCredential()
        
        # Initialize clients
        self._init_clients()
        
        # Performance tracking
        self.metrics: Dict[str, List[float]] = {
            "response_times": [],
            "costs": [],
            "safety_scores": [],
            "confidence_scores": []
        }
    
    def _init_clients(self):
        """Initialize all Azure service clients"""
        try:
            # ML Client for AI Foundry
            self.ml_client = MLClient(
                credential=self.credential,
                subscription_id=self.config.subscription_id,
                resource_group_name=self.config.resource_group,
                workspace_name=self.config.ml_workspace
            )
            
            # Azure OpenAI Client
            self.openai_client = AzureOpenAI(
                azure_endpoint=self.config.openai_endpoint,
                api_key=self.config.openai_api_key,
                api_version=self.config.openai_api_version
            )
            
            # Azure Search Client
            search_credential = AzureKeyCredential(self.config.search_api_key)
            self.search_client = SearchClient(
                endpoint=self.config.search_endpoint,
                index_name=self.config.search_index,
                credential=search_credential
            )
            
            # Computer Vision Client
            cv_credential = CognitiveServicesCredentials(self.config.cv_api_key)
            self.cv_client = ComputerVisionClient(
                self.config.cv_endpoint,
                cv_credential
            )
            
            # Content Safety Client
            self.safety_client = ContentSafetyClient(
                endpoint=self.config.safety_endpoint,
                credential=AzureKeyCredential(self.config.safety_api_key)
            )
            
            # Storage Client
            self.storage_client = BlobServiceClient(
                account_url=f"https://{self.config.storage_account}.blob.core.windows.net",
                credential=self.config.storage_key
            )
            
            # Key Vault Client
            self.kv_client = SecretClient(
                vault_url=self.config.key_vault_url,
                credential=self.credential
            )
            
            self.logger.info("Azure AI clients initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize Azure AI clients", error=str(e))
            raise
    
    async def analyze_document(
        self, 
        query: str, 
        document_content: Optional[str] = None,
        include_safety_check: bool = True,
        use_rag: bool = True
    ) -> DocumentAnalysisResult:
        """
        Analyze document query using complete Azure AI pipeline
        
        Args:
            query: User's question about the document
            document_content: Optional pre-provided document content
            include_safety_check: Whether to perform content safety checks
            use_rag: Whether to use RAG for context retrieval
            
        Returns:
            DocumentAnalysisResult with comprehensive analysis
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Content Safety Check
            safety_score = 1.0
            if include_safety_check:
                safety_score = await self._check_content_safety(query)
                if safety_score < 0.5:
                    return DocumentAnalysisResult(
                        query=query,
                        response="I cannot process this query due to safety concerns.",
                        confidence_score=0.0,
                        safety_score=safety_score,
                        retrieved_contexts=[],
                        processing_time=0.0,
                        cost_estimate=0.0,
                        metadata={"safety_violation": True}
                    )
            
            # Step 2: Context Retrieval
            retrieved_contexts = []
            if use_rag and not document_content:
                retrieved_contexts = await self._retrieve_context(query)
                document_content = "\n".join(retrieved_contexts)
            
            # Step 3: Generate Response
            response = await self._generate_response(query, document_content or "")
            
            # Step 4: Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            cost_estimate = self._estimate_cost(query, response)
            confidence_score = await self._calculate_confidence(query, response, document_content or "")
            
            # Step 5: Track metrics
            await self._track_metrics(processing_time, cost_estimate, safety_score, confidence_score)
            
            result = DocumentAnalysisResult(
                query=query,
                response=response,
                confidence_score=confidence_score,
                safety_score=safety_score,
                retrieved_contexts=retrieved_contexts,
                processing_time=processing_time,
                cost_estimate=cost_estimate,
                metadata={
                    "model_used": self.config.gpt_deployment,
                    "timestamp": start_time.isoformat(),
                    "use_rag": use_rag,
                    "context_length": len(document_content or "")
                }
            )
            
            self.logger.info(
                "Document analysis completed",
                query_length=len(query),
                response_length=len(response),
                processing_time=processing_time,
                confidence_score=confidence_score
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Document analysis failed", error=str(e))
            raise
    
    async def _check_content_safety(self, text: str) -> float:
        """Check content safety using Azure Content Safety service"""
        try:
            request = TextAnalyzeRequest(text=text)
            response = self.safety_client.analyze_text(request)
            
            # Calculate overall safety score (1.0 = safe, 0.0 = unsafe)
            severity_scores = []
            
            if hasattr(response, 'hate_result') and response.hate_result:
                severity_scores.append(response.hate_result.severity)
            if hasattr(response, 'self_harm_result') and response.self_harm_result:
                severity_scores.append(response.self_harm_result.severity)
            if hasattr(response, 'sexual_result') and response.sexual_result:
                severity_scores.append(response.sexual_result.severity)
            if hasattr(response, 'violence_result') and response.violence_result:
                severity_scores.append(response.violence_result.severity)
            
            if not severity_scores:
                return 1.0
            
            max_severity = max(severity_scores)
            safety_score = max(0.0, 1.0 - (max_severity / 6.0))  # Assuming severity scale 0-6
            
            return safety_score
            
        except Exception as e:
            self.logger.warning("Content safety check failed", error=str(e))
            return 1.0  # Default to safe if check fails
    
    async def _retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant context using Azure AI Search"""
        try:
            # Generate query embedding
            embedding_response = self.openai_client.embeddings.create(
                input=query,
                model=self.config.embedding_deployment
            )
            query_vector = embedding_response.data[0].embedding
            
            # Create vectorized query
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            # Search for relevant documents
            search_results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=["content", "title", "metadata"],
                top=top_k
            )
            
            # Extract context from results
            contexts = []
            for result in search_results:
                if "content" in result:
                    contexts.append(result["content"])
            
            return contexts
            
        except Exception as e:
            self.logger.error("Context retrieval failed", error=str(e))
            return []
    
    async def _generate_response(self, query: str, context: str) -> str:
        """Generate response using Azure OpenAI"""
        try:
            system_prompt = """
            You are an expert document analyst with deep knowledge across business domains.
            Your task is to answer questions based solely on the provided document context.
            
            Guidelines:
            1. Use only information from the provided context
            2. Be precise, comprehensive, and well-structured in your responses
            3. If the context doesn't contain relevant information, clearly state this
            4. Provide specific details and numbers when available
            5. Maintain a professional and helpful tone
            6. Structure responses with numbered points when appropriate
            """
            
            user_prompt = f"""
            Context from documents:
            {context}
            
            User question: {query}
            
            Please provide a detailed, accurate answer based on the document context above.
            """
            
            response = self.openai_client.chat.completions.create(
                model=self.config.gpt_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error("Response generation failed", error=str(e))
            return "I apologize, but I encountered an error while processing your request."
    
    async def _calculate_confidence(self, query: str, response: str, context: str) -> float:
        """Calculate confidence score for the response"""
        try:
            # Simple heuristic-based confidence calculation
            # In production, this could use more sophisticated methods
            
            confidence_factors = []
            
            # Context relevance factor
            context_length = len(context)
            if context_length > 100:
                confidence_factors.append(0.8)
            elif context_length > 50:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.3)
            
            # Response completeness factor
            response_length = len(response)
            if response_length > 200:
                confidence_factors.append(0.9)
            elif response_length > 100:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
            
            # Uncertainty indicators in response
            uncertainty_phrases = ["I don't know", "unclear", "not specified", "not mentioned"]
            if any(phrase in response.lower() for phrase in uncertainty_phrases):
                confidence_factors.append(0.4)
            else:
                confidence_factors.append(0.8)
            
            # Calculate average confidence
            return sum(confidence_factors) / len(confidence_factors)
            
        except Exception as e:
            self.logger.warning("Confidence calculation failed", error=str(e))
            return 0.5
    
    def _estimate_cost(self, query: str, response: str) -> float:
        """Estimate the cost of the API call"""
        try:
            # Rough token estimation (1 token ≈ 4 characters for English)
            input_tokens = len(query) / 4
            output_tokens = len(response) / 4
            
            # GPT-4o-mini pricing (approximate in USD)
            input_cost_per_token = 0.000015 / 1000  # $0.15 per 1M tokens
            output_cost_per_token = 0.0006 / 1000   # $0.60 per 1M tokens
            
            total_cost_usd = (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)
            
            # Convert to INR (approximate rate: 1 USD = 83 INR)
            total_cost_inr = total_cost_usd * 83
            
            return total_cost_inr
            
        except Exception as e:
            self.logger.warning("Cost estimation failed", error=str(e))
            return 0.0
    
    async def _track_metrics(self, processing_time: float, cost: float, safety_score: float, confidence_score: float):
        """Track performance metrics"""
        try:
            self.metrics["response_times"].append(processing_time)
            self.metrics["costs"].append(cost)
            self.metrics["safety_scores"].append(safety_score)
            self.metrics["confidence_scores"].append(confidence_score)
            
            # Send telemetry if enabled
            if self.config.enable_telemetry:
                # This would integrate with Azure Application Insights
                pass
                
        except Exception as e:
            self.logger.warning("Metrics tracking failed", error=str(e))
    
    async def process_document_with_ocr(self, image_path: str) -> str:
        """Process document image using Azure Computer Vision OCR"""
        try:
            with open(image_path, "rb") as image_stream:
                # Use Read API for text extraction
                read_operation = self.cv_client.read_in_stream(
                    image_stream,
                    raw=True
                )
                
                # Get operation ID from response headers
                operation_location = read_operation.headers["Operation-Location"]
                operation_id = operation_location.split("/")[-1]
                
                # Wait for operation to complete
                import time
                while True:
                    result = self.cv_client.get_read_result(operation_id)
                    if result.status not in ["notStarted", "running"]:
                        break
                    time.sleep(1)
                
                # Extract text from results
                extracted_text = ""
                if result.status == "succeeded":
                    for page in result.analyze_result.read_results:
                        for line in page.lines:
                            extracted_text += line.text + "\n"
                
                self.logger.info(
                    "OCR processing completed",
                    text_length=len(extracted_text),
                    pages_processed=len(result.analyze_result.read_results)
                )
                
                return extracted_text
                
        except Exception as e:
            self.logger.error("OCR processing failed", error=str(e))
            raise
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get aggregated performance metrics"""
        try:
            if not any(self.metrics.values()):
                return {"message": "No metrics available yet"}
            
            metrics_summary = {}
            
            for metric_name, values in self.metrics.items():
                if values:
                    metrics_summary[metric_name] = {
                        "count": len(values),
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "latest": values[-1] if values else 0
                    }
            
            # Calculate derived metrics
            if self.metrics["response_times"] and self.metrics["costs"]:
                total_requests = len(self.metrics["response_times"])
                total_cost = sum(self.metrics["costs"])
                avg_cost_per_request = total_cost / total_requests if total_requests > 0 else 0
                
                metrics_summary["summary"] = {
                    "total_requests": total_requests,
                    "total_cost_inr": total_cost,
                    "average_cost_per_request": avg_cost_per_request,
                    "average_response_time": sum(self.metrics["response_times"]) / total_requests
                }
            
            return metrics_summary
            
        except Exception as e:
            self.logger.error("Failed to get performance metrics", error=str(e))
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all Azure services"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "services": {}
        }
        
        services_to_check = [
            ("azure_openai", self._check_openai_health),
            ("azure_search", self._check_search_health),
            ("computer_vision", self._check_cv_health),
            ("content_safety", self._check_safety_health),
            ("storage", self._check_storage_health)
        ]
        
        for service_name, check_function in services_to_check:
            try:
                status = await check_function()
                health_status["services"][service_name] = status
                if status["status"] != "healthy":
                    health_status["overall_status"] = "degraded"
            except Exception as e:
                health_status["services"][service_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["overall_status"] = "unhealthy"
        
        return health_status
    
    async def _check_openai_health(self) -> Dict[str, Any]:
        """Check Azure OpenAI service health"""
        try:
            # Simple completion test
            response = self.openai_client.chat.completions.create(
                model=self.config.gpt_deployment,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            return {
                "status": "healthy",
                "response_time": "< 1s",
                "model": self.config.gpt_deployment
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_search_health(self) -> Dict[str, Any]:
        """Check Azure Search service health"""
        try:
            # Simple search test
            results = self.search_client.search(
                search_text="test",
                top=1
            )
            
            # Just checking if we can connect
            list(results)
            
            return {
                "status": "healthy",
                "index": self.config.search_index
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e)
            }
    
    async def _check_cv_health(self) -> Dict[str, Any]:
        """Check Computer Vision service health"""
        try:
            # Check service availability
            # Note: This is a placeholder as CV doesn't have a simple health endpoint
            return {
                "status": "healthy",
                "endpoint": self.config.cv_endpoint
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_safety_health(self) -> Dict[str, Any]:
        """Check Content Safety service health"""
        try:
            # Simple safety check test
            request = TextAnalyzeRequest(text="Hello world")
            self.safety_client.analyze_text(request)
            
            return {
                "status": "healthy",
                "endpoint": self.config.safety_endpoint
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_storage_health(self) -> Dict[str, Any]:
        """Check Storage service health"""
        try:
            # List containers to check connectivity
            containers = self.storage_client.list_containers()
            list(containers)  # Force evaluation
            
            return {
                "status": "healthy",
                "account": self.config.storage_account
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Factory function to create Azure AI client
def create_azure_ai_client(
    subscription_id: str,
    tenant_id: str,
    resource_group: str,
    openai_endpoint: str,
    openai_api_key: str,
    search_endpoint: str,
    search_api_key: str,
    cv_endpoint: str,
    cv_api_key: str,
    safety_endpoint: str,
    safety_api_key: str,
    storage_account: str,
    storage_key: str,
    key_vault_url: str,
    app_insights_key: str
) -> AzureAIFoundryClient:
    """Factory function to create Azure AI Foundry client"""
    
    config = AzureAIConfig(
        subscription_id=subscription_id,
        tenant_id=tenant_id,
        resource_group=resource_group,
        openai_endpoint=openai_endpoint,
        openai_api_key=openai_api_key,
        search_endpoint=search_endpoint,
        search_api_key=search_api_key,
        cv_endpoint=cv_endpoint,
        cv_api_key=cv_api_key,
        safety_endpoint=safety_endpoint,
        safety_api_key=safety_api_key,
        storage_account=storage_account,
        storage_key=storage_key,
        key_vault_url=key_vault_url,
        app_insights_key=app_insights_key
    )
    
    return AzureAIFoundryClient(config)


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Create client from environment variables
    client = create_azure_ai_client(
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
        tenant_id=os.getenv("AZURE_TENANT_ID"),
        resource_group=os.getenv("AZURE_RESOURCE_GROUP"),
        openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        search_api_key=os.getenv("AZURE_SEARCH_API_KEY"),
        cv_endpoint=os.getenv("AZURE_COMPUTER_VISION_ENDPOINT"),
        cv_api_key=os.getenv("AZURE_COMPUTER_VISION_KEY"),
        safety_endpoint=os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT"),
        safety_api_key=os.getenv("AZURE_CONTENT_SAFETY_KEY"),
        storage_account=os.getenv("AZURE_STORAGE_ACCOUNT"),
        storage_key=os.getenv("AZURE_STORAGE_KEY"),
        key_vault_url=os.getenv("AZURE_KEY_VAULT"),
        app_insights_key=os.getenv("AZURE_APP_INSIGHTS_KEY")
    )
    
    async def demo():
        # Example document analysis
        result = await client.analyze_document(
            query="What are the key performance metrics mentioned?",
            use_rag=True
        )
        
        print(f"Query: {result.query}")
        print(f"Response: {result.response}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Safety Score: {result.safety_score:.2f}")
        print(f"Cost: ₹{result.cost_estimate:.4f}")
        
        # Get performance metrics
        metrics = await client.get_performance_metrics()
        print(f"Performance Metrics: {metrics}")
        
        # Health check
        health = await client.health_check()
        print(f"Health Status: {health}")
    
    # Run demo
    asyncio.run(demo())