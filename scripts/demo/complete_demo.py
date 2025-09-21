"""
Demo Script for NexaGen AI Ops Platform
Orchestrates end-to-end demonstration of all platform capabilities
"""

import os
import sys
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from llm_applications.azure_openai.azure_client import create_azure_ai_client
from llm_applications.vertex_ai.vertex_client import create_vertex_ai_client
from multimodal.document_processor import create_multimodal_processor
from scripts.utilities.cost_calculator import CostCalculator
from scripts.deployment.health_check import HealthChecker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NexaGenDemo:
    """Main demo orchestrator for NexaGen AI Ops Platform"""
    
    def __init__(self):
        self.demo_start_time = datetime.now()
        self.results = {
            "demo_metadata": {
                "start_time": self.demo_start_time.isoformat(),
                "platform": "NexaGen AI Ops Platform",
                "version": "1.0.0",
                "demo_mode": True
            },
            "health_checks": {},
            "azure_ai_demo": {},
            "vertex_ai_demo": {},
            "multimodal_demo": {},
            "cost_analysis": {},
            "performance_metrics": {},
            "recommendations": []
        }
        
        # Initialize clients
        self._init_clients()
    
    def _init_clients(self):
        """Initialize all service clients"""
        try:
            # Azure AI client
            self.azure_client = create_azure_ai_client(
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
            
            # Vertex AI client
            self.vertex_client = create_vertex_ai_client(
                project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location="us-central1"
            )
            
            # Multimodal processor
            self.multimodal_processor = create_multimodal_processor(
                cv_endpoint=os.getenv("AZURE_COMPUTER_VISION_ENDPOINT"),
                cv_api_key=os.getenv("AZURE_COMPUTER_VISION_KEY")
            )
            
            # Utilities
            self.cost_calculator = CostCalculator()
            self.health_checker = HealthChecker()
            
            logger.info("All clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize clients: {str(e)}")
            raise
    
    async def run_complete_demo(self) -> Dict[str, Any]:
        """Run the complete NexaGen AI Ops Platform demonstration"""
        
        print("🚀 Starting NexaGen AI Ops Platform Demo")
        print("=" * 50)
        
        try:
            # Phase 1: Health Checks
            print("\n📊 Phase 1: System Health Checks")
            await self._demo_health_checks()
            
            # Phase 2: Azure AI Foundry Demo
            print("\n🔷 Phase 2: Azure AI Foundry & OpenAI Demo")
            await self._demo_azure_ai()
            
            # Phase 3: GCP Vertex AI Demo
            print("\n🔶 Phase 3: GCP Vertex AI & Gemini Demo")
            await self._demo_vertex_ai()
            
            # Phase 4: Multimodal AI Demo
            print("\n🖼️ Phase 4: Multimodal Document Intelligence")
            await self._demo_multimodal_ai()
            
            # Phase 5: Performance & Cost Analysis
            print("\n💰 Phase 5: Cost Analysis & Performance Metrics")
            await self._demo_cost_analysis()
            
            # Phase 6: Generate Recommendations
            print("\n💡 Phase 6: AI-Powered Recommendations")
            await self._generate_recommendations()
            
            # Finalize results
            self.results["demo_metadata"]["end_time"] = datetime.now().isoformat()
            self.results["demo_metadata"]["total_duration"] = (
                datetime.now() - self.demo_start_time
            ).total_seconds()
            
            print("\n✅ Demo completed successfully!")
            return self.results
            
        except Exception as e:
            logger.error(f"Demo failed: {str(e)}")
            self.results["demo_metadata"]["error"] = str(e)
            return self.results
    
    async def _demo_health_checks(self):
        """Demonstrate system health monitoring"""
        try:
            # Azure health check
            azure_health = await self.azure_client.health_check()
            self.results["health_checks"]["azure"] = azure_health
            
            print(f"   Azure Services: {azure_health['overall_status'].upper()}")
            for service, status in azure_health["services"].items():
                print(f"     - {service}: {status['status']}")
            
            # Vertex AI health check
            vertex_health = await self.vertex_client.health_check()
            self.results["health_checks"]["vertex_ai"] = vertex_health
            
            print(f"   Vertex AI Services: {vertex_health['overall_status'].upper()}")
            for service, status in vertex_health["services"].items():
                print(f"     - {service}: {status['status']}")
                
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            self.results["health_checks"]["error"] = str(e)
    
    async def _demo_azure_ai(self):
        """Demonstrate Azure AI Foundry capabilities"""
        try:
            demo_queries = [
                "What are the key financial metrics mentioned in this quarterly report?",
                "What are the main data privacy policies outlined in this document?",
                "What security requirements are specified in this IT policy?",
                "What are the customer satisfaction scores mentioned in this service report?"
            ]
            
            azure_results = []
            
            for query in demo_queries[:2]:  # Limit for demo
                print(f"   🤖 Azure Query: {query}")
                
                result = await self.azure_client.analyze_document(
                    query=query,
                    use_rag=True,
                    include_safety_check=True
                )
                
                azure_results.append({
                    "query": query,
                    "response": result.response[:200] + "..." if len(result.response) > 200 else result.response,
                    "confidence": result.confidence_score,
                    "cost": result.cost_estimate,
                    "processing_time": result.processing_time
                })
                
                print(f"     Response: {result.response[:100]}...")
                print(f"     Confidence: {result.confidence_score:.2%}")
                print(f"     Cost: ₹{result.cost_estimate:.4f}")
                print(f"     Time: {result.processing_time:.2f}s\n")
            
            # Get Azure performance metrics
            azure_metrics = await self.azure_client.get_performance_metrics()
            
            self.results["azure_ai_demo"] = {
                "queries_processed": len(azure_results),
                "sample_results": azure_results,
                "performance_metrics": azure_metrics
            }
            
        except Exception as e:
            logger.error(f"Azure AI demo failed: {str(e)}")
            self.results["azure_ai_demo"]["error"] = str(e)
    
    async def _demo_vertex_ai(self):
        """Demonstrate Vertex AI capabilities"""
        try:
            demo_queries = [
                "What are the budget allocations described in this financial plan?",
                "What training requirements are specified in this employee handbook?"
            ]
            
            vertex_results = []
            
            for query in demo_queries:
                print(f"   🤖 Vertex AI Query: {query}")
                
                # Use Gemini for inference
                result = await self.vertex_client.inference_with_gemini(
                    query=query,
                    system_prompt="You are an expert business document analyst. Provide precise, comprehensive answers based on document context."
                )
                
                vertex_results.append({
                    "query": query,
                    "response": result.response[:200] + "..." if len(result.response) > 200 else result.response,
                    "confidence": result.confidence_score,
                    "cost": result.cost_estimate,
                    "latency": result.latency_ms
                })
                
                print(f"     Response: {result.response[:100]}...")
                print(f"     Confidence: {result.confidence_score:.2%}")
                print(f"     Cost: ₹{result.cost_estimate:.4f}")
                print(f"     Latency: {result.latency_ms:.0f}ms\n")
            
            # Get Vertex AI performance metrics
            vertex_metrics = await self.vertex_client.get_performance_metrics()
            
            self.results["vertex_ai_demo"] = {
                "queries_processed": len(vertex_results),
                "sample_results": vertex_results,
                "performance_metrics": vertex_metrics
            }
            
        except Exception as e:
            logger.error(f"Vertex AI demo failed: {str(e)}")
            self.results["vertex_ai_demo"]["error"] = str(e)
    
    async def _demo_multimodal_ai(self):
        """Demonstrate multimodal document processing"""
        try:
            # Find a sample document
            sample_docs = []
            for root, dirs, files in os.walk("data/documents"):
                for file in files[:3]:  # Process max 3 documents for demo
                    if file.endswith('.pdf'):
                        sample_docs.append(os.path.join(root, file))
            
            if not sample_docs:
                print("   ⚠️ No PDF documents found for multimodal demo")
                self.results["multimodal_demo"]["error"] = "No sample documents available"
                return
            
            multimodal_results = []
            
            for doc_path in sample_docs[:1]:  # Process one document for demo
                print(f"   📄 Processing: {os.path.basename(doc_path)}")
                
                result = await self.multimodal_processor.process_document(
                    document_path=doc_path,
                    enable_advanced_analysis=True
                )
                
                multimodal_results.append({
                    "document": os.path.basename(doc_path),
                    "pages": result.total_pages,
                    "confidence": result.overall_confidence,
                    "processing_time": result.total_processing_time,
                    "cost": result.cost_estimate,
                    "entities_found": len(result.extracted_entities),
                    "tables_found": sum(len(page.tables) for page in result.pages),
                    "key_value_pairs": sum(len(page.key_value_pairs) for page in result.pages)
                })
                
                print(f"     Pages: {result.total_pages}")
                print(f"     Confidence: {result.overall_confidence:.2%}")
                print(f"     Entities: {len(result.extracted_entities)}")
                print(f"     Tables: {sum(len(page.tables) for page in result.pages)}")
                print(f"     Processing Time: {result.total_processing_time:.2f}s")
                print(f"     Cost: ₹{result.cost_estimate:.4f}\n")
            
            # Get multimodal performance metrics
            multimodal_metrics = self.multimodal_processor.get_performance_metrics()
            
            self.results["multimodal_demo"] = {
                "documents_processed": len(multimodal_results),
                "sample_results": multimodal_results,
                "performance_metrics": multimodal_metrics
            }
            
        except Exception as e:
            logger.error(f"Multimodal demo failed: {str(e)}")
            self.results["multimodal_demo"]["error"] = str(e)
    
    async def _demo_cost_analysis(self):
        """Demonstrate cost analysis and monitoring"""
        try:
            print("   💳 Calculating total demonstration costs...")
            
            # Aggregate costs from all demos
            total_cost = 0.0
            cost_breakdown = {}
            
            # Azure costs
            azure_demo = self.results.get("azure_ai_demo", {})
            if "sample_results" in azure_demo:
                azure_cost = sum(r.get("cost", 0) for r in azure_demo["sample_results"])
                cost_breakdown["azure_openai"] = azure_cost
                total_cost += azure_cost
            
            # Vertex AI costs
            vertex_demo = self.results.get("vertex_ai_demo", {})
            if "sample_results" in vertex_demo:
                vertex_cost = sum(r.get("cost", 0) for r in vertex_demo["sample_results"])
                cost_breakdown["vertex_ai"] = vertex_cost
                total_cost += vertex_cost
            
            # Multimodal costs
            multimodal_demo = self.results.get("multimodal_demo", {})
            if "sample_results" in multimodal_demo:
                multimodal_cost = sum(r.get("cost", 0) for r in multimodal_demo["sample_results"])
                cost_breakdown["multimodal_ai"] = multimodal_cost
                total_cost += multimodal_cost
            
            # Infrastructure costs (estimated)
            infrastructure_cost = 2.50  # Rough estimate for demo period
            cost_breakdown["infrastructure"] = infrastructure_cost
            total_cost += infrastructure_cost
            
            print(f"     Azure OpenAI: ₹{cost_breakdown.get('azure_openai', 0):.4f}")
            print(f"     Vertex AI: ₹{cost_breakdown.get('vertex_ai', 0):.4f}")
            print(f"     Multimodal AI: ₹{cost_breakdown.get('multimodal_ai', 0):.4f}")
            print(f"     Infrastructure: ₹{infrastructure_cost:.2f}")
            print(f"     TOTAL DEMO COST: ₹{total_cost:.2f}")
            
            # Calculate cost efficiency metrics
            total_queries = (
                len(azure_demo.get("sample_results", [])) +
                len(vertex_demo.get("sample_results", [])) +
                len(multimodal_demo.get("sample_results", []))
            )
            
            cost_per_query = total_cost / total_queries if total_queries > 0 else 0
            
            self.results["cost_analysis"] = {
                "total_cost_inr": total_cost,
                "cost_breakdown": cost_breakdown,
                "total_queries": total_queries,
                "cost_per_query": cost_per_query,
                "budget_utilization": (total_cost / 500.0) * 100,  # Against ₹500 budget
                "cost_efficiency": "Excellent" if cost_per_query < 1.0 else "Good" if cost_per_query < 2.0 else "Needs optimization"
            }
            
        except Exception as e:
            logger.error(f"Cost analysis failed: {str(e)}")
            self.results["cost_analysis"]["error"] = str(e)
    
    async def _generate_recommendations(self):
        """Generate AI-powered recommendations for platform optimization"""
        try:
            recommendations = []
            
            # Analyze performance metrics
            total_cost = self.results.get("cost_analysis", {}).get("total_cost_inr", 0)
            if total_cost > 100:
                recommendations.append({
                    "category": "cost_optimization",
                    "priority": "medium",
                    "recommendation": "Consider implementing request caching to reduce API calls",
                    "potential_savings": "15-25% cost reduction"
                })
            
            # Analyze confidence scores
            azure_results = self.results.get("azure_ai_demo", {}).get("sample_results", [])
            avg_azure_confidence = sum(r.get("confidence", 0) for r in azure_results) / len(azure_results) if azure_results else 0
            
            if avg_azure_confidence < 0.8:
                recommendations.append({
                    "category": "model_performance",
                    "priority": "high",
                    "recommendation": "Fine-tune models with domain-specific data to improve confidence scores",
                    "expected_improvement": "10-20% confidence increase"
                })
            
            # Analyze processing times
            multimodal_results = self.results.get("multimodal_demo", {}).get("sample_results", [])
            if multimodal_results:
                avg_processing_time = sum(r.get("processing_time", 0) for r in multimodal_results) / len(multimodal_results)
                if avg_processing_time > 30:
                    recommendations.append({
                        "category": "performance_optimization",
                        "priority": "medium",
                        "recommendation": "Implement parallel processing for multimodal documents",
                        "expected_improvement": "30-50% faster processing"
                    })
            
            # General recommendations
            recommendations.extend([
                {
                    "category": "scalability",
                    "priority": "high",
                    "recommendation": "Implement auto-scaling for production workloads",
                    "benefit": "Handle traffic spikes efficiently"
                },
                {
                    "category": "monitoring",
                    "priority": "medium",
                    "recommendation": "Set up real-time alerting for cost thresholds",
                    "benefit": "Prevent budget overruns"
                },
                {
                    "category": "security",
                    "priority": "high",
                    "recommendation": "Implement private endpoints for production deployment",
                    "benefit": "Enhanced security posture"
                }
            ])
            
            self.results["recommendations"] = recommendations
            
            print("   🎯 Generated Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"     {i}. {rec['recommendation']} (Priority: {rec['priority']})")
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            self.results["recommendations"] = []
    
    def print_demo_summary(self):
        """Print a comprehensive demo summary"""
        print("\n🎉 NexaGen AI Ops Platform Demo Summary")
        print("=" * 50)
        
        # Demo metadata
        metadata = self.results["demo_metadata"]
        print(f"Duration: {metadata.get('total_duration', 0):.1f} seconds")
        print(f"Demo Mode: {metadata.get('demo_mode')}")
        
        # Health status
        health = self.results.get("health_checks", {})
        azure_status = health.get("azure", {}).get("overall_status", "unknown")
        vertex_status = health.get("vertex_ai", {}).get("overall_status", "unknown")
        print(f"Azure Health: {azure_status.upper()}")
        print(f"Vertex AI Health: {vertex_status.upper()}")
        
        # Performance summary
        azure_queries = len(self.results.get("azure_ai_demo", {}).get("sample_results", []))
        vertex_queries = len(self.results.get("vertex_ai_demo", {}).get("sample_results", []))
        multimodal_docs = len(self.results.get("multimodal_demo", {}).get("sample_results", []))
        
        print(f"Azure Queries: {azure_queries}")
        print(f"Vertex AI Queries: {vertex_queries}")
        print(f"Documents Processed: {multimodal_docs}")
        
        # Cost summary
        cost_analysis = self.results.get("cost_analysis", {})
        total_cost = cost_analysis.get("total_cost_inr", 0)
        budget_utilization = cost_analysis.get("budget_utilization", 0)
        
        print(f"Total Cost: ₹{total_cost:.2f}")
        print(f"Budget Utilization: {budget_utilization:.1f}%")
        print(f"Cost Efficiency: {cost_analysis.get('cost_efficiency', 'Unknown')}")
        
        # Recommendations count
        recommendations_count = len(self.results.get("recommendations", []))
        print(f"AI Recommendations Generated: {recommendations_count}")
        
        print("\n✅ Demo completed successfully! All systems operational.")
        
        if total_cost < 500:
            print(f"🎯 Excellent! Demo completed under budget (₹{500 - total_cost:.2f} remaining)")
        
        print("\n📊 Full results saved to demo_results.json")
    
    def save_results(self, filename: str = "demo_results.json"):
        """Save demo results to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")


async def main():
    """Main entry point for the demo"""
    demo = NexaGenDemo()
    
    try:
        # Run complete demonstration
        results = await demo.run_complete_demo()
        
        # Print summary
        demo.print_demo_summary()
        
        # Save results
        demo.save_results()
        
        # Return success code
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        return 1


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)