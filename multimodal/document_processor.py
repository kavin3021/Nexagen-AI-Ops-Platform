"""
Multimodal AI Processing for NexaGen AI Ops Platform
OCR + Vision + LLM integration for document intelligence
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
import base64
from PIL import Image
import io

# Azure Computer Vision
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

# PDF processing
import fitz  # PyMuPDF for PDF processing
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np

# Document processing
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

# Configuration
import structlog
from pydantic import BaseSettings

logger = structlog.get_logger(__name__)


@dataclass
class MultimodalConfig:
    """Configuration for multimodal AI processing"""
    
    # Azure Computer Vision
    cv_endpoint: str
    cv_api_key: str
    
    # Azure Form Recognizer
    form_recognizer_endpoint: str
    form_recognizer_key: str
    
    # OCR Settings
    ocr_confidence_threshold: float = 0.8
    enable_handwriting: bool = True
    detect_orientation: bool = True
    
    # Image processing settings
    image_quality_threshold: float = 0.7
    max_image_size_mb: int = 50
    supported_formats: List[str] = None
    
    # Document analysis settings
    enable_layout_analysis: bool = True
    enable_table_extraction: bool = True
    enable_key_value_extraction: bool = True
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp']


@dataclass
class DocumentPage:
    """Represents a single page of a document"""
    page_number: int
    text_content: str
    confidence_score: float
    layout_info: Dict[str, Any]
    tables: List[Dict[str, Any]]
    key_value_pairs: Dict[str, str]
    image_path: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class MultimodalAnalysisResult:
    """Complete result from multimodal document analysis"""
    document_path: str
    total_pages: int
    pages: List[DocumentPage]
    overall_confidence: float
    total_processing_time: float
    document_summary: str
    extracted_entities: List[Dict[str, Any]]
    cost_estimate: float
    metadata: Dict[str, Any]


class MultimodalDocumentProcessor:
    """Main class for multimodal document processing"""
    
    def __init__(self, config: MultimodalConfig):
        self.config = config
        self.logger = logger.bind(service="multimodal_processor")
        
        # Initialize Azure clients
        self._init_azure_clients()
        
        # Performance tracking
        self.metrics = {
            "documents_processed": 0,
            "pages_processed": 0,
            "ocr_confidence_scores": [],
            "processing_times": [],
            "costs": []
        }
    
    def _init_azure_clients(self):
        """Initialize Azure Computer Vision and Form Recognizer clients"""
        try:
            # Computer Vision client
            cv_credential = CognitiveServicesCredentials(self.config.cv_api_key)
            self.cv_client = ComputerVisionClient(
                self.config.cv_endpoint,
                cv_credential
            )
            
            # Form Recognizer client
            self.form_client = DocumentAnalysisClient(
                endpoint=self.config.form_recognizer_endpoint,
                credential=AzureKeyCredential(self.config.form_recognizer_key)
            )
            
            self.logger.info("Azure multimodal clients initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize Azure clients", error=str(e))
            raise
    
    async def process_document(
        self,
        document_path: str,
        enable_advanced_analysis: bool = True
    ) -> MultimodalAnalysisResult:
        """
        Process a document using multimodal AI pipeline
        
        Args:
            document_path: Path to document file
            enable_advanced_analysis: Enable layout and entity analysis
            
        Returns:
            Complete multimodal analysis result
        """
        start_time = datetime.now()
        
        try:
            # Validate document
            await self._validate_document(document_path)
            
            # Convert document to images if PDF
            image_paths = await self._convert_to_images(document_path)
            
            # Process each page
            pages = []
            for i, image_path in enumerate(image_paths):
                page_result = await self._process_page(
                    image_path, 
                    i + 1,
                    enable_advanced_analysis
                )
                pages.append(page_result)
            
            # Calculate overall metrics
            total_processing_time = (datetime.now() - start_time).total_seconds()
            overall_confidence = sum(p.confidence_score for p in pages) / len(pages) if pages else 0.0
            cost_estimate = self._estimate_processing_cost(len(pages))
            
            # Generate document summary
            document_summary = await self._generate_document_summary(pages)
            
            # Extract entities
            extracted_entities = await self._extract_entities(pages) if enable_advanced_analysis else []
            
            # Update metrics
            self._update_metrics(len(pages), total_processing_time, cost_estimate, overall_confidence)
            
            result = MultimodalAnalysisResult(
                document_path=document_path,
                total_pages=len(pages),
                pages=pages,
                overall_confidence=overall_confidence,
                total_processing_time=total_processing_time,
                document_summary=document_summary,
                extracted_entities=extracted_entities,
                cost_estimate=cost_estimate,
                metadata={
                    "processing_timestamp": start_time.isoformat(),
                    "advanced_analysis_enabled": enable_advanced_analysis,
                    "image_paths": image_paths
                }
            )
            
            self.logger.info(
                "Document processing completed",
                document=document_path,
                pages=len(pages),
                confidence=overall_confidence,
                processing_time=total_processing_time
            )
            
            # Cleanup temporary image files
            await self._cleanup_temp_files(image_paths, document_path)
            
            return result
            
        except Exception as e:
            self.logger.error("Document processing failed", document=document_path, error=str(e))
            raise
    
    async def _process_page(
        self,
        image_path: str,
        page_number: int,
        enable_advanced_analysis: bool
    ) -> DocumentPage:
        """Process a single page using multiple analysis methods"""
        page_start_time = datetime.now()
        
        try:
            # Method 1: Azure Computer Vision OCR
            ocr_result = await self._ocr_with_computer_vision(image_path)
            
            # Method 2: Azure Form Recognizer (if advanced analysis enabled)
            form_result = None
            if enable_advanced_analysis:
                form_result = await self._analyze_with_form_recognizer(image_path)
            
            # Method 3: Tesseract OCR (fallback/comparison)
            tesseract_result = await self._ocr_with_tesseract(image_path)
            
            # Combine results and choose best text extraction
            text_content, confidence_score = await self._combine_ocr_results(
                ocr_result, form_result, tesseract_result
            )
            
            # Extract layout information
            layout_info = await self._extract_layout_info(form_result) if form_result else {}
            
            # Extract tables
            tables = await self._extract_tables(form_result) if form_result else []
            
            # Extract key-value pairs
            key_value_pairs = await self._extract_key_value_pairs(form_result) if form_result else {}
            
            processing_time = (datetime.now() - page_start_time).total_seconds()
            
            return DocumentPage(
                page_number=page_number,
                text_content=text_content,
                confidence_score=confidence_score,
                layout_info=layout_info,
                tables=tables,
                key_value_pairs=key_value_pairs,
                image_path=image_path,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Page processing failed", page=page_number, error=str(e))
            # Return empty page result on error
            return DocumentPage(
                page_number=page_number,
                text_content=f"Error processing page {page_number}: {str(e)}",
                confidence_score=0.0,
                layout_info={},
                tables=[],
                key_value_pairs={},
                image_path=image_path,
                processing_time=0.0
            )
    
    async def _ocr_with_computer_vision(self, image_path: str) -> Dict[str, Any]:
        """Perform OCR using Azure Computer Vision"""
        try:
            with open(image_path, 'rb') as image_stream:
                # Submit OCR request
                read_operation = self.cv_client.read_in_stream(
                    image_stream,
                    raw=True
                )
                
                # Get operation ID
                operation_location = read_operation.headers["Operation-Location"]
                operation_id = operation_location.split("/")[-1]
                
                # Poll for results
                import time
                while True:
                    result = self.cv_client.get_read_result(operation_id)
                    if result.status not in [OperationStatusCodes.not_started, OperationStatusCodes.running]:
                        break
                    time.sleep(1)
                
                if result.status == OperationStatusCodes.succeeded:
                    text_lines = []
                    confidences = []
                    
                    for page in result.analyze_result.read_results:
                        for line in page.lines:
                            text_lines.append(line.text)
                            if hasattr(line, 'confidence'):
                                confidences.append(line.confidence)
                    
                    return {
                        "text": "\n".join(text_lines),
                        "confidence": sum(confidences) / len(confidences) if confidences else 0.8,
                        "lines": text_lines,
                        "source": "azure_cv"
                    }
                else:
                    return {
                        "text": "",
                        "confidence": 0.0,
                        "lines": [],
                        "source": "azure_cv",
                        "error": f"OCR failed with status: {result.status}"
                    }
                    
        except Exception as e:
            self.logger.error("Azure Computer Vision OCR failed", error=str(e))
            return {
                "text": "",
                "confidence": 0.0,
                "lines": [],
                "source": "azure_cv",
                "error": str(e)
            }
    
    async def _analyze_with_form_recognizer(self, image_path: str) -> Dict[str, Any]:
        """Analyze document using Azure Form Recognizer"""
        try:
            with open(image_path, 'rb') as image_stream:
                poller = self.form_client.begin_analyze_document(
                    "prebuilt-document",
                    image_stream
                )
                result = poller.result()
                
                # Extract text and confidence
                text_content = result.content if result.content else ""
                
                # Calculate average confidence from paragraphs
                confidences = []
                if result.paragraphs:
                    for paragraph in result.paragraphs:
                        if hasattr(paragraph, 'confidence'):
                            confidences.append(paragraph.confidence)
                
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.8
                
                return {
                    "text": text_content,
                    "confidence": avg_confidence,
                    "result": result,
                    "source": "azure_form_recognizer"
                }
                
        except Exception as e:
            self.logger.error("Azure Form Recognizer analysis failed", error=str(e))
            return {
                "text": "",
                "confidence": 0.0,
                "result": None,
                "source": "azure_form_recognizer",
                "error": str(e)
            }
    
    async def _ocr_with_tesseract(self, image_path: str) -> Dict[str, Any]:
        """Perform OCR using Tesseract (fallback method)"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply image preprocessing for better OCR
            # 1. Noise removal
            denoised = cv2.medianBlur(gray, 5)
            
            # 2. Thresholding
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Perform OCR with confidence data
            ocr_data = pytesseract.image_to_data(
                thresh,
                output_type=pytesseract.Output.DICT,
                config='--psm 3'  # Automatic page segmentation
            )
            
            # Extract text and calculate confidence
            words = []
            confidences = []
            
            for i, conf in enumerate(ocr_data['conf']):
                if int(conf) > 0:  # Only include words with confidence > 0
                    words.append(ocr_data['text'][i])
                    confidences.append(int(conf))
            
            text = ' '.join(words)
            avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
            
            return {
                "text": text,
                "confidence": avg_confidence,
                "words": words,
                "source": "tesseract"
            }
            
        except Exception as e:
            self.logger.error("Tesseract OCR failed", error=str(e))
            return {
                "text": "",
                "confidence": 0.0,
                "words": [],
                "source": "tesseract",
                "error": str(e)
            }
    
    async def _combine_ocr_results(
        self,
        cv_result: Dict[str, Any],
        form_result: Optional[Dict[str, Any]],
        tesseract_result: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Combine OCR results from multiple sources and choose the best"""
        
        results = [cv_result, tesseract_result]
        if form_result:
            results.append(form_result)
        
        # Filter out results with errors
        valid_results = [r for r in results if "error" not in r and r["confidence"] > 0]
        
        if not valid_results:
            return "No text could be extracted from this page.", 0.0
        
        # Choose result with highest confidence
        best_result = max(valid_results, key=lambda x: x["confidence"])
        
        # If confidences are close, prefer Azure Form Recognizer, then Computer Vision, then Tesseract
        preference_order = {"azure_form_recognizer": 3, "azure_cv": 2, "tesseract": 1}
        
        # Find results within 10% confidence of the best
        best_confidence = best_result["confidence"]
        close_results = [r for r in valid_results if abs(r["confidence"] - best_confidence) <= 0.1]
        
        if len(close_results) > 1:
            # Choose based on preference
            best_result = max(close_results, key=lambda x: preference_order.get(x["source"], 0))
        
        return best_result["text"], best_result["confidence"]
    
    async def _extract_layout_info(self, form_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract layout information from Form Recognizer result"""
        if not form_result or not form_result.get("result"):
            return {}
        
        try:
            result = form_result["result"]
            layout_info = {
                "page_width": 0,
                "page_height": 0,
                "paragraphs": [],
                "lines": []
            }
            
            # Extract page dimensions
            if result.pages:
                page = result.pages[0]
                layout_info["page_width"] = page.width if hasattr(page, 'width') else 0
                layout_info["page_height"] = page.height if hasattr(page, 'height') else 0
            
            # Extract paragraph information
            if result.paragraphs:
                for paragraph in result.paragraphs:
                    para_info = {
                        "content": paragraph.content,
                        "bounding_regions": []
                    }
                    
                    if hasattr(paragraph, 'bounding_regions') and paragraph.bounding_regions:
                        for region in paragraph.bounding_regions:
                            para_info["bounding_regions"].append({
                                "page_number": region.page_number,
                                "polygon": [{"x": p.x, "y": p.y} for p in region.polygon] if region.polygon else []
                            })
                    
                    layout_info["paragraphs"].append(para_info)
            
            return layout_info
            
        except Exception as e:
            self.logger.error("Layout extraction failed", error=str(e))
            return {}
    
    async def _extract_tables(self, form_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract table information from Form Recognizer result"""
        if not form_result or not form_result.get("result") or not self.config.enable_table_extraction:
            return []
        
        try:
            result = form_result["result"]
            tables = []
            
            if hasattr(result, 'tables') and result.tables:
                for table in result.tables:
                    table_info = {
                        "row_count": table.row_count,
                        "column_count": table.column_count,
                        "cells": []
                    }
                    
                    if hasattr(table, 'cells') and table.cells:
                        for cell in table.cells:
                            cell_info = {
                                "content": cell.content,
                                "row_index": cell.row_index,
                                "column_index": cell.column_index,
                                "row_span": getattr(cell, 'row_span', 1),
                                "column_span": getattr(cell, 'column_span', 1),
                                "kind": getattr(cell, 'kind', 'content')
                            }
                            table_info["cells"].append(cell_info)
                    
                    tables.append(table_info)
            
            return tables
            
        except Exception as e:
            self.logger.error("Table extraction failed", error=str(e))
            return []
    
    async def _extract_key_value_pairs(self, form_result: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Extract key-value pairs from Form Recognizer result"""
        if not form_result or not form_result.get("result") or not self.config.enable_key_value_extraction:
            return {}
        
        try:
            result = form_result["result"]
            key_value_pairs = {}
            
            if hasattr(result, 'key_value_pairs') and result.key_value_pairs:
                for kv_pair in result.key_value_pairs:
                    if kv_pair.key and kv_pair.value:
                        key_content = kv_pair.key.content if hasattr(kv_pair.key, 'content') else str(kv_pair.key)
                        value_content = kv_pair.value.content if hasattr(kv_pair.value, 'content') else str(kv_pair.value)
                        key_value_pairs[key_content] = value_content
            
            return key_value_pairs
            
        except Exception as e:
            self.logger.error("Key-value extraction failed", error=str(e))
            return {}
    
    async def _convert_to_images(self, document_path: str) -> List[str]:
        """Convert document to images (handles PDF and image formats)"""
        try:
            file_ext = os.path.splitext(document_path)[1].lower().replace('.', '')
            
            if file_ext == 'pdf':
                return await self._convert_pdf_to_images(document_path)
            elif file_ext in self.config.supported_formats:
                # Already an image, return as-is
                return [document_path]
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            self.logger.error("Document conversion failed", error=str(e))
            raise
    
    async def _convert_pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert PDF pages to images"""
        try:
            # Create temporary directory for images
            temp_dir = f"temp_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=200,  # Good balance between quality and file size
                fmt='png'
            )
            
            image_paths = []
            for i, image in enumerate(images):
                image_path = os.path.join(temp_dir, f"page_{i+1:03d}.png")
                image.save(image_path, 'PNG')
                image_paths.append(image_path)
            
            self.logger.info(f"PDF converted to {len(image_paths)} images", pdf=pdf_path)
            
            return image_paths
            
        except Exception as e:
            self.logger.error("PDF conversion failed", error=str(e))
            raise
    
    async def _generate_document_summary(self, pages: List[DocumentPage]) -> str:
        """Generate a summary of the document content"""
        try:
            # Combine all text content
            full_text = "\n".join(page.text_content for page in pages if page.text_content)
            
            # Simple extractive summary (first few sentences + key statistics)
            sentences = full_text.split('.')[:3]  # First 3 sentences
            summary_parts = ['. '.join(sentences) + '.']
            
            # Add document statistics
            total_words = len(full_text.split())
            avg_confidence = sum(p.confidence_score for p in pages) / len(pages) if pages else 0.0
            
            summary_parts.append(f"\n\nDocument Statistics: {len(pages)} pages, {total_words} words, {avg_confidence:.1%} average confidence.")
            
            # Add table summary if tables found
            total_tables = sum(len(page.tables) for page in pages)
            if total_tables > 0:
                summary_parts.append(f" Contains {total_tables} tables.")
            
            # Add key-value pairs summary
            total_kv_pairs = sum(len(page.key_value_pairs) for page in pages)
            if total_kv_pairs > 0:
                summary_parts.append(f" Extracted {total_kv_pairs} key-value pairs.")
            
            return ''.join(summary_parts)
            
        except Exception as e:
            self.logger.error("Document summary generation failed", error=str(e))
            return f"Summary generation failed: {str(e)}"
    
    async def _extract_entities(self, pages: List[DocumentPage]) -> List[Dict[str, Any]]:
        """Extract entities from document pages (simplified implementation)"""
        try:
            entities = []
            
            # Simple entity extraction patterns
            import re
            
            # Combine all text
            full_text = "\n".join(page.text_content for page in pages if page.text_content)
            
            # Extract dates
            date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
            dates = re.findall(date_pattern, full_text)
            for date in dates:
                entities.append({
                    "type": "date",
                    "value": date,
                    "confidence": 0.9
                })
            
            # Extract monetary amounts
            money_pattern = r'\$[\d,]+\.?\d*|\b\d+\.?\d*\s*(?:dollars?|USD|INR|₹)\b'
            amounts = re.findall(money_pattern, full_text, re.IGNORECASE)
            for amount in amounts:
                entities.append({
                    "type": "monetary_amount",
                    "value": amount,
                    "confidence": 0.8
                })
            
            # Extract percentages
            percent_pattern = r'\b\d+\.?\d*\s*%\b'
            percentages = re.findall(percent_pattern, full_text)
            for percent in percentages:
                entities.append({
                    "type": "percentage",
                    "value": percent,
                    "confidence": 0.9
                })
            
            # Extract email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, full_text)
            for email in emails:
                entities.append({
                    "type": "email",
                    "value": email,
                    "confidence": 0.95
                })
            
            # Extract phone numbers
            phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\b\+?\d{1,3}[-.]?\d{3,4}[-.]?\d{6,10}\b'
            phones = re.findall(phone_pattern, full_text)
            for phone in phones:
                entities.append({
                    "type": "phone_number",
                    "value": phone,
                    "confidence": 0.8
                })
            
            return entities[:50]  # Limit to 50 entities
            
        except Exception as e:
            self.logger.error("Entity extraction failed", error=str(e))
            return []
    
    async def _validate_document(self, document_path: str):
        """Validate document before processing"""
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Document not found: {document_path}")
        
        # Check file size
        file_size_mb = os.path.getsize(document_path) / (1024 * 1024)
        if file_size_mb > self.config.max_image_size_mb:
            raise ValueError(f"Document too large: {file_size_mb:.1f}MB > {self.config.max_image_size_mb}MB")
        
        # Check file format
        file_ext = os.path.splitext(document_path)[1].lower().replace('.', '')
        if file_ext not in self.config.supported_formats:
            raise ValueError(f"Unsupported format: {file_ext}")
    
    async def _cleanup_temp_files(self, image_paths: List[str], original_path: str):
        """Clean up temporary image files"""
        try:
            for image_path in image_paths:
                # Only delete if it's a temporary file (not the original)
                if image_path != original_path and 'temp_images_' in image_path:
                    try:
                        os.remove(image_path)
                    except:
                        pass
            
            # Remove temporary directories
            for image_path in image_paths:
                if 'temp_images_' in image_path:
                    temp_dir = os.path.dirname(image_path)
                    try:
                        os.rmdir(temp_dir)
                        break
                    except:
                        pass
                        
        except Exception as e:
            self.logger.warning("Temp file cleanup failed", error=str(e))
    
    def _estimate_processing_cost(self, num_pages: int) -> float:
        """Estimate processing cost in INR"""
        # Azure Computer Vision: ₹0.83 per 1000 transactions
        # Azure Form Recognizer: ₹4.15 per page
        cv_cost = (num_pages / 1000) * 0.83
        form_cost = num_pages * 0.004  # Approximation for prebuilt model
        
        total_cost = cv_cost + form_cost
        return total_cost
    
    def _update_metrics(self, pages_processed: int, processing_time: float, cost: float, confidence: float):
        """Update performance metrics"""
        self.metrics["documents_processed"] += 1
        self.metrics["pages_processed"] += pages_processed
        self.metrics["processing_times"].append(processing_time)
        self.metrics["costs"].append(cost)
        self.metrics["ocr_confidence_scores"].append(confidence)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        if self.metrics["documents_processed"] == 0:
            return {"message": "No documents processed yet"}
        
        return {
            "documents_processed": self.metrics["documents_processed"],
            "pages_processed": self.metrics["pages_processed"],
            "average_processing_time": sum(self.metrics["processing_times"]) / len(self.metrics["processing_times"]),
            "total_cost_inr": sum(self.metrics["costs"]),
            "average_confidence": sum(self.metrics["ocr_confidence_scores"]) / len(self.metrics["ocr_confidence_scores"]),
            "pages_per_document": self.metrics["pages_processed"] / self.metrics["documents_processed"]
        }


# Factory function
def create_multimodal_processor(
    cv_endpoint: str,
    cv_api_key: str,
    form_recognizer_endpoint: str = None,
    form_recognizer_key: str = None
) -> MultimodalDocumentProcessor:
    """Factory function to create multimodal processor"""
    
    config = MultimodalConfig(
        cv_endpoint=cv_endpoint,
        cv_api_key=cv_api_key,
        form_recognizer_endpoint=form_recognizer_endpoint or cv_endpoint.replace('cognitiveservices', 'formrecognizer'),
        form_recognizer_key=form_recognizer_key or cv_api_key
    )
    
    return MultimodalDocumentProcessor(config)


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def demo():
        processor = create_multimodal_processor(
            cv_endpoint=os.getenv("AZURE_COMPUTER_VISION_ENDPOINT"),
            cv_api_key=os.getenv("AZURE_COMPUTER_VISION_KEY")
        )
        
        # Process a document
        result = await processor.process_document(
            "data/documents/reports/sample_report.pdf",
            enable_advanced_analysis=True
        )
        
        print(f"Document: {result.document_path}")
        print(f"Pages: {result.total_pages}")
        print(f"Overall Confidence: {result.overall_confidence:.2%}")
        print(f"Processing Time: {result.total_processing_time:.2f}s")
        print(f"Cost: ₹{result.cost_estimate:.4f}")
        print(f"Summary: {result.document_summary}")
        print(f"Entities Found: {len(result.extracted_entities)}")
        
        # Show metrics
        metrics = processor.get_performance_metrics()
        print(f"Performance Metrics: {metrics}")
    
    # Run demo
    asyncio.run(demo())