import streamlit as st
import pandas as pd
import io
import base64
from datetime import datetime
import json
import re
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from pathlib import Path
import zipfile
import tempfile
import os
from dotenv import load_dotenv
import anthropic
import PyPDF2
from docx import Document
import asyncio
import time

# ==================== CONFIGURATION ====================

def initialize_app():
    """Initialize Streamlit app configuration and load environment variables"""
    load_dotenv()
    
    st.set_page_config(
        page_title="AI Business Process Optimizer",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'claude_enabled' not in st.session_state:
        st.session_state.claude_enabled = True

def load_custom_css():
    """Load custom CSS for enhanced styling"""
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #3498db;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        .optimization-tag {
            background: #f39c12;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        .step-eliminated {
            background: #ffebee;
            border-left: 4px solid #e74c3c;
            opacity: 0.7;
        }
        .step-optimized {
            background: #e8f5e8;
            border-left: 4px solid #27ae60;
        }
        .download-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .success-metric {
            background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

# ==================== CLAUDE API INTEGRATION ====================

class ClaudeAPIManager:
    """Manage Claude API interactions and error handling"""
    
    def __init__(self):
        self.api_key = self._get_api_key()
        self.client = self._initialize_client()
    
    def _get_api_key(self) -> str:
        """Get API key from environment variables"""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            st.error("❌ ANTHROPIC_API_KEY not found in .env file!")
            st.info("💡 Create a .env file with: ANTHROPIC_API_KEY=your_api_key_here")
            st.stop()
        return api_key
    
    def _initialize_client(self) -> anthropic.Anthropic:
        """Initialize Anthropic client"""
        return anthropic.Anthropic(api_key=self.api_key)
    
    def make_api_call(self, prompt: str, max_tokens: int = 4000) -> str:
        """Make API call to Claude with error handling"""
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return self._clean_response(response.content[0].text)
        except Exception as e:
            st.error(f"Error calling Claude API: {str(e)}")
            raise
    
    def _clean_response(self, response_text: str) -> str:
        """Clean and format API response"""
        response_text = response_text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        return response_text

class ProcessAnalyzer:
    """Handle process analysis using Claude AI"""
    
    def __init__(self, api_manager: ClaudeAPIManager):
        self.api_manager = api_manager
    
    def extract_process_steps(self, content: str, filename: str) -> Dict:
        """Extract and analyze process steps from document content"""
        prompt = self._build_extraction_prompt(content, filename)
        
        try:
            response = self.api_manager.make_api_call(prompt)
            return json.loads(response)
        except (json.JSONDecodeError, Exception) as e:
            st.error(f"Error analyzing document: {str(e)}")
            return self._create_fallback_analysis(filename)
    
    def _build_extraction_prompt(self, content: str, filename: str) -> str:
        """Build prompt for process extraction"""
        return f"""
        Analyze the following process document and extract detailed information about each process step.

        Document: {filename}
        Content: {content}

        Please provide a JSON response with the following structure:
        {{
            "process_name": "Name of the process",
            "process_description": "Brief description of what this process accomplishes",
            "steps": [
                {{
                    "step_number": 1,
                    "step_name": "Short descriptive name",
                    "description": "Detailed description of what happens in this step",
                    "duration_hours": 8.0,
                    "responsible_unit": "Who is responsible (HR, IT, Manager, etc.)",
                    "dependencies": ["Step 2", "Step 3"],
                    "complexity": "LOW/MEDIUM/HIGH",
                    "automation_potential": "LOW/MEDIUM/HIGH",
                    "value_add": "HIGH/MEDIUM/LOW"
                }}
            ],
            "pain_points": ["List of identified issues or inefficiencies"],
            "current_total_time": 120.0,
            "estimated_cost_per_cycle": 5000.0
        }}

        Focus on:
        1. Identifying all sequential steps in the process
        2. Estimating realistic time durations for each step
        3. Identifying bottlenecks and inefficiencies
        4. Assessing automation potential
        5. Determining value-add vs administrative overhead

        If time durations are not explicitly mentioned, estimate based on:
        - Simple administrative tasks: 2-4 hours
        - Review/approval steps: 8-16 hours  
        - Complex analysis/preparation: 16-32 hours
        - Committee meetings/collaborative work: 4-8 hours
        - Documentation/archiving: 2-6 hours

        Return only valid JSON.
        """
    
    def _create_fallback_analysis(self, filename: str) -> Dict:
        """Create fallback analysis when API fails"""
        return {
            "process_name": filename.replace('.pdf', '').replace('.txt', '').replace('.docx', '').title(),
            "process_description": "Process analysis from uploaded document",
            "steps": [
                {
                    "step_number": 1,
                    "step_name": "Step 1",
                    "description": "Initial process step",
                    "duration_hours": 8.0,
                    "responsible_unit": "Department",
                    "dependencies": [],
                    "complexity": "MEDIUM",
                    "automation_potential": "MEDIUM",
                    "value_add": "MEDIUM"
                }
            ],
            "pain_points": ["Manual processing", "Multiple handoffs"],
            "current_total_time": 40.0,
            "estimated_cost_per_cycle": 2000.0
        }

class OptimizationEngine:
    """Handle optimization recommendations using Claude AI"""
    
    def __init__(self, api_manager: ClaudeAPIManager):
        self.api_manager = api_manager
    
    def generate_optimizations(self, process_data: Dict) -> Dict:
        """Generate intelligent optimization recommendations"""
        prompt = self._build_optimization_prompt(process_data)
        
        try:
            response = self.api_manager.make_api_call(prompt)
            return json.loads(response)
        except (json.JSONDecodeError, Exception) as e:
            st.error(f"Error generating optimizations: {str(e)}")
            return self._create_fallback_optimization(process_data)
    
    def _build_optimization_prompt(self, process_data: Dict) -> str:
        """Build prompt for optimization generation"""
        return f"""
        Based on the following process analysis, generate detailed optimization recommendations:

        Process Data: {json.dumps(process_data, indent=2)}

        Please provide optimization recommendations in the following JSON format:
        {{
            "optimized_steps": [
                {{
                    "step_number": 1,
                    "step_name": "Optimized step name",
                    "description": "Description of optimized step",
                    "original_duration": 8.0,
                    "optimized_duration": 2.0,
                    "optimization_type": "ELIMINATED/MERGED/AUTOMATED/PARALLEL/DIGITAL",
                    "optimization_rationale": "Why this optimization was recommended",
                    "responsible_unit": "Who handles this optimized step",
                    "technology_requirements": ["List of technology needs"],
                    "risk_level": "LOW/MEDIUM/HIGH",
                    "implementation_complexity": "LOW/MEDIUM/HIGH"
                }}
            ],
            "eliminated_steps": [1, 3, 7],
            "merged_step_groups": [
                {{
                    "original_steps": [2, 3, 4],
                    "new_step": {{
                        "step_name": "Merged process name",
                        "description": "Combined functionality",
                        "duration": 12.0
                    }}
                }}
            ],
            "optimization_summary": {{
                "total_time_before": 120.0,
                "total_time_after": 45.0,
                "time_saved_percentage": 62.5,
                "steps_eliminated": 5,
                "automation_opportunities": 8,
                "estimated_implementation_cost": 25000.0,
                "estimated_annual_savings": 75000.0,
                "payback_months": 4.0,
                "key_benefits": ["List of main benefits"],
                "implementation_phases": [
                    {{
                        "phase": "Phase 1: Foundation",
                        "duration_months": 2,
                        "activities": ["Activity 1", "Activity 2"],
                        "cost": 10000.0
                    }}
                ]
            }}
        }}

        Optimization Strategies to Consider:
        1. ELIMINATION: Remove steps that don't add value
        2. AUTOMATION: Replace manual work with technology
        3. PARALLELIZATION: Run independent steps simultaneously  
        4. DIGITIZATION: Convert paper-based processes to digital
        5. CONSOLIDATION: Merge similar or related activities
        6. DELEGATION: Push decisions to appropriate levels
        7. STANDARDIZATION: Create consistent templates and procedures

        Focus on:
        - Maximum time reduction while maintaining quality
        - Realistic implementation timelines
        - Cost-benefit analysis
        - Risk mitigation strategies
        - Technology enablers

        Return only valid JSON.
        """
    
    def _create_fallback_optimization(self, process_data: Dict) -> Dict:
        """Create fallback optimization when API fails"""
        return {
            "optimized_steps": process_data["steps"][:3],
            "eliminated_steps": [4, 5, 6] if len(process_data["steps"]) > 3 else [],
            "optimization_summary": {
                "total_time_before": process_data.get("current_total_time", 40.0),
                "total_time_after": 20.0,
                "time_saved_percentage": 50.0,
                "steps_eliminated": 3,
                "automation_opportunities": 2,
                "estimated_implementation_cost": 20000.0,
                "estimated_annual_savings": 50000.0,
                "payback_months": 4.8,
                "key_benefits": ["Faster processing", "Reduced errors", "Lower costs"]
            }
        }

class ImplementationPlanner:
    """Generate implementation plans using Claude AI"""
    
    def __init__(self, api_manager: ClaudeAPIManager):
        self.api_manager = api_manager
    
    def generate_implementation_plan(self, optimization_data: Dict) -> str:
        """Generate detailed implementation plan"""
        prompt = self._build_implementation_prompt(optimization_data)
        
        try:
            return self.api_manager.make_api_call(prompt)
        except Exception as e:
            st.error(f"Error generating implementation plan: {str(e)}")
            return "Implementation plan generation failed."
    
    def _build_implementation_prompt(self, optimization_data: Dict) -> str:
        """Build prompt for implementation plan generation"""
        return f"""
        Create a comprehensive implementation plan for the following process optimization:

        Optimization Data: {json.dumps(optimization_data, indent=2)}

        Generate a detailed implementation plan with the following sections:

        1. EXECUTIVE SUMMARY
        2. IMPLEMENTATION ROADMAP (with specific timelines)
        3. TECHNOLOGY REQUIREMENTS
        4. CHANGE MANAGEMENT STRATEGY
        5. RISK MITIGATION PLAN
        6. SUCCESS METRICS AND KPIs
        7. BUDGET BREAKDOWN
        8. RESOURCE REQUIREMENTS
        9. TRAINING PLAN
        10. COMMUNICATION STRATEGY

        Format as a professional document suitable for executive presentation.
        Include specific timelines, costs, and measurable outcomes.
        """

# ==================== DOCUMENT PROCESSING ====================

class DocumentExtractor:
    """Enhanced document content extraction with multiple format support"""
    
    @staticmethod
    def extract_text_from_pdf(file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file) -> str:
        """Extract text from TXT file"""
        try:
            return str(file.read(), "utf-8")
        except Exception as e:
            st.error(f"Error reading TXT: {str(e)}")
            return ""
    
    @classmethod
    def extract_content(cls, file) -> str:
        """Extract content based on file type"""
        if file.type == "application/pdf":
            return cls.extract_text_from_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return cls.extract_text_from_docx(file)
        elif file.type == "text/plain":
            return cls.extract_text_from_txt(file)
        else:
            # Fallback - try as text
            try:
                return str(file.read(), "utf-8", errors='ignore')
            except:
                return ""

# ==================== CORE PROCESSING ====================

class ProcessOptimizer:
    """Main process optimization orchestrator"""
    
    def __init__(self):
        self.api_manager = ClaudeAPIManager()
        self.analyzer = ProcessAnalyzer(self.api_manager)
        self.optimizer = OptimizationEngine(self.api_manager)
        self.planner = ImplementationPlanner(self.api_manager)
        self.doc_extractor = DocumentExtractor()
    
    def analyze_document(self, file) -> Dict:
        """Analyze uploaded document using Claude"""
        content = self._extract_and_validate_content(file)
        
        # Use Claude for intelligent analysis
        with st.spinner(f"Analyzing {file.name} ..."):
            process_data = self.analyzer.extract_process_steps(content, file.name)
        
        # Generate optimizations
        with st.spinner("Generating optimization recommendations..."):
            optimization_data = self.optimizer.generate_optimizations(process_data)
        
        return {
            "process_data": process_data,
            "optimization_data": optimization_data,
            "original_content": content[:1000] + "..." if len(content) > 1000 else content
        }
    
    def _extract_and_validate_content(self, file) -> str:
        """Extract and validate document content"""
        content = self.doc_extractor.extract_content(file)
        
        if not content or len(content.strip()) < 100:
            st.warning(f"⚠️ Limited content extracted from {file.name}. Results may be incomplete.")
            content = f"Process document: {file.name}\nContent extraction was limited."
        
        return content
    
    def generate_comprehensive_report(self, analysis_results: List[Dict]) -> Dict:
        """Generate portfolio-level analysis report"""
        return {
            "portfolio_metrics": PortfolioAnalyzer.calculate_portfolio_metrics(analysis_results),
            "summary": f"Portfolio analysis of {len(analysis_results)} processes",
            "generated_at": datetime.now().isoformat()
        }

# ==================== PORTFOLIO ANALYSIS ====================

class PortfolioAnalyzer:
    """Analyze multiple processes as a portfolio"""
    
    @staticmethod
    def calculate_portfolio_metrics(analysis_results: List[Dict]) -> Dict:
        """Calculate portfolio-level metrics safely"""
        portfolio_data = {
            "processes": [],
            "total_savings": 0,
            "total_time_saved": 0,
            "average_roi": 0,
            "implementation_cost": 0,
            "valid_analyses": 0
        }
        
        for result in analysis_results:
            try:
                # Safely navigate nested structure
                analysis_result = result.get('analysis_result', {})
                process_data = analysis_result.get('process_data', {})
                optimization_data = analysis_result.get('optimization_data', {})
                opt_summary = optimization_data.get('optimization_summary', {})
                
                # Only process if we have complete data
                if all([process_data, optimization_data, opt_summary]):
                    portfolio_data["processes"].append({
                        "name": process_data.get('process_name', 'Unknown Process'),
                        "time_saved": opt_summary.get('time_saved_percentage', 0),
                        "annual_savings": opt_summary.get('estimated_annual_savings', 0),
                        "implementation_cost": opt_summary.get('estimated_implementation_cost', 0)
                    })
                    
                    portfolio_data["total_savings"] += opt_summary.get('estimated_annual_savings', 0)
                    portfolio_data["total_time_saved"] += opt_summary.get('time_saved_percentage', 0)
                    portfolio_data["implementation_cost"] += opt_summary.get('estimated_implementation_cost', 0)
                    portfolio_data["valid_analyses"] += 1
                    
            except Exception as e:
                # Skip invalid results
                continue
        
        # Calculate averages safely
        if portfolio_data["valid_analyses"] > 0:
            portfolio_data["average_time_saved"] = portfolio_data["total_time_saved"] / portfolio_data["valid_analyses"]
            if portfolio_data["implementation_cost"] > 0:
                portfolio_data["average_roi"] = (
                    (portfolio_data["total_savings"] * 3 - portfolio_data["implementation_cost"]) 
                    / portfolio_data["implementation_cost"] * 100
                )
        
        return portfolio_data

# ==================== DOCUMENT GENERATION ====================

class ReportGenerator:
    """Generate various types of reports and documents"""
    
    def __init__(self, planner: ImplementationPlanner):
        self.planner = planner
    
    def generate_comprehensive_report(self, analysis_result: Dict) -> str:
        """Generate comprehensive markdown report"""
        process_data = analysis_result["process_data"]
        optimization_data = analysis_result["optimization_data"]
        opt_summary = optimization_data["optimization_summary"]
        
        # Generate implementation plan using Claude
        implementation_plan = self.planner.generate_implementation_plan(optimization_data)
        
        return self._build_markdown_report(process_data, optimization_data, opt_summary, implementation_plan)
    
    def _build_markdown_report(self, process_data: Dict, optimization_data: Dict, 
                             opt_summary: Dict, implementation_plan: str) -> str:
        """Build comprehensive markdown report"""
        report = f"""
# {process_data['process_name']} - Comprehensive Optimization Analysis

## Executive Summary

**Process Overview:**
{process_data['process_description']}

**Key Metrics:**
- **Steps Reduction**: {len(process_data['steps'])} → {len(optimization_data['optimized_steps'])} ({len(optimization_data.get('eliminated_steps', []))} eliminated)
- **Time Improvement**: {opt_summary['total_time_before']:.1f} → {opt_summary['total_time_after']:.1f} hours ({opt_summary['time_saved_percentage']:.1f}% reduction)
- **Annual Savings**: AED {opt_summary['estimated_annual_savings']:,.0f}
- **Implementation Cost**: AED {opt_summary['estimated_implementation_cost']:,.0f}
- **Payback Period**: {opt_summary['payback_months']:.1f} months
- **3-Year ROI**: {((opt_summary['estimated_annual_savings'] * 3 - opt_summary['estimated_implementation_cost']) / opt_summary['estimated_implementation_cost'] * 100):.1f}%

---

## Current Process Analysis

### Identified Pain Points
"""
        
        for pain_point in process_data.get('pain_points', []):
            report += f"- {pain_point}\n"
        
        report += self._add_current_process_steps(process_data, optimization_data)
        report += self._add_optimization_strategy(optimization_data)
        report += self._add_financial_impact(opt_summary)
        report += f"\n---\n\n## Implementation Plan\n\n{implementation_plan}\n"
        report += self._add_conclusion(process_data, opt_summary)
        
        return report
    
    def _add_current_process_steps(self, process_data: Dict, optimization_data: Dict) -> str:
        """Add current process steps section"""
        opt_summary = optimization_data['optimization_summary']
        section = f"""

### Current Process Steps ({len(process_data['steps'])} steps - {opt_summary['total_time_before']:.1f} hours)

"""
        
        for step in process_data['steps']:
            eliminated = step['step_number'] in optimization_data.get('eliminated_steps', [])
            status = "🔴 ELIMINATED" if eliminated else "🟡 TO BE OPTIMIZED"
            
            section += f"""
**Step {step['step_number']}: {step['step_name']}** {status}
- **Description**: {step['description']}
- **Duration**: {step['duration_hours']} hours
- **Responsible**: {step['responsible_unit']}
- **Complexity**: {step.get('complexity', 'MEDIUM')}
- **Automation Potential**: {step.get('automation_potential', 'MEDIUM')}
- **Value Add**: {step.get('value_add', 'MEDIUM')}
- **Dependencies**: {', '.join(step.get('dependencies', [])) or 'None'}

"""
        return section
    
    def _add_optimization_strategy(self, optimization_data: Dict) -> str:
        """Add optimization strategy section"""
        opt_summary = optimization_data['optimization_summary']
        section = f"""
---

## Optimization Strategy

### Optimized Process Steps ({len(optimization_data['optimized_steps'])} steps - {opt_summary['total_time_after']:.1f} hours)

"""
        
        for step in optimization_data['optimized_steps']:
            time_savings_pct = ((step['original_duration'] - step['optimized_duration'])/step['original_duration']*100)
            
            section += f"""
**Step {step['step_number']}: {step['step_name']}** ({step['optimization_type']})
- **Description**: {step['description']}
- **Original Duration**: {step['original_duration']} hours
- **Optimized Duration**: {step['optimized_duration']} hours
- **Time Savings**: {step['original_duration'] - step['optimized_duration']:.1f} hours ({time_savings_pct:.1f}%)
- **Responsible**: {step['responsible_unit']}
- **Optimization Rationale**: {step['optimization_rationale']}
- **Technology Requirements**: {', '.join(step.get('technology_requirements', [])) or 'None'}
- **Implementation Risk**: {step.get('risk_level', 'MEDIUM')}
- **Implementation Complexity**: {step.get('implementation_complexity', 'MEDIUM')}

"""
        return section
    
    def _add_financial_impact(self, opt_summary: Dict) -> str:
        """Add financial impact section"""
        return f"""
---

## Financial Impact Analysis

### Cost-Benefit Summary
- **Total Time Saved**: {opt_summary['total_time_before'] - opt_summary['total_time_after']:.1f} hours per cycle
- **Time Savings**: {opt_summary['time_saved_percentage']:.1f}%
- **Annual Savings**: AED {opt_summary['estimated_annual_savings']:,.0f}
- **Implementation Investment**: AED {opt_summary['estimated_implementation_cost']:,.0f}
- **Payback Period**: {opt_summary['payback_months']:.1f} months
- **Break-even Analysis**: Month {opt_summary['payback_months']:.0f}

### Key Benefits
""" + "".join([f"- {benefit}\n" for benefit in opt_summary.get('key_benefits', [])]) + f"""

### Optimization Impact Breakdown
- **Steps Eliminated**: {opt_summary['steps_eliminated']} steps
- **Automation Opportunities**: {opt_summary['automation_opportunities']} processes
- **Efficiency Improvement**: {opt_summary['time_saved_percentage']:.1f}%
"""
    
    def _add_conclusion(self, process_data: Dict, opt_summary: Dict) -> str:
        """Add conclusion section"""
        return f"""
---

## Conclusion

The optimization of {process_data['process_name']} represents a significant opportunity for operational transformation. With a {opt_summary['time_saved_percentage']:.1f}% reduction in processing time and AED {opt_summary['estimated_annual_savings']:,.0f} in annual savings, this initiative delivers compelling ROI while enhancing user experience and process quality.

The recommended approach balances ambitious optimization goals with practical implementation considerations, ensuring sustainable long-term benefits for the organization.

---

*Report generated by AI Business Process Optimizer with Claude Sonnet 4*
*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

class SVGGenerator:
    """Generate SVG workflow diagrams"""
    
    
    @staticmethod
    def generate_advanced_svg_workflow(analysis_result: Dict) -> str:
        """Generate enhanced SVG workflow with optimization insights"""
        process_data = analysis_result["process_data"]
        optimization_data = analysis_result["optimization_data"]
        opt_summary = optimization_data["optimization_summary"]
        
        # Calculate positions for steps
        current_steps = process_data.get('steps', [])
        optimized_steps = optimization_data.get('optimized_steps', [])
        eliminated_steps = optimization_data.get('eliminated_steps', [])
        
        return f'''<?xml version="1.0" encoding="UTF-8"?>
    <svg width="1400" height="1200" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <style>
        .title-font {{ font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif; }}
        .body-font {{ font-family: 'Inter', 'Segoe UI', 'system-ui', sans-serif; }}
        .metric-font {{ font-family: 'SF Pro Display', 'Segoe UI', 'Roboto', sans-serif; }}
        </style>
        
        <linearGradient id="headerGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
        <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
        </linearGradient>
        
        <linearGradient id="successGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:#56ab2f;stop-opacity:1" />
        <stop offset="100%" style="stop-color:#a8e6cf;stop-opacity:1" />
        </linearGradient>
        
        <linearGradient id="eliminatedGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:#e74c3c;stop-opacity:1" />
        <stop offset="100%" style="stop-color:#c0392b;stop-opacity:1" />
        </linearGradient>
        
        <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="3" dy="6" stdDeviation="4" flood-color="#000" flood-opacity="0.2"/>
        </filter>
        
        <filter id="glow">
        <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
        <feMerge> 
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
        </feMerge>
        </filter>
    </defs>
    
    <!-- Background -->
    <rect width="1400" height="1200" fill="#f8fafc"/>
    
    <!-- Header -->
    <rect x="0" y="0" width="1400" height="120" fill="url(#headerGradient)" filter="url(#shadow)"/>
    <text x="700" y="45" text-anchor="middle" fill="white" font-size="32" class="title-font" font-weight="300" letter-spacing="1px">{process_data['process_name']} - AI-Optimized Workflow</text>
    <text x="700" y="75" text-anchor="middle" fill="white" font-size="16" class="body-font" font-weight="400" opacity="0.9">Powered by Claude Sonnet 4 | {opt_summary.get('time_saved_percentage', 0):.1f}% Time Reduction | AED {opt_summary.get('estimated_annual_savings', 0):,.0f} Annual Savings</text>
    <text x="700" y="100" text-anchor="middle" fill="white" font-size="14" class="body-font" font-weight="300" opacity="0.8">🚀 Implementation ROI: {((opt_summary.get('estimated_annual_savings', 0) * 3 - opt_summary.get('estimated_implementation_cost', 0)) / max(opt_summary.get('estimated_implementation_cost', 1), 1) * 100):.1f}% over 3 years</text>
    
    <!-- Metrics Dashboard -->
    <rect x="50" y="150" width="1300" height="80" fill="white" stroke="#e2e8f0" stroke-width="1" rx="16" filter="url(#shadow)"/>
    
    <!-- Metric Cards -->
    <g transform="translate(100, 170)">
        <rect width="220" height="40" fill="url(#successGradient)" rx="20" filter="url(#glow)"/>
        <text x="110" y="18" text-anchor="middle" fill="white" font-size="14" class="metric-font" font-weight="600">Steps: {len(current_steps)} → {len(optimized_steps)}</text>
        <text x="110" y="32" text-anchor="middle" fill="white" font-size="10" class="body-font" font-weight="400" opacity="0.9">({len(eliminated_steps)} eliminated)</text>
    </g>
    
    <g transform="translate(350, 170)">
        <rect width="220" height="40" fill="#4a90e2" rx="20" filter="url(#glow)"/>
        <text x="110" y="18" text-anchor="middle" fill="white" font-size="14" class="metric-font" font-weight="600">Time: {opt_summary.get('total_time_before', 0):.0f}h → {opt_summary.get('total_time_after', 0):.0f}h</text>
        <text x="110" y="32" text-anchor="middle" fill="white" font-size="10" class="body-font" font-weight="400" opacity="0.9">({opt_summary.get('time_saved_percentage', 0):.1f}% faster)</text>
    </g>
    
    <g transform="translate(600, 170)">
        <rect width="220" height="40" fill="#f59e0b" rx="20" filter="url(#glow)"/>
        <text x="110" y="18" text-anchor="middle" fill="white" font-size="14" class="metric-font" font-weight="600">Savings: AED {opt_summary.get('estimated_annual_savings', 0)/1000:.0f}K</text>
        <text x="110" y="32" text-anchor="middle" fill="white" font-size="10" class="body-font" font-weight="400" opacity="0.9">Annual Value</text>
    </g>
    
    <g transform="translate(850, 170)">
        <rect width="220" height="40" fill="#ef4444" rx="20" filter="url(#glow)"/>
        <text x="110" y="18" text-anchor="middle" fill="white" font-size="14" class="metric-font" font-weight="600">Payback: {opt_summary.get('payback_months', 0):.1f} months</text>
        <text x="110" y="32" text-anchor="middle" fill="white" font-size="10" class="body-font" font-weight="400" opacity="0.9">Break-even point</text>
    </g>
    
    <g transform="translate(1100, 170)">
        <rect width="220" height="40" fill="#8b5cf6" rx="20" filter="url(#glow)"/>
        <text x="110" y="18" text-anchor="middle" fill="white" font-size="14" class="metric-font" font-weight="600">AI Ops: {opt_summary.get('automation_opportunities', 0)}</text>
        <text x="110" y="32" text-anchor="middle" fill="white" font-size="10" class="body-font" font-weight="400" opacity="0.9">Automation Opportunities</text>
    </g>
    
    <!-- Process Comparison Section -->
    <text x="700" y="280" text-anchor="middle" fill="#1e293b" font-size="22" class="title-font" font-weight="500" letter-spacing="0.5px">🔄 AI-Powered Process Transformation</text>
    
    <!-- Before Process -->
    <rect x="70" y="300" width="600" height="450" fill="#fef2f2" stroke="#ef4444" stroke-width="2" rx="20" filter="url(#shadow)"/>
    <rect x="80" y="310" width="580" height="40" fill="#dc2626" rx="12"/>
    <text x="90" y="335" fill="white" font-size="16" class="title-font" font-weight="500">❌ BEFORE: Current Process ({len(current_steps)} steps - {opt_summary.get('total_time_before', 0):.0f}h)</text>
    
    <!-- Current Process Steps -->
    {SVGGenerator._generate_current_steps_svg(current_steps, eliminated_steps, 90, 365)}
    
    <!-- After Process -->
    <rect x="730" y="300" width="600" height="450" fill="#f0fdf4" stroke="#22c55e" stroke-width="2" rx="20" filter="url(#shadow)"/>
    <rect x="740" y="310" width="580" height="40" fill="#16a34a" rx="12"/>
    <text x="750" y="335" fill="white" font-size="16" class="title-font" font-weight="500">✅ AFTER: Optimized Process ({len(optimized_steps)} steps - {opt_summary.get('total_time_after', 0):.0f}h)</text>
    
    <!-- Optimized Process Steps -->
    {SVGGenerator._generate_optimized_steps_svg(optimized_steps, 750, 365)}
    
    <!-- Transformation Arrow -->
    <g transform="translate(670, 525)">
        <circle cx="0" cy="0" r="30" fill="#f59e0b" filter="url(#shadow)"/>
        <text x="0" y="8" text-anchor="middle" fill="white" font-size="24" class="title-font" font-weight="300">→</text>
    </g>
    
    <!-- Benefits Summary -->
    <rect x="70" y="800" width="1260" height="100" fill="white" stroke="#e2e8f0" stroke-width="1" rx="16" filter="url(#shadow)"/>
    <text x="700" y="830" text-anchor="middle" fill="#1e293b" font-size="20" class="title-font" font-weight="500" letter-spacing="0.5px">💎 Key Benefits</text>
    
    <!-- Benefit items -->
    <text x="90" y="860" fill="#475569" font-size="14" class="body-font" font-weight="400">• {opt_summary.get('time_saved_percentage', 0):.1f}% faster processing</text>
    <text x="90" y="880" fill="#475569" font-size="14" class="body-font" font-weight="400">• AED {opt_summary.get('estimated_annual_savings', 0):,.0f} annual savings</text>
    
    <text x="450" y="860" fill="#475569" font-size="14" class="body-font" font-weight="400">• {len(eliminated_steps)} steps eliminated</text>
    <text x="450" y="880" fill="#475569" font-size="14" class="body-font" font-weight="400">• {opt_summary.get('automation_opportunities', 0)} automation opportunities</text>
    
    <text x="800" y="860" fill="#475569" font-size="14" class="body-font" font-weight="400">• ROI payback in {opt_summary.get('payback_months', 0):.1f} months</text>
    <text x="800" y="880" fill="#475569" font-size="14" class="body-font" font-weight="400">• Reduced errors and improved quality</text>
    
    <!-- Implementation Timeline -->
    <rect x="70" y="930" width="1260" height="80" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="16"/>
    <text x="700" y="955" text-anchor="middle" fill="#1e293b" font-size="18" class="title-font" font-weight="500" letter-spacing="0.5px">📅 Implementation Timeline</text>
    
    <!-- Timeline phases -->
    <rect x="100" y="965" width="200" height="28" fill="#3b82f6" rx="14" filter="url(#glow)"/>
    <text x="200" y="982" text-anchor="middle" fill="white" font-size="12" class="body-font" font-weight="500">Phase 1: Planning (Month 1)</text>
    
    <rect x="320" y="965" width="200" height="28" fill="#f59e0b" rx="14" filter="url(#glow)"/>
    <text x="420" y="982" text-anchor="middle" fill="white" font-size="12" class="body-font" font-weight="500">Phase 2: Implementation (Months 2-3)</text>
    
    <rect x="540" y="965" width="200" height="28" fill="#10b981" rx="14" filter="url(#glow)"/>
    <text x="640" y="982" text-anchor="middle" fill="white" font-size="12" class="body-font" font-weight="500">Phase 3: Training (Month 4)</text>
    
    <rect x="760" y="965" width="200" height="28" fill="#8b5cf6" rx="14" filter="url(#glow)"/>
    <text x="860" y="982" text-anchor="middle" fill="white" font-size="12" class="body-font" font-weight="500">Phase 4: Go-Live (Month 5)</text>
    
    <rect x="980" y="965" width="200" height="28" fill="#f97316" rx="14" filter="url(#glow)"/>
    <text x="1080" y="982" text-anchor="middle" fill="white" font-size="12" class="body-font" font-weight="500">Phase 5: Optimization (Month 6)</text>
    
    <!-- Footer -->
    <rect x="0" y="1040" width="1400" height="60" fill="#1e293b"/>
    <text x="700" y="1065" text-anchor="middle" fill="white" font-size="16" class="title-font" font-weight="400" letter-spacing="0.5px">AI Business Process Optimizer - Powered by Claude Sonnet 4</text>
    <text x="700" y="1085" text-anchor="middle" fill="white" font-size="12" class="body-font" font-weight="300" opacity="0.8">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Transform your business processes with AI</text>
    
    </svg>'''

    @staticmethod
    def _generate_current_steps_svg(steps, eliminated_steps, start_x, start_y):
        """Generate SVG for current process steps"""
        svg_content = ""
        y_pos = start_y
        
        for i, step in enumerate(steps[:8]):  # Limit to 8 steps for space
            is_eliminated = step.get('step_number', i+1) in eliminated_steps
            
            if is_eliminated:
                color = "#ef4444"
                text_color = "white"
                status = "ELIMINATED"
                opacity = "0.8"
            else:
                color = "#64748b"
                text_color = "white"
                status = "CURRENT"
                opacity = "1"
            
            svg_content += f'''
            <rect x="{start_x}" y="{y_pos}" width="500" height="38" fill="{color}" rx="12" opacity="{opacity}"/>
            <text x="{start_x + 15}" y="{y_pos + 16}" fill="{text_color}" font-size="13" class="body-font" font-weight="500">Step {step.get('step_number', i+1)}: {step.get('step_name', 'Process Step')[:35]}...</text>
            <text x="{start_x + 15}" y="{y_pos + 32}" fill="{text_color}" font-size="11" class="body-font" font-weight="400" opacity="0.9">{step.get('duration_hours', 0)}h | {status}</text>
            '''
            y_pos += 48
        
        if len(steps) > 8:
            svg_content += f'''
            <text x="{start_x + 15}" y="{y_pos + 15}" fill="#64748b" font-size="12" class="body-font" font-weight="400">... and {len(steps) - 8} more steps</text>
            '''
        
        return svg_content

    @staticmethod
    def _generate_optimized_steps_svg(optimized_steps, start_x, start_y):
        """Generate SVG for optimized process steps"""
        svg_content = ""
        y_pos = start_y
        
        optimization_colors = {
            'AUTOMATED': '#3b82f6',
            'MERGED': '#f59e0b', 
            'PARALLEL': '#8b5cf6',
            'DIGITAL': '#06b6d4',
            'ELIMINATED': '#ef4444'
        }
        
        for i, step in enumerate(optimized_steps[:8]):  # Limit to 8 steps for space
            opt_type = step.get('optimization_type', 'OPTIMIZED')
            color = optimization_colors.get(opt_type, '#10b981')
            
            svg_content += f'''
            <rect x="{start_x}" y="{y_pos}" width="500" height="38" fill="{color}" rx="12" filter="url(#glow)"/>
            <text x="{start_x + 15}" y="{y_pos + 16}" fill="white" font-size="13" class="body-font" font-weight="500">Step {step.get('step_number', i+1)}: {step.get('step_name', 'Optimized Step')[:35]}...</text>
            <text x="{start_x + 15}" y="{y_pos + 32}" fill="white" font-size="11" class="body-font" font-weight="400" opacity="0.9">{step.get('optimized_duration', 0)}h | {opt_type}</text>
            '''
            y_pos += 48
        
        if len(optimized_steps) > 8:
            svg_content += f'''
            <text x="{start_x + 15}" y="{y_pos + 15}" fill="#10b981" font-size="12" class="body-font" font-weight="400">... and {len(optimized_steps) - 8} more optimized steps</text>
            '''
        
        return svg_content
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


# ==================== UI COMPONENTS ====================

class UIComponents:
    """Reusable UI components for the Streamlit interface"""
    
    @staticmethod
    def render_header_with_status():
        """Render application header with API status"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            <div class="main-header">
                <h1>🤖 AI-Powered Business Process Optimizer</h1>
                <p>Upload 1-5 process documents for intelligent optimization analysis powered by Claude Sonnet 4</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            UIComponents._check_api_status()
    
    @staticmethod
    def _check_api_status():
        """Check and display API connection status"""
        try:
            ClaudeAPIManager()
            st.success("✅ Claude Sonnet 4 Connected")
            st.caption("AI-powered analysis enabled")
        except Exception as e:
            st.error("❌ Claude API Error")
            st.caption("Check .env file for ANTHROPIC_API_KEY")
            st.stop()
    
    @staticmethod
    def render_sidebar_config():
        """Render enhanced sidebar configuration"""
        with st.sidebar:
            st.header("🎛️ Analysis Configuration")
            
            config = UIComponents._get_analysis_settings()
            config.update(UIComponents._get_financial_parameters())
            config.update(UIComponents._get_output_preferences())
            UIComponents._render_quick_actions()
            
            return config
    
    @staticmethod
    def _get_analysis_settings():
        """Get analysis configuration settings"""
        st.subheader("AI Analysis Settings")
        return {
            'analysis_depth': st.selectbox(
                "Analysis Depth",
                ["Standard", "Comprehensive", "Deep Dive"],
                index=1,
                help="Choose the level of analysis detail"
            ),
            'include_ai_recommendations': st.checkbox("AI Optimization Recommendations", value=True),
            'include_implementation_plan': st.checkbox("Auto-generate Implementation Plan", value=True),
            'include_risk_analysis': st.checkbox("AI Risk Assessment", value=True)
        }
    
    @staticmethod
    def _get_financial_parameters():
        """Get financial configuration parameters"""
        st.divider()
        st.subheader("💰 Financial Parameters")
        return {
            'cost_per_hour': st.number_input("Cost per Hour (AED)", min_value=10.0, max_value=500.0, value=50.0, step=5.0),
            'annual_cycles': st.number_input("Annual Process Cycles", min_value=1, max_value=10000, value=50, step=10),
            'implementation_budget': st.number_input("Implementation Budget (AED)", min_value=5000.0, max_value=1000000.0, value=50000.0, step=5000.0)
        }
    
    @staticmethod
    def _get_output_preferences():
        """Get output format preferences"""
        st.divider()
        st.subheader("📄 Output Preferences")
        return {
            'output_format': st.selectbox("Report Format", ["Comprehensive", "Executive Summary", "Technical Detail"]),
            'include_charts': st.checkbox("Include Visualization Charts", value=True),
            'include_timeline': st.checkbox("Include Implementation Timeline", value=True)
        }
    
    @staticmethod
    def _render_quick_actions():
        """Render quick action buttons"""
        st.divider()
        st.subheader("🚀 Quick Actions")
        
        if st.button("🔄 Clear All Results", type="secondary", use_container_width=True):
            st.session_state.analysis_results = []
            st.cache_data.clear()
            st.success("Results cleared!")
            st.rerun()
        
        if st.session_state.analysis_results:
            UIComponents._render_portfolio_download()
    
    
    
    
    @staticmethod
    def _render_portfolio_download():
        """Render portfolio download button"""
        if st.button("📦 Download Portfolio Report", type="primary", use_container_width=True):
            with st.spinner("Generating portfolio report..."):
                try:
                    # Calculate portfolio metrics safely
                    portfolio_data = {
                        "processes": [],
                        "total_savings": 0,
                        "total_time_saved": 0,
                        "average_roi": 0,
                        "implementation_cost": 0,
                        "total_processes": len(st.session_state.analysis_results)
                    }
                    
                    for result in st.session_state.analysis_results:
                        # Safely access nested data
                        analysis_result = result.get('analysis_result', {})
                        process_data = analysis_result.get('process_data', {})
                        optimization_data = analysis_result.get('optimization_data', {})
                        opt_summary = optimization_data.get('optimization_summary', {})
                        
                        # Only process if we have valid data
                        if process_data and optimization_data and opt_summary:
                            portfolio_data["processes"].append({
                                "name": process_data.get('process_name', 'Unknown Process'),
                                "time_saved": opt_summary.get('time_saved_percentage', 0),
                                "annual_savings": opt_summary.get('estimated_annual_savings', 0),
                                "implementation_cost": opt_summary.get('estimated_implementation_cost', 0)
                            })
                            
                            portfolio_data["total_savings"] += opt_summary.get('estimated_annual_savings', 0)
                            portfolio_data["total_time_saved"] += opt_summary.get('time_saved_percentage', 0)
                            portfolio_data["implementation_cost"] += opt_summary.get('estimated_implementation_cost', 0)
                    
                    # Calculate averages safely
                    if portfolio_data["processes"]:
                        portfolio_data["average_time_saved"] = portfolio_data["total_time_saved"] / len(portfolio_data["processes"])
                        if portfolio_data["implementation_cost"] > 0:
                            portfolio_data["average_roi"] = (
                                (portfolio_data["total_savings"] * 3 - portfolio_data["implementation_cost"]) 
                                / portfolio_data["implementation_cost"] * 100
                            )
                        else:
                            portfolio_data["average_roi"] = 0
                    else:
                        portfolio_data["average_time_saved"] = 0
                        portfolio_data["average_roi"] = 0
                    
                    # Generate portfolio JSON
                    portfolio_json = json.dumps(portfolio_data, indent=2)
                    
                    st.download_button(
                        label="📊 Download Portfolio Data (JSON)",
                        data=portfolio_json.encode('utf-8'),
                        file_name=f"Portfolio_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="portfolio_download"
                    )
                    
                    # Show preview of portfolio data
                    st.success(f"✅ Portfolio report generated for {len(portfolio_data['processes'])} processes")
                    
                    with st.expander("📊 Portfolio Summary Preview"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Processes", portfolio_data["total_processes"])
                            st.metric("Avg Time Saved", f"{portfolio_data['average_time_saved']:.1f}%")
                        
                        with col2:
                            st.metric("Total Annual Savings", f"AED {portfolio_data['total_savings']:,.0f}")
                            st.metric("Total Implementation Cost", f"AED {portfolio_data['implementation_cost']:,.0f}")
                        
                        with col3:
                            st.metric("Portfolio ROI", f"{portfolio_data['average_roi']:.1f}%")
                            
                            if portfolio_data["implementation_cost"] > 0:
                                payback_months = (portfolio_data["implementation_cost"] / 
                                            max(portfolio_data["total_savings"] / 12, 1))
                                st.metric("Payback Period", f"{payback_months:.1f} months")
                    
                except Exception as e:
                    st.error(f"❌ Error generating portfolio report: {str(e)}")
                    st.info("💡 Please ensure all analyses completed successfully before generating portfolio report.")
                    
                    # Debug information
                    if st.checkbox("Show debug info"):
                        st.write("Analysis results structure:")
                        for i, result in enumerate(st.session_state.analysis_results):
                            st.write(f"Result {i+1} keys:", list(result.keys()) if result else "None")
                            if result and 'analysis_result' in result:
                                st.write(f"  - analysis_result keys:", list(result['analysis_result'].keys()))
                            
                            
                            

class FileHandler:
    """Handle file upload and validation"""
    
    @staticmethod
    def handle_file_upload():
        """Handle file upload with validation and preview"""
        uploaded_files = st.file_uploader(
            "Upload 1-5 process documents (PDF, TXT, DOCX)",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="📋 Supported formats: PDF, Word documents, and text files. Maximum 5 files per analysis."
        )
        
        if uploaded_files:
            uploaded_files = FileHandler._validate_file_count(uploaded_files)
            FileHandler._display_file_preview(uploaded_files)
            return uploaded_files
        
        FileHandler._display_empty_state()
        return None
    
    @staticmethod
    def _validate_file_count(uploaded_files):
        """Validate and limit file count"""
        if len(uploaded_files) > 5:
            st.error("⚠️ Please upload maximum 5 files. Using first 5 files.")
            return uploaded_files[:5]
        return uploaded_files
    
    @staticmethod
    def _display_file_preview(uploaded_files):
        """Display file details and preview"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
            
            for file in uploaded_files:
                with st.expander(f"📄 {file.name} ({file.size:,} bytes)"):
                    st.write(f"**Type:** {file.type}")
                    st.write(f"**Size:** {file.size:,} bytes")
                    
                    if file.type == "text/plain":
                        FileHandler._show_text_preview(file)
        
        with col2:
            FileHandler._display_analysis_summary()
    
    @staticmethod
    def _show_text_preview(file):
        """Show preview for text files"""
        try:
            content_preview = str(file.read(500), "utf-8")
            file.seek(0)  # Reset file pointer
            st.text_area("Preview:", content_preview, height=100, disabled=True)
        except:
            st.info("Preview not available")
    
    @staticmethod
    def _display_analysis_summary():
        """Display analysis configuration summary"""
        st.markdown("**Analysis Configuration:**")
        # Note: In a real implementation, you'd pass config from sidebar
        st.write("🎯 Analysis Depth: Comprehensive")
        st.write("💰 Cost/Hour: AED 50")
        st.write("🔄 Annual Cycles: 50")
        st.write("💵 Budget: AED 50,000")
    
    @staticmethod
    def _display_empty_state():
        """Display empty state with examples"""
        st.info("👆 Upload process documents above to begin AI-powered analysis")
        
        with st.expander("📚 Sample Process Documents for Testing", expanded=False):
            FileHandler._show_sample_documents()
    
    @staticmethod
    def _show_sample_documents():
        """Show sample document examples"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📋 Sample: Faculty Promotion Process**
            
            ```
            Step 1: Performance evaluation review (24 hours)
            Step 2: Department head recommendation (16 hours)  
            Step 3: HR documentation preparation (20 hours)
            Step 4: Promotion committee assessment (32 hours)
            Step 5: Budget approval process (16 hours)
            Step 6: Administrative approval workflow (24 hours)
            Step 7: Document preparation and forms (28 hours)
            Step 8: Signature collection process (40 hours)
            Step 9: Government approval submission (48 hours)
            Step 10: Decree generation and system update (20 hours)
            Step 11: Employee notification and signing (16 hours)
            Step 12: Final documentation and archiving (12 hours)
            ```
            """)
        
        with col2:
            st.markdown("""
            **📋 Sample: Procurement Process**
            
            ```
            Step 1: Requirements identification (8 hours)
            Step 2: Budget verification (12 hours)
            Step 3: Vendor research and selection (24 hours)
            Step 4: Quote collection and analysis (32 hours)
            Step 5: Approval workflow (20 hours)
            Step 6: Purchase order generation (4 hours)
            Step 7: Vendor communication (8 hours)
            Step 8: Delivery coordination (16 hours)
            Step 9: Quality inspection (12 hours)
            Step 10: Invoice processing (8 hours)
            Step 11: Payment authorization (16 hours)
            Step 12: Documentation and closure (6 hours)
            ```
            """)
        
        st.info("💡 **Tip:** Copy any of these samples into a .txt file and upload to test the AI analysis capabilities!")

class AnalysisOrchestrator:
    """Orchestrate the analysis process with progress tracking"""
    
    @staticmethod
    def run_analysis(uploaded_files):
        """Run analysis with progress tracking"""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
                AnalysisOrchestrator._execute_analysis(uploaded_files)
    
    @staticmethod
    def _execute_analysis(uploaded_files):
        """Execute the analysis process"""
        st.session_state.analysis_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            optimizer = ProcessOptimizer()
            planner = ImplementationPlanner(optimizer.api_manager)
            doc_generator = ReportGenerator(planner)
            
            total_files = len(uploaded_files)
            
            for i, file in enumerate(uploaded_files):
                AnalysisOrchestrator._process_single_file(
                    file, i, total_files, progress_bar, status_text, 
                    optimizer, doc_generator
                )
            
            AnalysisOrchestrator._complete_analysis(progress_bar, status_text, total_files)
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.info("💡 Tip: Ensure your .env file contains a valid ANTHROPIC_API_KEY")
    
    @staticmethod
    def _process_single_file(file, index, total_files, progress_bar, status_text, optimizer, doc_generator):
        """Process a single file"""
        progress = (index + 1) / total_files
        progress_bar.progress(progress)
        status_text.text(f"Analyzing {file.name} ... ({index+1}/{total_files})")
        
        # Analyze document
        analysis_result = optimizer.analyze_document(file)
        
        # Generate reports
        status_text.text(f"Generating reports for {file.name}...")
        comprehensive_report = doc_generator.generate_comprehensive_report(analysis_result)
        advanced_svg = SVGGenerator.generate_advanced_svg_workflow(analysis_result)
        
        # Store results
        st.session_state.analysis_results.append({
            'analysis_result': analysis_result,
            'comprehensive_report': comprehensive_report,
            'advanced_svg': advanced_svg,
            'filename': file.name,
            'timestamp': datetime.now().isoformat()
        })
    
    @staticmethod
    def _complete_analysis(progress_bar, status_text, total_files):
        """Complete the analysis process"""
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")
        
        st.balloons()
        st.success(f"""
        🎉 **Analysis Complete!**
        - **{total_files} processes** analyzed
        - **AI-powered optimization** recommendations generated
        - **Implementation plans** created
        - **Financial impact** calculated
        """)
        
        time.sleep(2)
        st.rerun()

# ==================== RESULTS DISPLAY ====================

class ResultsDisplay:
    """Handle display of analysis results"""
    
    @staticmethod
    def display_results():
        """Display analysis results if available"""
        if not st.session_state.analysis_results:
            return
        
        st.header("AI Analysis Results")
        
        if len(st.session_state.analysis_results) > 1:
            ResultsDisplay._display_portfolio_overview()
        
        ResultsDisplay._display_individual_results()
    
    @staticmethod
    def _display_portfolio_overview():
        """Display portfolio overview dashboard"""
        st.subheader("📊 Portfolio Overview")
        
        metrics = PortfolioAnalyzer.calculate_portfolio_metrics(st.session_state.analysis_results)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Processes", len(st.session_state.analysis_results), "📋")
        
        with col2:
            avg_time_saved = metrics.get("average_time_saved", 0)
            st.metric("Avg Time Reduction", f"{avg_time_saved:.1f}%", "⚡")
        
        with col3:
            st.metric("Total Annual Savings", f"AED {metrics['total_savings']:,.0f}", "💰")
        
        with col4:
            st.metric("Implementation Cost", f"AED {metrics['implementation_cost']:,.0f}", "💵")
        
        with col5:
            st.metric("Portfolio ROI", f"{metrics['average_roi']:.1f}%", "📈")
        
        st.divider()
    
    @staticmethod
    def _display_individual_results():
        """Display individual process results in tabs"""
        tab_names = [
            f"📄 {result['analysis_result']['process_data']['process_name'][:25]}..." 
            for result in st.session_state.analysis_results
        ]
        tabs = st.tabs(tab_names)
        
        for tab, result in zip(tabs, st.session_state.analysis_results):
            with tab:
                ResultsDisplay._display_single_result(result)
    
    @staticmethod
    def _display_single_result(result):
        """Display a single analysis result"""
        analysis_result = result['analysis_result']
        process_data = analysis_result['process_data']
        optimization_data = analysis_result.get('optimization_data', {})
        opt_summary = optimization_data.get('optimization_summary', {})
        
        # Generate unique identifier for this process
        process_id = process_data['process_name'].replace(' ', '_').replace('-', '_').lower()
        
        # Only display metrics if optimization data exists
        if optimization_data and opt_summary:
            ResultsDisplay._display_metrics_header(process_data, opt_summary, optimization_data)
            ResultsDisplay._display_process_comparison(process_data, optimization_data, process_id)
        else:
            st.warning("⚠️ Optimization data not available for this analysis")
            st.json(process_data)  # Display raw process data as fallback
        
        ResultsDisplay._display_download_section(result, process_data, process_id)
    
    @staticmethod
    def _display_metrics_header(process_data, opt_summary, optimization_data):
        """Display key metrics header"""
        st.subheader(f"{process_data['process_name']} - AI Analysis")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Safely get values with defaults
        optimized_steps = optimization_data.get('optimized_steps', [])
        eliminated_steps = optimization_data.get('eliminated_steps', [])
        time_saved_percentage = opt_summary.get('time_saved_percentage', 0)
        total_time_before = opt_summary.get('total_time_before', 0)
        total_time_after = opt_summary.get('total_time_after', 0)
        estimated_annual_savings = opt_summary.get('estimated_annual_savings', 0)
        estimated_implementation_cost = opt_summary.get('estimated_implementation_cost', 0)
        payback_months = opt_summary.get('payback_months', 0)
        
        with col1:
            st.metric(
                "Steps Optimized",
                f"{len(process_data.get('steps', []))}→{len(optimized_steps)}",
                f"-{len(eliminated_steps)}"
            )
        
        with col2:
            st.metric(
                "Time Reduction",
                f"{time_saved_percentage:.1f}%",
                f"{total_time_before - total_time_after:.0f}h saved"
            )
        
        with col3:
            st.metric(
                "Annual Value",
                f"AED {estimated_annual_savings:,.0f}",
                "💰"
            )
        
        with col4:
            st.metric(
                "Implementation",
                f"AED {estimated_implementation_cost:,.0f}",
                f"{payback_months:.1f}m payback"
            )
        
        with col5:
            if estimated_implementation_cost > 0:
                roi_3_year = ((estimated_annual_savings * 3 - estimated_implementation_cost) 
                            / estimated_implementation_cost * 100)
            else:
                roi_3_year = 0
            st.metric(
                "3-Year ROI",
                f"{roi_3_year:.1f}%",
                "📈"
            )
        
        st.divider()
    
    @staticmethod
    def _display_process_comparison(process_data, optimization_data, process_id):
        """Display interactive process comparison"""
        st.subheader("🔄 Process Transformation")
        
        comparison_view = st.radio(
            "View Mode:",
            ["Side-by-Side", "Before Only", "After Only", "Optimization Summary"],
            horizontal=True,
            key=f"view_mode_{process_id}"
        )
        
        if comparison_view == "Side-by-Side":
            ResultsDisplay._display_side_by_side(process_data, optimization_data, process_id)
        elif comparison_view == "Before Only":
            st.markdown("#### 🔴 Current Process")
            ProcessStepRenderer.display_current_steps(process_data['steps'], optimization_data.get('eliminated_steps', []))
        elif comparison_view == "After Only":
            st.markdown("#### 🟢 Optimized Process")
            ProcessStepRenderer.display_optimized_steps(optimization_data.get('optimized_steps', []))
        elif comparison_view == "Optimization Summary":
            OptimizationSummaryRenderer.display_summary(optimization_data, process_id)
    
    @staticmethod
    def _display_side_by_side(process_data, optimization_data, process_id):
        """Display side-by-side comparison"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 Current Process")
            ProcessStepRenderer.display_current_steps(process_data['steps'], optimization_data.get('eliminated_steps', []))
        
        with col2:
            st.markdown("#### 🟢 Optimized Process")
            ProcessStepRenderer.display_optimized_steps(optimization_data.get('optimized_steps', []))
    
    @staticmethod
    def _display_download_section(result, process_data, process_id):
        """Display download section"""
        st.divider()
        st.subheader("⬇️ Download Analysis Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.download_button(
                label="📄 Comprehensive Report",
                data=result['comprehensive_report'].encode('utf-8'),
                file_name=f"{process_data['process_name']}_Analysis.md",
                mime="text/markdown",
                use_container_width=True,
                key=f"download_report_{process_id}"
            )
        
        with col2:
            st.download_button(
                label="🎨 Workflow Diagram",
                data=result['advanced_svg'].encode('utf-8'),
                file_name=f"{process_data['process_name']}_Workflow.svg",
                mime="image/svg+xml",
                use_container_width=True,
                key=f"download_svg_{process_id}"
            )
        
        with col3:
            export_data = {
                'process_analysis': process_data,
                'optimization_results': result['analysis_result']['optimization_data'],
                'metadata': {
                    'filename': result['filename'],
                    'analysis_timestamp': result['timestamp'],
                    'ai_powered': True
                }
            }
            
            st.download_button(
                label="📊 Raw Data (JSON)",
                data=json.dumps(export_data, indent=2).encode('utf-8'),
                file_name=f"{process_data['process_name']}_Data.json",
                mime="application/json",
                use_container_width=True,
                key=f"download_json_{process_id}"
            )
        
        with col4:
            st.button(
                "📈 Excel Report",
                disabled=True,
                help="Excel export coming soon!",
                use_container_width=True,
                key=f"excel_btn_{process_id}"
            )

class ProcessStepRenderer:
    """Render process steps with styling"""
    
    @staticmethod
    def display_current_steps(steps, eliminated_steps):
        """Display current process steps with elimination indicators"""
        for step in steps:
            is_eliminated = step['step_number'] in eliminated_steps
            
            if is_eliminated:
                st.markdown(f"""
                <div class="step-eliminated" style="padding: 10px; margin: 5px 0; border-radius: 5px;">
                    <strong>Step {step['step_number']}: {step['step_name']}</strong> 
                    <span style="background: #e74c3c; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">ELIMINATED</span><br>
                    <small>{step['description']}</small><br>
                    <small>⏱️ {step['duration_hours']}h | 👥 {step['responsible_unit']}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: white; padding: 10px; margin: 5px 0; border-left: 4px solid #3498db; border-radius: 5px;">
                    <strong>Step {step['step_number']}: {step['step_name']}</strong><br>
                    <small>{step['description']}</small><br>
                    <small>⏱️ {step['duration_hours']}h | 👥 {step['responsible_unit']}</small>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def display_optimized_steps(optimized_steps):
        """Display optimized process steps with optimization indicators"""
        optimization_colors = {
            'AUTOMATED': '#3498db',
            'MERGED': '#f39c12', 
            'PARALLEL': '#9b59b6',
            'DIGITAL': '#1abc9c',
            'ELIMINATED': '#e74c3c'
        }
        
        for step in optimized_steps:
            opt_type = step.get('optimization_type', 'OPTIMIZED')
            color = optimization_colors.get(opt_type, '#27ae60')
            
            # Safely get values with defaults
            step_number = step.get('step_number', 'N/A')
            step_name = step.get('step_name', 'Optimized Step')
            description = step.get('description', 'Step description not available')
            optimized_duration = step.get('optimized_duration', 0)
            original_duration = step.get('original_duration', optimized_duration)
            responsible_unit = step.get('responsible_unit', 'Not specified')
            optimization_rationale = step.get('optimization_rationale', 'Optimization details not available')
            
            st.markdown(f"""
            <div style="background: white; padding: 10px; margin: 5px 0; border-left: 4px solid {color}; border-radius: 5px;">
                <strong>Step {step_number}: {step_name}</strong> 
                <span style="background: {color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">{opt_type}</span><br>
                <small>{description}</small><br>
                <small>⏱️ {optimized_duration:.1f}h (was {original_duration}h) | 👥 {responsible_unit}</small><br>
                <small style="color: #666;">💡 {optimization_rationale}</small>
            </div>
            """, unsafe_allow_html=True)
            
            
            
            
            
            

class OptimizationSummaryRenderer:
    """Render optimization summary with charts"""
    
    @staticmethod
    def display_summary(optimization_data, process_id):
        """Display detailed optimization summary"""
        opt_summary = optimization_data['optimization_summary']
        
        col1, col2 = st.columns(2)
        
        with col1:
            OptimizationSummaryRenderer._display_statistics(optimization_data, process_id)
        
        with col2:
            OptimizationSummaryRenderer._display_benefits_and_risks(opt_summary, optimization_data)
        
        OptimizationSummaryRenderer._display_implementation_phases(opt_summary, process_id)
    
    @staticmethod
    def _display_statistics(optimization_data, process_id):
        """Display optimization statistics with charts"""
        st.markdown("#### Optimization Statistics")
        
        optimization_counts = {}
        for step in optimization_data.get('optimized_steps', []):
            opt_type = step.get('optimization_type', 'OPTIMIZED')
            optimization_counts[opt_type] = optimization_counts.get(opt_type, 0) + 1
        
        if optimization_counts:
            fig, ax = plt.subplots(figsize=(8, 6))
            strategies = list(optimization_counts.keys())
            counts = list(optimization_counts.values())
            colors = ['#3498db', '#f39c12', '#27ae60', '#9b59b6', '#e74c3c']
            
            bars = ax.bar(strategies, counts, color=colors[:len(strategies)])
            ax.set_title('Optimization Strategies Applied', fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of Steps')
            ax.set_xlabel('Optimization Type')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{int(height)}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Use st.pyplot without the key parameter
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No optimization data available for chart display")
    
    @staticmethod
    def _display_benefits_and_risks(opt_summary, optimization_data):
        """Display key benefits and implementation considerations"""
        st.markdown("#### Key Benefits")
        
        for benefit in opt_summary.get('key_benefits', []):
            st.write(f"✅ {benefit}")
        
        st.markdown("#### ⚠️ Implementation Considerations")
        
        high_risk_steps = [step for step in optimization_data.get('optimized_steps', []) 
                          if step.get('risk_level') == 'HIGH']
        
        if high_risk_steps:
            st.warning(f"⚠️ {len(high_risk_steps)} steps have high implementation risk")
        
        complex_steps = [step for step in optimization_data.get('optimized_steps', []) 
                        if step.get('implementation_complexity') == 'HIGH']
        
        if complex_steps:
            st.info(f"🔧 {len(complex_steps)} steps have high implementation complexity")
    
    @staticmethod
    def _display_implementation_phases(opt_summary, process_id):
        """Display implementation phases if available"""
        if 'implementation_phases' in opt_summary:
            st.markdown("#### Implementation Roadmap")
            
            phases_df = pd.DataFrame(opt_summary['implementation_phases'])
            st.dataframe(phases_df, use_container_width=True, key=f"phases_df_{process_id}")

# ==================== MAIN APPLICATION ====================

def main():
    """Enhanced main Streamlit application with modular architecture"""
    
    # Initialize application
    initialize_app()
    load_custom_css()
    
    # Render UI components
    UIComponents.render_header_with_status()
    config = UIComponents.render_sidebar_config()
    
    # Main content area
    st.header("📁 Document Upload & Analysis")
    
    # Handle file upload
    uploaded_files = FileHandler.handle_file_upload()
    
    if uploaded_files:
        st.divider()
        AnalysisOrchestrator.run_analysis(uploaded_files)
    
    # Display results
    ResultsDisplay.display_results()
    
    # Footer
    display_footer()

def display_footer():
    """Display application footer"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem; padding: 2rem;">
        🤖 <strong>AI Business Process Optimizer v2.0</strong><br/>
        Powered by Claude Sonnet 4 | Built with Streamlit | 
        <a href="mailto:support@processoptimizer.ai" style="color: #3498db;">Support</a> | 
        <a href="#" style="color: #3498db;">Documentation</a> | 
        <a href="#" style="color: #3498db;">GitHub</a><br/>
        <small>© 2024 Process Optimizer. Transform your business processes with AI.</small>
    </div>
    """, unsafe_allow_html=True)

# ==================== ERROR HANDLING & UTILITIES ====================

class ErrorHandler:
    """Centralized error handling utilities"""
    
    @staticmethod
    def handle_api_error(error: Exception, context: str = "API call"):
        """Handle API errors with user-friendly messages"""
        error_msg = str(error)
        
        if "authentication" in error_msg.lower():
            st.error("❌ Authentication failed. Please check your ANTHROPIC_API_KEY.")
        elif "rate limit" in error_msg.lower():
            st.error("⚠️ Rate limit exceeded. Please wait a moment and try again.")
        elif "timeout" in error_msg.lower():
            st.error("⏱️ Request timed out. Please try again.")
        else:
            st.error(f"❌ {context} failed: {error_msg}")
        
        st.info("💡 If the problem persists, please check your internet connection and API key.")
    
    @staticmethod
    def handle_file_error(error: Exception, filename: str):
        """Handle file processing errors"""
        st.error(f"❌ Error processing {filename}: {str(error)}")
        st.info("💡 Please ensure the file is not corrupted and is in a supported format.")
    
    @staticmethod
    def handle_json_error(error: Exception, context: str = "JSON parsing"):
        """Handle JSON parsing errors"""
        st.error(f"❌ {context} failed: Invalid response format")
        st.info("💡 The AI response was not in the expected format. Please try again.")

class ValidationUtils:
    """Utility functions for data validation"""
    
    @staticmethod
    def validate_process_data(process_data: Dict) -> bool:
        """Validate process data structure"""
        required_fields = ['process_name', 'process_description', 'steps']
        
        for field in required_fields:
            if field not in process_data:
                st.warning(f"Missing required field: {field}")
                return False
        
        if not isinstance(process_data['steps'], list) or len(process_data['steps']) == 0:
            st.warning("Process must have at least one step")
            return False
        
        return True
    
    @staticmethod
    def validate_optimization_data(optimization_data: Dict) -> bool:
        """Validate optimization data structure"""
        required_fields = ['optimized_steps', 'optimization_summary']
        
        for field in required_fields:
            if field not in optimization_data:
                st.warning(f"Missing required optimization field: {field}")
                return False
        
        return True
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe usage"""
        import re
        # Remove special characters and replace with underscores
        sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        return sanitized

class CacheManager:
    """Manage Streamlit caching for better performance"""
    
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def cache_analysis_result(file_content: str, filename: str) -> Dict:
        """Cache analysis results to avoid re-processing same files"""
        # This would be implemented with actual caching logic
        # For now, it's a placeholder for the caching mechanism
        return {}
    
    @staticmethod
    def clear_cache():
        """Clear all cached data"""
        st.cache_data.clear()
        st.success("Cache cleared successfully!")

class ConfigManager:
    """Manage application configuration"""
    
    @staticmethod
    def load_config() -> Dict:
        """Load application configuration"""
        default_config = {
            'max_files': 5,
            'supported_formats': ['pdf', 'txt', 'docx'],
            'max_file_size_mb': 10,
            'default_cost_per_hour': 50.0,
            'default_annual_cycles': 50,
            'default_implementation_budget': 50000.0
        }
        
        # In a real application, this could load from a config file
        return default_config
    
    @staticmethod
    def validate_config(config: Dict) -> bool:
        """Validate configuration parameters"""
        required_keys = ['max_files', 'supported_formats']
        return all(key in config for key in required_keys)

# ==================== PERFORMANCE MONITORING ====================

class PerformanceMonitor:
    """Monitor application performance and usage"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.metrics[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration"""
        if operation in self.metrics:
            duration = time.time() - self.metrics[operation]
            return duration
        return 0.0
    
    def log_performance(self, operation: str, duration: float):
        """Log performance metrics"""
        if duration > 30:  # Log slow operations (>30 seconds)
            st.info(f"⚠️ {operation} took {duration:.1f} seconds")

# ==================== EXPORT UTILITIES ====================

class ExportManager:
    """Handle various export formats and data preparation"""
    
    @staticmethod
    def prepare_excel_export(analysis_results: List[Dict]) -> bytes:
        """Prepare Excel export (placeholder for future implementation)"""
        # This would create an Excel file with multiple sheets
        # containing analysis results, charts, and summaries
        pass
    
    @staticmethod
    def prepare_powerpoint_export(analysis_results: List[Dict]) -> bytes:
        """Prepare PowerPoint export (placeholder for future implementation)"""
        # This would create a presentation with analysis results
        # and visualizations suitable for executive presentations
        pass
    
    @staticmethod
    def create_zip_package(analysis_results: List[Dict]) -> bytes:
        """Create a ZIP package with all analysis outputs"""
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, result in enumerate(analysis_results):
                process_name = result['analysis_result']['process_data']['process_name']
                safe_name = ValidationUtils.sanitize_filename(process_name)
                
                # Add comprehensive report
                zip_file.writestr(
                    f"{i+1:02d}_{safe_name}_Report.md",
                    result['comprehensive_report'].encode('utf-8')
                )
                
                # Add SVG workflow
                zip_file.writestr(
                    f"{i+1:02d}_{safe_name}_Workflow.svg",
                    result['advanced_svg'].encode('utf-8')
                )
                
                # Add JSON data
                export_data = {
                    'process_analysis': result['analysis_result']['process_data'],
                    'optimization_results': result['analysis_result']['optimization_data'],
                    'metadata': {
                        'filename': result['filename'],
                        'analysis_timestamp': result['timestamp'],
                        'ai_powered': True
                    }
                }
                
                zip_file.writestr(
                    f"{i+1:02d}_{safe_name}_Data.json",
                    json.dumps(export_data, indent=2).encode('utf-8')
                )
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()

# ==================== ANALYTICS & INSIGHTS ====================

class AnalyticsEngine:
    """Advanced analytics and insights generation"""
    
    @staticmethod
    def calculate_process_complexity_score(process_data: Dict) -> float:
        """Calculate a complexity score for the process"""
        steps = process_data.get('steps', [])
        
        complexity_factors = {
            'HIGH': 3,
            'MEDIUM': 2,
            'LOW': 1
        }
        
        total_complexity = sum(
            complexity_factors.get(step.get('complexity', 'MEDIUM'), 2) 
            for step in steps
        )
        
        # Normalize by number of steps
        if len(steps) > 0:
            return total_complexity / len(steps)
        return 0.0
    
    @staticmethod
    def identify_automation_candidates(process_data: Dict) -> List[Dict]:
        """Identify steps that are good candidates for automation"""
        steps = process_data.get('steps', [])
        
        automation_candidates = []
        for step in steps:
            automation_potential = step.get('automation_potential', 'LOW')
            complexity = step.get('complexity', 'HIGH')
            duration = step.get('duration_hours', 0)
            
            # High automation potential, low complexity, and significant duration
            if automation_potential == 'HIGH' and complexity == 'LOW' and duration >= 4:
                automation_candidates.append({
                    'step': step,
                    'priority': 'HIGH',
                    'reasoning': 'High automation potential with low complexity'
                })
            elif automation_potential == 'HIGH' and duration >= 8:
                automation_candidates.append({
                    'step': step,
                    'priority': 'MEDIUM',
                    'reasoning': 'High automation potential with significant time impact'
                })
        
        return automation_candidates
    
    @staticmethod
    def calculate_roi_projection(optimization_data: Dict, years: int = 5) -> Dict:
        """Calculate multi-year ROI projection"""
        opt_summary = optimization_data.get('optimization_summary', {})
        
        annual_savings = opt_summary.get('estimated_annual_savings', 0)
        implementation_cost = opt_summary.get('estimated_implementation_cost', 0)
        
        projection = {
            'years': [],
            'cumulative_savings': [],
            'cumulative_roi': [],
            'break_even_year': None
        }
        
        cumulative_savings = 0
        for year in range(1, years + 1):
            cumulative_savings += annual_savings
            net_benefit = cumulative_savings - implementation_cost
            roi = (net_benefit / implementation_cost * 100) if implementation_cost > 0 else 0
            
            projection['years'].append(year)
            projection['cumulative_savings'].append(cumulative_savings)
            projection['cumulative_roi'].append(roi)
            
            if net_benefit >= 0 and projection['break_even_year'] is None:
                projection['break_even_year'] = year
        
        return projection

# ==================== INTEGRATION HELPERS ====================

class IntegrationManager:
    """Manage integrations with external systems"""
    
    @staticmethod
    def export_to_project_management(analysis_results: List[Dict], platform: str = "generic"):
        """Export implementation tasks to project management tools"""
        # This would integrate with tools like Asana, Jira, Monday.com, etc.
        # For now, it's a placeholder that generates a structured task list
        
        tasks = []
        for result in analysis_results:
            process_name = result['analysis_result']['process_data']['process_name']
            opt_summary = result['analysis_result']['optimization_data']['optimization_summary']
            
            phases = opt_summary.get('implementation_phases', [])
            for phase in phases:
                tasks.append({
                    'project': f"Process Optimization: {process_name}",
                    'phase': phase.get('phase', 'Unknown Phase'),
                    'duration_months': phase.get('duration_months', 1),
                    'activities': phase.get('activities', []),
                    'cost': phase.get('cost', 0)
                })
        
        return tasks
    
    @staticmethod
    def generate_change_management_plan(analysis_results: List[Dict]) -> str:
        """Generate a comprehensive change management plan"""
        # This would create a detailed change management strategy
        # including stakeholder analysis, communication plan, training requirements, etc.
        
        total_processes = len(analysis_results)
        total_affected_employees = total_processes * 10  # Rough estimate
        
        plan = f"""
# Change Management Plan - Process Optimization Initiative

## Executive Summary
This change management plan supports the implementation of AI-optimized processes across {total_processes} business processes, potentially affecting {total_affected_employees} employees.

## Stakeholder Analysis
- **Primary Stakeholders**: Process owners, end users, IT department
- **Secondary Stakeholders**: Management, external vendors
- **Change Champions**: To be identified from each affected department

## Communication Strategy
1. **Announcement Phase**: Executive communication to all staff
2. **Education Phase**: Detailed workshops and training sessions
3. **Implementation Phase**: Regular updates and support
4. **Adoption Phase**: Feedback collection and continuous improvement

## Training Requirements
- Process-specific training for end users
- Technical training for IT support staff
- Change leadership training for managers

## Success Metrics
- Process adoption rate: Target 95% within 6 months
- User satisfaction: Target 4.5/5 rating
- Performance improvement: Target metrics per process
"""
        
        return plan

# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again. If the problem persists, contact support.")
        
        # Optional: Log error for debugging
        if st.checkbox("Show technical details"):
            st.exception(e)