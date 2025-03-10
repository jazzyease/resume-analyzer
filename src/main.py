"""
Resume Analyzer Main Module

This is the main entry point for the Resume Analyzer application.
It handles command-line arguments, orchestrates the analysis process,
and outputs the results.
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
from typing import Dict, Any, Optional
import time

# Import our modules
from pdf_parser import PDFParser
from text_analyzer import TextAnalyzer
from matcher import Matcher

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class ResumeAnalyzer:
    """
    Main class for the Resume Analyzer application.
    """
    
    def __init__(self):
        """Initialize the Resume Analyzer with its component modules."""
        logger.info("Initializing Resume Analyzer")
        self.pdf_parser = PDFParser()
        self.text_analyzer = TextAnalyzer()
        self.matcher = Matcher()
    
    def analyze(self, resume_path: str, jd_path: str) -> Dict[str, Any]:
        """
        Analyze a resume against a job description.
        
        Args:
            resume_path (str): Path to the resume PDF file
            jd_path (str): Path to the job description file (can be PDF or text)
            
        Returns:
            Dict[str, Any]: Analysis results including match score and feedback
        """
        logger.info(f"Starting analysis of resume: {resume_path}")
        logger.info(f"Against job description: {jd_path}")
        
        # Step 1: Extract text from resume PDF
        resume_text = self.pdf_parser.extract_text(resume_path)
        if not resume_text:
            logger.error("Failed to extract text from resume")
            return {"error": "Failed to extract text from resume"}
        
        # Step 2: Read job description text
        try:
            # Check if job description is a PDF file
            if jd_path.lower().endswith('.pdf'):
                logger.info("Job description is in PDF format, extracting text...")
                jd_text = self.pdf_parser.extract_text(jd_path)
                if not jd_text:
                    logger.error("Failed to extract text from job description PDF")
                    return {"error": "Failed to extract text from job description PDF"}
            else:
                # Assume it's a text file
                with open(jd_path, 'r', encoding='utf-8') as file:
                    jd_text = file.read()
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            try:
                with open(jd_path, 'r', encoding='latin-1') as file:
                    jd_text = file.read()
                logger.info("Read job description with latin-1 encoding")
            except Exception as e:
                logger.error(f"Failed to read job description with alternative encoding: {e}")
                return {"error": f"Failed to read job description: {e}"}
        except Exception as e:
            logger.error(f"Failed to read job description: {e}")
            return {"error": f"Failed to read job description: {e}"}
        
        # Step 3: Analyze resume text
        resume_data = self.text_analyzer.analyze_resume(resume_text)
        
        # Step 4: Analyze job description text
        jd_data = self.text_analyzer.analyze_job_description(jd_text)
        
        # Step 5: Calculate match score
        try:
            match_results = self.matcher.calculate_match(resume_data, jd_data, jd_text)
        except Exception as e:
            logger.error(f"Error calculating match: {e}")
            return {"error": f"Error calculating match: {e}"}
        
        # Step 6: Prepare final results
        results = {
            "resume_path": resume_path,
            "jd_path": jd_path,
            "resume_data": resume_data,
            "jd_data": jd_data,
            "match_results": match_results
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any], detailed: bool = False) -> None:
        """
        Print analysis results to the console.
        
        Args:
            results (Dict[str, Any]): Analysis results
            detailed (bool): Whether to print detailed results
        """
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        match_results = results["match_results"]
        overall_score = match_results["overall_score"]
        feedback = match_results["feedback"]
        profession_mismatch = match_results.get("profession_mismatch", False)
        
        print("\n" + "="*80)
        print(f"RESUME ANALYZER RESULTS")
        print("="*80)
        
        print(f"\nOverall Compatibility Score: {overall_score}/100")
        
        if profession_mismatch:
            print("\n⚠️ PROFESSION MISMATCH DETECTED ⚠️")
            print("Your resume appears to be for a different profession than what this job requires.")
            print("The compatibility score has been adjusted accordingly.")
        
        print(f"\nOverall Assessment: {feedback['overall']}")
        
        print("\nComponent Scores:")
        for component, data in match_results["component_scores"].items():
            print(f"- {component.capitalize()}: {data['score']}/100 (Weight: {data['weight']*100}%)")
        
        print("\nStrengths:")
        if feedback["strengths"]:
            for strength in feedback["strengths"]:
                print(f"✓ {strength}")
        else:
            print("  No significant strengths identified.")
        
        print("\nGaps:")
        for gap in feedback["gaps"]:
            print(f"✗ {gap}")
        
        print("\nSuggestions:")
        for suggestion in feedback["suggestions"]:
            print(f"→ {suggestion}")
        
        if detailed:
            print("\n" + "="*80)
            print("DETAILED ANALYSIS")
            print("="*80)
            
            # Resume data
            resume_data = results["resume_data"]
            print("\nResume Skills:")
            for skill in resume_data["skills"]:
                print(f"- {skill}")
            
            print(f"\nResume Experience: {resume_data['experience']['years']} years")
            print("Resume Job Roles:")
            for role in resume_data["job_roles"]:
                print(f"- {role}")
            
            print("\nResume Projects:")
            for project in resume_data["projects"]:
                print(f"- {project['name']}")
                print(f"  Skills: {', '.join(project['skills'])}")
            
            # Job description data
            jd_data = results["jd_data"]
            print("\nRequired Skills:")
            for skill in jd_data["required_skills"]:
                print(f"- {skill}")
            
            print("\nPreferred Skills:")
            for skill in jd_data["preferred_skills"]:
                print(f"- {skill}")
            
            print(f"\nRequired Experience: {jd_data['required_experience']['years']} years")
            print("Job Roles:")
            for role in jd_data["job_roles"]:
                print(f"- {role}")
        
        print("\n" + "="*80)
    
    def save_results(self, results: Dict[str, Any], output_path: Optional[str] = None) -> None:
        """
        Save analysis results to a JSON file.
        
        Args:
            results (Dict[str, Any]): Analysis results
            output_path (Optional[str]): Path to save the results file
        """
        if "error" in results:
            logger.error(f"Cannot save results due to error: {results['error']}")
            return
        
        if not output_path:
            # Generate a default filename based on the resume filename
            resume_filename = os.path.basename(results["resume_path"])
            resume_name = os.path.splitext(resume_filename)[0]
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = f"{resume_name}_analysis_{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                # Use the custom encoder to handle numpy types
                json.dump(results, file, indent=2, cls=NumpyEncoder)
            logger.info(f"Results saved to {output_path}")
            print(f"\nResults saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            print(f"\nError: Failed to save results: {e}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze a resume against a job description")
    
    parser.add_argument("--resume", "-r", required=True, help="Path to the resume PDF file")
    parser.add_argument("--job_description", "-j", required=True, help="Path to the job description file (PDF or text)")
    parser.add_argument("--output", "-o", help="Path to save the results JSON file")
    parser.add_argument("--detailed", "-d", action="store_true", help="Print detailed analysis results")
    
    return parser.parse_args()


def main():
    """Main function to run the Resume Analyzer."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create the analyzer
    analyzer = ResumeAnalyzer()
    
    # Analyze the resume
    results = analyzer.analyze(args.resume, args.job_description)
    
    # Print results
    analyzer.print_results(results, detailed=args.detailed)
    
    # Save results if requested
    if args.output or "error" not in results:
        analyzer.save_results(results, args.output)


if __name__ == "__main__":
    main() 