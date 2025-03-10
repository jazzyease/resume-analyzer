"""
Test script to verify that the fixes for the UnboundLocalError work correctly.
"""

import os
import sys
import logging
from src.pdf_parser import PDFParser
from src.text_analyzer import TextAnalyzer
from src.matcher import Matcher

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_matcher():
    """Test the matcher with empty data to ensure no UnboundLocalError occurs."""
    logger.info("Testing matcher with empty data")
    
    # Create instances
    matcher = Matcher()
    
    # Test with empty data
    resume_data = {
        "skills": [],
        "experience": {"years": 0, "positions": []},
        "job_roles": [],
        "projects": []
    }
    
    jd_data = {
        "required_skills": [],
        "preferred_skills": [],
        "required_experience": {"years": 0, "positions": []},
        "job_roles": []
    }
    
    jd_text = ""
    
    try:
        # This should not raise UnboundLocalError
        match_results = matcher.calculate_match(resume_data, jd_data, jd_text)
        logger.info("Test passed: No UnboundLocalError with empty data")
        return True
    except UnboundLocalError as e:
        logger.error(f"Test failed: UnboundLocalError occurred: {e}")
        return False
    except Exception as e:
        logger.error(f"Test failed: Other exception occurred: {e}")
        return False

def test_with_minimal_data():
    """Test the matcher with minimal data to ensure no errors occur."""
    logger.info("Testing matcher with minimal data")
    
    # Create instances
    matcher = Matcher()
    
    # Test with minimal data
    resume_data = {
        "skills": ["python"],
        "experience": {"years": 1, "positions": ["developer"]},
        "job_roles": ["developer"],
        "projects": [{"name": "Test", "description": "Test project", "skills": ["python"]}]
    }
    
    jd_data = {
        "required_skills": ["python"],
        "preferred_skills": ["javascript"],
        "required_experience": {"years": 2, "positions": ["engineer"]},
        "job_roles": ["engineer"]
    }
    
    jd_text = "Looking for a Python developer with 2 years of experience."
    
    try:
        # This should not raise any errors
        match_results = matcher.calculate_match(resume_data, jd_data, jd_text)
        logger.info(f"Test passed: Match score = {match_results['overall_score']}")
        return True
    except Exception as e:
        logger.error(f"Test failed: Exception occurred: {e}")
        return False

def main():
    """Run all tests."""
    tests_passed = 0
    tests_total = 2
    
    if test_matcher():
        tests_passed += 1
    
    if test_with_minimal_data():
        tests_passed += 1
    
    logger.info(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        logger.info("All tests passed! The fixes are working correctly.")
        return 0
    else:
        logger.error("Some tests failed. The fixes may not be complete.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 