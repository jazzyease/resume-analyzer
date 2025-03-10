"""
Test script to verify that profession mismatch detection works correctly.
"""

import logging
from src.matcher import Matcher

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_profession_mismatch():
    """Test that profession mismatch detection works correctly."""
    
    # Create a Matcher instance
    matcher = Matcher()
    
    # Test cases with different professions
    test_cases = [
        # Data Science resume vs Software Dev job
        {
            "name": "Data Science vs Software Dev",
            "resume_data": {
                "job_roles": ["data scientist", "machine learning engineer"],
                "skills": ["python", "machine learning", "tensorflow", "data analysis"],
                "experience": {"years": 2, "positions": ["data scientist"]},
                "projects": []
            },
            "jd_data": {
                "job_roles": ["software engineer", "backend developer"],
                "required_skills": ["java", "spring", "sql", "react"],
                "preferred_skills": ["docker", "kubernetes"],
                "required_experience": {"years": 5, "positions": ["software engineer"]}
            },
            "expected_mismatch": True
        },
        # Software Dev resume vs Software Dev job
        {
            "name": "Software Dev vs Software Dev",
            "resume_data": {
                "job_roles": ["software engineer", "backend developer"],
                "skills": ["java", "spring", "sql", "javascript"],
                "experience": {"years": 3, "positions": ["software engineer"]},
                "projects": []
            },
            "jd_data": {
                "job_roles": ["software engineer", "full stack developer"],
                "required_skills": ["java", "spring", "sql", "react"],
                "preferred_skills": ["docker", "kubernetes"],
                "required_experience": {"years": 5, "positions": ["software engineer"]}
            },
            "expected_mismatch": False
        },
        # Legal resume vs Marketing job
        {
            "name": "Legal vs Marketing",
            "resume_data": {
                "job_roles": ["lawyer", "attorney", "legal counsel"],
                "skills": ["legal research", "contract law", "litigation"],
                "experience": {"years": 7, "positions": ["attorney"]},
                "projects": []
            },
            "jd_data": {
                "job_roles": ["marketing manager", "content strategist"],
                "required_skills": ["seo", "content marketing", "social media"],
                "preferred_skills": ["adobe creative suite", "google analytics"],
                "required_experience": {"years": 3, "positions": ["marketing"]}
            },
            "expected_mismatch": True
        },
        # Data Science resume vs Data Science job
        {
            "name": "Data Science vs Data Science",
            "resume_data": {
                "job_roles": ["data scientist", "machine learning engineer"],
                "skills": ["python", "machine learning", "tensorflow", "data analysis"],
                "experience": {"years": 2, "positions": ["data scientist"]},
                "projects": []
            },
            "jd_data": {
                "job_roles": ["data scientist", "ml engineer"],
                "required_skills": ["python", "machine learning", "sql", "statistics"],
                "preferred_skills": ["tensorflow", "pytorch", "cloud"],
                "required_experience": {"years": 3, "positions": ["data scientist"]}
            },
            "expected_mismatch": False
        },
        # Junior resume vs Senior job (same profession but big experience gap)
        {
            "name": "Junior vs Senior (same profession)",
            "resume_data": {
                "job_roles": ["software engineer", "junior developer"],
                "skills": ["java", "javascript", "html", "css"],
                "experience": {"years": 1, "positions": ["junior developer"]},
                "projects": []
            },
            "jd_data": {
                "job_roles": ["senior software engineer", "tech lead"],
                "required_skills": ["java", "spring", "sql", "system design"],
                "preferred_skills": ["microservices", "kubernetes"],
                "required_experience": {"years": 8, "positions": ["senior software engineer"]}
            },
            "expected_mismatch": True
        }
    ]
    
    # Run tests
    results = []
    for test_case in test_cases:
        # Calculate match
        match_results = matcher.calculate_match(
            test_case["resume_data"], 
            test_case["jd_data"], 
            "Sample job description text"
        )
        
        # Check if profession mismatch detection is correct
        detected_mismatch = match_results.get("profession_mismatch", False)
        expected_mismatch = test_case["expected_mismatch"]
        
        result = detected_mismatch == expected_mismatch
        results.append((test_case["name"], result, detected_mismatch, expected_mismatch))
        
        # Print results
        print(f"\nTest: {test_case['name']}")
        print(f"Detected mismatch: {detected_mismatch}, Expected: {expected_mismatch}")
        print(f"Score: {match_results['overall_score']}/100")
        print(f"Result: {'PASSED' if result else 'FAILED'}")
    
    # Print summary
    print("\n" + "="*50)
    print("Test Results Summary")
    print("="*50)
    all_passed = True
    for name, result, detected, expected in results:
        status = "PASSED" if result else "FAILED"
        print(f"{name}: {status} (Detected: {detected}, Expected: {expected})")
        if not result:
            all_passed = False
    
    print(f"\nOverall test result: {'PASSED' if all_passed else 'FAILED'}")
    return 0 if all_passed else 1

if __name__ == "__main__":
    test_profession_mismatch() 