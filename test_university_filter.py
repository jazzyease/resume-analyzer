"""
Test script to verify that the university filtering works correctly.
"""

import logging
from src.text_analyzer import TextAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_university_filtering():
    """Test that university names are not identified as projects."""
    
    # Create a TextAnalyzer instance
    analyzer = TextAnalyzer()
    
    # Test text with university names
    test_text = """
    Education:
    Lovely Professional University, Punjab, India
    Bachelor of Technology in Computer Science, 2019-2023
    GPA: 3.8/4.0
    
    Projects:
    Data Analysis Dashboard - Created a dashboard using Python, Pandas, and Streamlit
    Machine Learning Model - Built a classification model using scikit-learn
    """
    
    # Extract projects
    projects = analyzer.extract_projects(test_text)
    
    # Print results
    print("\nExtracted Projects:")
    for i, project in enumerate(projects):
        print(f"{i+1}. {project['name']}")
        print(f"   Skills: {', '.join(project['skills'])}")
        print(f"   Is Data Science: {project['is_data_science']}")
    
    # Check if any project contains university name
    university_keywords = ['university', 'college', 'school', 'institute', 'academy', 'lpu', 'lovely professional']
    university_projects = [p for p in projects if any(kw in p['name'].lower() for kw in university_keywords)]
    
    if university_projects:
        print("\nFAILED: Found university names in projects:")
        for p in university_projects:
            print(f"- {p['name']}")
    else:
        print("\nPASSED: No university names found in projects")
    
    # Check if real projects were extracted
    expected_projects = ['data analysis dashboard', 'machine learning model']
    found_expected = all(any(exp in p['name'].lower() for p in projects) for exp in expected_projects)
    
    if found_expected:
        print("PASSED: Found expected projects")
    else:
        print("FAILED: Did not find all expected projects")
        print("Expected: ", expected_projects)
        print("Found: ", [p['name'] for p in projects])
    
    return len(university_projects) == 0 and found_expected

if __name__ == "__main__":
    success = test_university_filtering()
    print(f"\nOverall test result: {'PASSED' if success else 'FAILED'}") 