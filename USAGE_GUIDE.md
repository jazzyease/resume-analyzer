# Resume Analyzer Usage Guide

This guide will help you set up and use the Resume Analyzer tool to compare resumes against job descriptions.

## Setup Instructions

### Prerequisites

- Python 3.11 or higher
- Pip (Python package installer)

### Installation

1. **Clone or download the repository**

2. **Create a virtual environment (recommended)**

   On Windows:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

   On macOS/Linux:
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

4. **Download required NLP models**
   ```
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   python -m spacy download en_core_web_md
   ```

## Using the Resume Analyzer

### Basic Usage

The basic command to analyze a resume against a job description is:

```
python src/main.py --resume path/to/resume.pdf --job_description path/to/jd.txt
```

For example:
```
python src/main.py --resume data/my_resume.pdf --job_description data/sample_job_description.txt
```

You can also use a PDF file for the job description:
```
python src/main.py --resume data/my_resume.pdf --job_description data/job_description.pdf
```

### Command-line Options

- `--resume` or `-r`: Path to the resume PDF file (required)
- `--job_description` or `-j`: Path to the job description file (PDF or text) (required)
- `--output` or `-o`: Path to save the results JSON file (optional)
- `--detailed` or `-d`: Print detailed analysis results (optional)

Example with all options:
```
python src/main.py --resume data/my_resume.pdf --job_description data/sample_job_description.txt --output results/my_analysis.json --detailed
```

## Understanding the Results

The analyzer will output:

1. **Overall Compatibility Score (0-100)**: How well the resume matches the job description
2. **Component Scores**:
   - Skills: Match between resume skills and job requirements
   - Experience: Match between resume experience and job requirements
   - Job Roles: Match between resume roles and job roles
   - Projects: Relevance of resume projects to the job
3. **Feedback**:
   - Strengths: Areas where the resume matches well
   - Gaps: Areas where the resume falls short
   - Suggestions: Recommendations to improve the match

## Example Output

```
================================================================================
RESUME ANALYZER RESULTS
================================================================================

Overall Compatibility Score: 75/100

Overall Assessment: Good match for the position. Some improvements could strengthen your application.

Component Scores:
- Skills: 80/100 (Weight: 40%)
- Experience: 70/100 (Weight: 30%)
- Job_roles: 85/100 (Weight: 20%)
- Projects: 60/100 (Weight: 10%)

Strengths:
✓ Strong skills match with the job requirements.
✓ Job roles align well with the position.
✓ 'E-commerce Platform' is particularly relevant to this position.

Gaps:
✗ Missing required skills: aws
✗ Experience gap of 2 years compared to the requirement.

Suggestions:
→ Consider acquiring or highlighting these skills: aws
→ Highlight relevant projects or transferable skills to compensate for the experience gap.
================================================================================
```

## Tips for Better Results

1. **Use PDF format for resumes**: The tool is optimized for PDF resumes.
2. **Job descriptions can be in PDF or text format**: The tool can handle both formats.
3. **Ensure clear formatting**: Well-structured resumes with clear sections yield better results.
4. **Be specific in your resume**: Explicitly mention skills, years of experience, and project details.
5. **Test with different job descriptions**: Compare your resume against various job descriptions to identify improvement areas.
6. **Update your skills list**: Make sure your resume includes all relevant skills, even those you consider obvious.

## Troubleshooting

### Common Issues

1. **PDF extraction fails**:
   - Ensure the PDF is not password-protected
   - Try converting to a different PDF format
   - Check if the PDF contains actual text (not just images)

2. **Missing dependencies**:
   - Run `pip install -r requirements.txt` again
   - Check for error messages during installation

3. **NLP models not found**:
   - Run the download commands again:
     ```
     python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
     python -m spacy download en_core_web_md
     ```

4. **Low scores despite relevant experience**:
   - Check if your resume uses different terminology than the job description
   - Ensure your resume explicitly mentions years of experience
   - Add more details about relevant projects

5. **Encoding errors with job description files**:
   - If you get encoding errors with text files, try saving them with UTF-8 encoding
   - For PDF job descriptions with extraction issues, try converting to text format

## Extending the Tool

The Resume Analyzer can be extended in several ways:

1. **Add more skills to the dictionary**: Edit the `common_skills` dictionary in `text_analyzer.py`
2. **Adjust scoring weights**: Modify the `weights` dictionary in `matcher.py`
3. **Add industry-specific analyzers**: Create specialized analyzers for different industries
4. **Implement a web interface**: Build a web UI for easier use

## Support

If you encounter any issues or have questions, please open an issue on the GitHub repository. 