# Resume Analyzer

A Python tool that analyzes resumes against job descriptions to calculate a compatibility score.

## Features

- Parse resumes in PDF format
- Extract key information (job roles, experience, skills, projects)
- Compare resume data with job description requirements
- Generate a compatibility score (0-100)
- Provide detailed feedback on matches and gaps

## Project Structure

```
resume_analyzer/
├── data/              # Sample resumes and job descriptions
├── models/            # Saved models and data files
├── src/               # Source code
│   ├── __init__.py
│   ├── pdf_parser.py  # PDF extraction functionality
│   ├── text_analyzer.py # NLP and text analysis
│   ├── matcher.py     # Resume-JD matching logic
│   └── main.py        # Main application entry point
└── tests/             # Test files
```

## Setup Instructions

### Prerequisites

- Python 3.11 or higher

### Installation

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```
     source venv/bin/activate
     ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download required NLTK data:
   ```
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

5. Download required spaCy model:
   ```
   python -m spacy download en_core_web_md
   ```

## Usage

Run the analyzer with:
```
python src/main.py --resume path/to/resume.pdf --job_description path/to/jd.txt
```

## License

MIT 