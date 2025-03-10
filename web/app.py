#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flask web application for the Resume Analyzer.
This app provides a web interface for analyzing resumes against job descriptions.
"""

import os
import json
import logging
import tempfile
import gc
import nltk
from flask import Flask, render_template, request, redirect, url_for, flash, session

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Import resume analyzer components
from src.pdf_parser import PDFParser
from src.text_analyzer import TextAnalyzer
from src.matcher import Matcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert NumPy types to Python standard types for JSON serialization."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Initialize Flask app
app = Flask(__name__, 
            template_folder='web/templates',
            static_folder='web/static')
app.secret_key = os.environ.get('SECRET_KEY', 'resume-analyzer-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Initialize components lazily to save memory
pdf_parser = None
text_analyzer = None
matcher = None

def get_pdf_parser():
    global pdf_parser
    if pdf_parser is None:
        pdf_parser = PDFParser()
    return pdf_parser

def get_text_analyzer():
    global text_analyzer
    if text_analyzer is None:
        text_analyzer = TextAnalyzer()
    return text_analyzer

def get_matcher():
    global matcher
    if matcher is None:
        matcher = Matcher()
    return matcher

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'resume': {'pdf'},
    'job_description': {'pdf', 'txt'}
}

def allowed_file(filename, file_type):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]

@app.route('/')
def index():
    """Render the home page."""
    # Clear any previous session data to free memory
    if 'results' in session:
        del session['results']
        session.modified = True
    
    # Force garbage collection
    gc.collect()
    
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle file uploads and analyze the resume against the job description."""
    # Check if files were uploaded
    if 'resume' not in request.files or 'job_description' not in request.files:
        flash('Please upload both a resume and a job description.', 'error')
        return redirect(url_for('index'))
    
    resume_file = request.files['resume']
    job_description_file = request.files['job_description']
    
    # Check if files are empty
    if resume_file.filename == '' or job_description_file.filename == '':
        flash('Please select both a resume and a job description file.', 'error')
        return redirect(url_for('index'))
    
    # Check if files have allowed extensions
    if not allowed_file(resume_file.filename, 'resume'):
        flash('Resume must be a PDF file.', 'error')
        return redirect(url_for('index'))
    
    if not allowed_file(job_description_file.filename, 'job_description'):
        flash('Job description must be a PDF or TXT file.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save uploaded files to temporary directory
        resume_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resume.pdf')
        resume_file.save(resume_path)
        
        job_description_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                           f"job_description.{job_description_file.filename.rsplit('.', 1)[1].lower()}")
        job_description_file.save(job_description_path)
        
        # Get components lazily
        pdf_parser = get_pdf_parser()
        text_analyzer = get_text_analyzer()
        matcher = get_matcher()
        
        # Extract text from resume
        resume_text = pdf_parser.extract_text(resume_path)
        
        # Extract text from job description
        if job_description_path.endswith('.pdf'):
            job_description_text = pdf_parser.extract_text(job_description_path)
        else:
            with open(job_description_path, 'r', encoding='utf-8') as f:
                job_description_text = f.read()
        
        # Analyze resume
        resume_data = text_analyzer.analyze_resume(resume_text)
        
        # Analyze job description
        jd_data = text_analyzer.analyze_job_description(job_description_text)
        
        # Match resume against job description
        match_results = matcher.calculate_match(resume_data, jd_data, job_description_text)
        
        # Log the structure of match_results for debugging
        logger.info(f"Match results structure: {list(match_results.keys())}")
        if 'component_scores' in match_results:
            logger.info(f"Component scores: {list(match_results['component_scores'].keys())}")
        if 'feedback' in match_results:
            logger.info(f"Feedback structure: {list(match_results['feedback'].keys())}")
        
        # Ensure feedback exists
        if 'feedback' not in match_results:
            match_results['feedback'] = {
                'overall': 'Analysis completed. No detailed feedback available.',
                'strengths': [],
                'gaps': [],
                'suggestions': []
            }
        
        # Convert NumPy types to standard Python types for JSON serialization
        match_results = convert_numpy_types(match_results)
        resume_data = convert_numpy_types(resume_data)
        jd_data = convert_numpy_types(jd_data)
        
        # Store results in session
        session['results'] = {
            'resume_data': resume_data,
            'jd_data': jd_data,
            'match_results': match_results
        }
        
        # Clean up temporary files
        try:
            os.remove(resume_path)
            os.remove(job_description_path)
        except Exception as e:
            logger.warning(f"Error removing temporary files: {e}")
        
        # Force garbage collection
        gc.collect()
        
        return redirect(url_for('results'))
    
    except Exception as e:
        logger.error(f"Error processing files: {e}", exc_info=True)
        flash(f"An error occurred while processing your files: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    """Display the analysis results."""
    if 'results' not in session:
        flash('No analysis results found. Please upload files to analyze.', 'error')
        return redirect(url_for('index'))
    
    # Log the structure of the results for debugging
    results = session['results']
    logger.info(f"Results keys: {list(results.keys())}")
    if 'match_results' in results:
        logger.info(f"Match results keys: {list(results['match_results'].keys())}")
    
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
