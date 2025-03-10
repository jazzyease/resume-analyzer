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
from flask import Flask, render_template, request, redirect, url_for, flash, session

# Import resume analyzer components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parser.pdf_parser import PDFParser
from analyzer.text_analyzer import TextAnalyzer
from matcher.matcher import Matcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'resume-analyzer-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Initialize components
pdf_parser = PDFParser()
text_analyzer = TextAnalyzer()
matcher = Matcher()

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
        match_results = matcher.match(resume_data, jd_data)
        
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
    
    return render_template('results.html', results=session['results'])

if __name__ == '__main__':
    app.run(debug=True) 