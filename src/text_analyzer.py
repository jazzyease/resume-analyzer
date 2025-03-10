"""
Text Analyzer Module

This module handles the extraction of key information from resume and job description texts
using Natural Language Processing (NLP) techniques.
"""

import re
import logging
import nltk
import spacy
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime
import pandas as pd
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    logger.warning("spaCy model not found. Please run: python -m spacy download en_core_web_md")
    # Use a smaller model as fallback
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Using smaller spaCy model as fallback")
    except OSError:
        logger.error("No spaCy model available. Text analysis will be limited.")
        nlp = None

class TextAnalyzer:
    """
    A class to analyze text from resumes and job descriptions using NLP.
    """
    
    def __init__(self):
        """Initialize the text analyzer with NLP models and resources."""
        logger.info("Initializing Text Analyzer")
        self.stop_words = set(stopwords.words('english'))
        
        # Common skills dictionary (can be expanded)
        self.common_skills = {
            'programming_languages': [
                'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 
                'kotlin', 'go', 'rust', 'typescript', 'scala', 'perl', 'r', 'matlab'
            ],
            'frameworks': [
                'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 
                'laravel', 'rails', 'asp.net', '.net', 'node.js', 'tensorflow', 
                'pytorch', 'pandas', 'numpy', 'scikit-learn'
            ],
            'databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'sqlite', 
                'redis', 'cassandra', 'dynamodb', 'firebase'
            ],
            'tools': [
                'git', 'docker', 'kubernetes', 'jenkins', 'aws', 'azure', 'gcp', 
                'jira', 'confluence', 'slack', 'figma', 'photoshop', 'illustrator'
            ],
            'soft_skills': [
                'communication', 'teamwork', 'leadership', 'problem solving', 
                'critical thinking', 'time management', 'creativity', 'adaptability'
            ],
            'chef_skills': [
                'cooking', 'baking', 'food preparation', 'menu planning', 'knife skills',
                'food safety', 'culinary', 'pastry', 'sous vide', 'catering'
            ],
            'legal_skills': [
                'litigation', 'contract law', 'legal research', 'legal writing',
                'negotiation', 'mediation', 'corporate law', 'intellectual property'
            ],
            'fashion_skills': [
                'fashion design', 'pattern making', 'sewing', 'sketching', 'textiles',
                'color theory', 'trend forecasting', 'draping', 'fashion merchandising'
            ],
            # Add more data science specific skills
            'data_science': [
                'machine learning', 'deep learning', 'neural networks', 'data mining',
                'data analysis', 'data visualization', 'statistical analysis', 'big data',
                'predictive modeling', 'natural language processing', 'nlp', 'computer vision',
                'feature engineering', 'data preprocessing', 'data cleaning', 'etl',
                'time series analysis', 'regression', 'classification', 'clustering',
                'dimensionality reduction', 'recommendation systems', 'a/b testing',
                'hypothesis testing', 'bayesian statistics', 'reinforcement learning',
                'supervised learning', 'unsupervised learning', 'semi-supervised learning'
            ],
            'data_science_tools': [
                'jupyter', 'tableau', 'power bi', 'excel', 'spss', 'sas', 'stata',
                'hadoop', 'spark', 'hive', 'pig', 'kafka', 'airflow', 'luigi',
                'matplotlib', 'seaborn', 'plotly', 'bokeh', 'd3.js', 'ggplot',
                'scikit-learn', 'sklearn', 'keras', 'tensorflow', 'pytorch', 'theano',
                'caffe', 'mxnet', 'xgboost', 'lightgbm', 'catboost', 'h2o',
                'dask', 'pyspark', 'databricks', 'snowflake', 'redshift', 'bigquery',
                'jupyter notebook', 'jupyter lab', 'colab', 'kaggle', 'dataiku',
                'rapidminer', 'knime', 'weka', 'orange', 'mlflow', 'kubeflow',
                'dvc', 'pachyderm', 'neptune', 'weights & biases', 'wandb',
                'streamlit', 'dash', 'flask', 'fastapi', 'shiny'
            ],
            'data_science_concepts': [
                'ai', 'artificial intelligence', 'ml', 'machine learning', 'dl', 'deep learning',
                'cnn', 'convolutional neural networks', 'rnn', 'recurrent neural networks',
                'lstm', 'transformers', 'bert', 'gpt', 'word2vec', 'glove', 'fasttext',
                'transfer learning', 'fine-tuning', 'hyperparameter tuning', 'grid search',
                'random search', 'bayesian optimization', 'cross-validation', 'k-fold',
                'train-test split', 'overfitting', 'underfitting', 'bias-variance tradeoff',
                'regularization', 'l1', 'l2', 'dropout', 'batch normalization',
                'gradient descent', 'sgd', 'adam', 'rmsprop', 'adagrad', 'adadelta',
                'precision', 'recall', 'f1-score', 'accuracy', 'roc', 'auc', 'confusion matrix',
                'precision-recall curve', 'silhouette score', 'elbow method', 'dbscan',
                'k-means', 'hierarchical clustering', 'pca', 't-sne', 'umap',
                'feature selection', 'feature importance', 'feature extraction',
                'data mining', 'data wrangling', 'data munging', 'data preprocessing',
                'data cleaning', 'data integration', 'data transformation', 'data reduction',
                'data discretization', 'data normalization', 'data standardization',
                'outlier detection', 'anomaly detection', 'missing value imputation'
            ]
        }
        
        # Flatten the skills list for easier searching
        self.all_skills = set()
        for category in self.common_skills.values():
            self.all_skills.update(category)
        
        # Initialize transformers pipeline for NER if spaCy is not available
        self.ner_pipeline = None
        if nlp is None:
            try:
                self.ner_pipeline = pipeline("ner")
                logger.info("Using transformers NER pipeline as fallback")
            except Exception as e:
                logger.error(f"Failed to initialize transformers NER pipeline: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing extra whitespace, converting to lowercase, etc.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep periods, commas, and parentheses
        text = re.sub(r'[^\w\s.,()@\-/]', ' ', text)
        
        # Remove extra whitespace
        text = text.strip()
        
        return text
    
    def extract_skills(self, text: str) -> Set[str]:
        """
        Extract skills from text using pattern matching and NLP.
        
        Args:
            text (str): Text to extract skills from
            
        Returns:
            Set[str]: Set of extracted skills
        """
        if not text:
            return set()
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Extract skills using pattern matching
        skills = set()
        
        # Direct skill mentions
        for skill in self.all_skills:
            # Look for the skill as a whole word
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, processed_text):
                skills.add(skill)
        
        # Extract skills from phrases like "experience with X" or "proficient in X"
        skill_phrases = [
            r'experience (?:with|in|using) ([\w\s]+)',
            r'proficient (?:with|in) ([\w\s]+)',
            r'knowledge of ([\w\s]+)',
            r'skilled (?:with|in) ([\w\s]+)',
            r'familiar with ([\w\s]+)',
            r'worked (?:with|on) ([\w\s]+)'
        ]
        
        for phrase in skill_phrases:
            matches = re.finditer(phrase, processed_text)
            for match in matches:
                potential_skill = match.group(1).strip()
                # Check if the extracted phrase contains any known skills
                for skill in self.all_skills:
                    if skill in potential_skill:
                        skills.add(skill)
        
        # Use spaCy for additional skill extraction if available
        if nlp:
            doc = nlp(processed_text)
            
            # Extract noun phrases as potential skills
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.lower()
                # Check if the noun phrase contains any known skills
                for skill in self.all_skills:
                    if skill in chunk_text and skill not in skills:
                        skills.add(skill)
        
        return skills
    
    def extract_job_roles(self, text: str) -> List[str]:
        """
        Extract job roles from text.
        
        Args:
            text (str): Text to extract job roles from
            
        Returns:
            List[str]: List of extracted job roles
        """
        if not text:
            return []
        
        # Common job titles by industry
        job_titles = {
            'tech': [
                'software engineer', 'developer', 'data scientist', 'product manager',
                'project manager', 'ux designer', 'ui designer', 'devops engineer',
                'system administrator', 'network engineer', 'database administrator',
                'qa engineer', 'tester', 'security engineer', 'data analyst',
                'machine learning engineer', 'frontend developer', 'backend developer',
                'full stack developer', 'mobile developer', 'ios developer', 'android developer'
            ],
            'finance': [
                'financial analyst', 'accountant', 'auditor', 'financial advisor',
                'investment banker', 'portfolio manager', 'risk analyst', 'trader',
                'financial manager', 'controller', 'actuary', 'underwriter'
            ],
            'healthcare': [
                'doctor', 'nurse', 'physician', 'surgeon', 'pharmacist', 'therapist',
                'medical assistant', 'healthcare administrator', 'clinical researcher',
                'radiologist', 'dentist', 'veterinarian', 'psychiatrist', 'psychologist'
            ],
            'culinary': [
                'chef', 'sous chef', 'pastry chef', 'head chef', 'executive chef',
                'line cook', 'prep cook', 'kitchen manager', 'food service manager',
                'baker', 'caterer', 'nutritionist', 'dietitian'
            ],
            'legal': [
                'lawyer', 'attorney', 'paralegal', 'legal assistant', 'judge',
                'legal counsel', 'compliance officer', 'contract specialist',
                'patent attorney', 'litigation attorney', 'corporate counsel'
            ],
            'fashion': [
                'fashion designer', 'fashion merchandiser', 'stylist', 'pattern maker',
                'textile designer', 'fashion buyer', 'fashion marketing manager',
                'fashion photographer', 'model', 'fashion journalist', 'costume designer'
            ],
            # Add more data science specific roles
            'data_science': [
                'data scientist', 'machine learning engineer', 'ml engineer', 'ai engineer',
                'data analyst', 'business analyst', 'data engineer', 'research scientist',
                'research engineer', 'applied scientist', 'ai researcher', 'ml researcher',
                'data science manager', 'head of data science', 'chief data scientist',
                'data science lead', 'data science consultant', 'analytics manager',
                'data architect', 'big data engineer', 'data science intern',
                'junior data scientist', 'senior data scientist', 'principal data scientist',
                'nlp engineer', 'nlp scientist', 'computer vision engineer',
                'computer vision scientist', 'deep learning engineer', 'deep learning scientist',
                'statistical analyst', 'quantitative analyst', 'quant', 'data mining engineer',
                'data visualization specialist', 'bi developer', 'bi analyst',
                'machine learning developer', 'ai product manager', 'data product manager',
                'data science educator', 'data science instructor', 'data science professor',
                'computational scientist', 'computational linguist', 'data journalist',
                'data storyteller', 'growth analyst', 'marketing analyst', 'revenue analyst',
                'data strategist', 'decision scientist', 'operations research scientist',
                'statistician', 'biostatistician', 'econometrician', 'data science fellow',
                'data science phd student', 'data science phd candidate', 'data science phd',
                'data science masters student', 'data science graduate student',
                'data science undergraduate', 'data science bootcamp student',
                'data science bootcamp graduate', 'data science bootcamp alumni'
            ]
        }
        
        # Flatten the job titles list
        all_job_titles = set()
        for industry_titles in job_titles.values():
            all_job_titles.update(industry_titles)
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Extract job roles
        roles = []
        
        # Direct job title mentions
        for title in all_job_titles:
            pattern = r'\b' + re.escape(title) + r'\b'
            if re.search(pattern, processed_text):
                roles.append(title)
        
        # Use spaCy for additional role extraction if available
        if nlp:
            doc = nlp(processed_text)
            
            # Look for job title patterns
            job_patterns = [
                r'(?:as a|as an|worked as|position as|role as|title of|position of|role of) ([\w\s]+)',
                r'(?:senior|junior|lead|chief|principal|associate|staff) ([\w\s]+)',
                r'(?:intern|internship|trainee) (?:as|at|in|with) ([\w\s]+)',
                r'(?:job title|position title|role title|title)(?:\s*:\s*|\s+is\s+|\s+-\s+|\s+–\s+) ([\w\s]+)'
            ]
            
            for pattern in job_patterns:
                matches = re.finditer(pattern, processed_text)
                for match in matches:
                    potential_role = match.group(1).strip()
                    # Check if not already found and not too long
                    if potential_role not in roles and len(potential_role.split()) <= 4:
                        # Check if the potential role contains any known job title words
                        for title in all_job_titles:
                            title_words = set(title.split())
                            potential_role_words = set(potential_role.split())
                            if title_words.intersection(potential_role_words):
                                roles.append(potential_role)
                                break
        
        # Check for data science specific role indicators
        data_science_indicators = [
            (r'\bdata\s+scien(?:ce|tist)\b', 'data scientist'),
            (r'\bmachine\s+learning\b', 'machine learning engineer'),
            (r'\bdeep\s+learning\b', 'deep learning engineer'),
            (r'\bai\b|\bartificial\s+intelligence\b', 'ai engineer'),
            (r'\bdata\s+analy(?:sis|tics|st)\b', 'data analyst'),
            (r'\bdata\s+engineer(?:ing)?\b', 'data engineer'),
            (r'\bresearch(?:er)?\b', 'research scientist'),
            (r'\bstatistic(?:s|ian|al)\b', 'statistician'),
            (r'\bquantitative\b|\bquant\b', 'quantitative analyst'),
            (r'\bcomputer\s+vision\b', 'computer vision engineer'),
            (r'\bnlp\b|\bnatural\s+language\s+processing\b', 'nlp engineer'),
            (r'\bdata\s+mining\b', 'data mining engineer'),
            (r'\bdata\s+visual(?:ization)?\b', 'data visualization specialist'),
            (r'\bbi\b|\bbusiness\s+intelligence\b', 'bi analyst')
        ]
        
        for pattern, role in data_science_indicators:
            if re.search(pattern, processed_text) and role not in roles:
                roles.append(role)
        
        # Check for student/education indicators to add student roles
        student_indicators = [
            (r'\bphd\b|\bdoctoral\b|\bdoctorate\b', 'data science phd student'),
            (r'\bmaster\'?s\b|\bmsc\b|\bms\b', 'data science masters student'),
            (r'\bundergraduate\b|\bbachelor\'?s\b|\bbs\b|\bba\b', 'data science undergraduate'),
            (r'\bbootcamp\b', 'data science bootcamp student'),
            (r'\bintern\b|\binternship\b', 'data science intern'),
            (r'\bstudent\b|\bstudy(?:ing)?\b', 'data science student')
        ]
        
        for pattern, role in student_indicators:
            if re.search(pattern, processed_text) and role not in roles:
                # Only add student roles if we also found data science indicators
                for ds_pattern, _ in data_science_indicators:
                    if re.search(ds_pattern, processed_text):
                        roles.append(role)
                        break
        
        return roles
    
    def extract_experience(self, text: str) -> Dict[str, any]:
        """
        Extract work experience details including years of experience.
        
        Args:
            text (str): Text to extract experience from
            
        Returns:
            Dict[str, any]: Dictionary with experience details
        """
        if not text:
            return {"years": 0, "positions": []}
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Check for student/education indicators that suggest no professional experience
        student_indicators = [
            r'\bstudent\b', r'\bundergraduate\b', r'\bgraduate\b', r'\bph\.?d\.? candidate\b',
            r'\bmaster\'?s student\b', r'\bcollege\b', r'\buniversity\b', r'\bcampus\b',
            r'\bfreshly graduated\b', r'\brecent graduate\b', r'\bfresh graduate\b',
            r'\bclass of \d{4}\b', r'\bexpected graduation\b', r'\bexpected to graduate\b',
            r'\bcurrently studying\b', r'\bcurrently enrolled\b', r'\bin progress\b',
            r'\bthesis\b', r'\bdissertation\b', r'\bacademic project\b', r'\bcourse project\b',
            r'\bsemester project\b', r'\bfinal year project\b', r'\bcapstone project\b'
        ]
        
        for indicator in student_indicators:
            if re.search(indicator, processed_text):
                # If strong student indicators are found and no clear professional experience,
                # assume 0 years of experience
                logger.info("Student/education indicators found, suggesting limited professional experience")
                
                # Still extract positions in case there are internships or part-time roles
                positions = self.extract_job_roles(text)
                return {"years": 0, "positions": positions}
        
        # Extract years of experience
        years_patterns = [
            r'(\d+)[\+]?\s*(?:years|yrs)(?:\s+of)?\s+experience',
            r'experience\s+(?:of|for)?\s+(\d+)[\+]?\s*(?:years|yrs)',
            r'(?:worked|working)(?:\s+for)?\s+(\d+)[\+]?\s*(?:years|yrs)'
        ]
        
        years_of_experience = 0
        for pattern in years_patterns:
            matches = re.finditer(pattern, processed_text)
            for match in matches:
                try:
                    years = int(match.group(1))
                    years_of_experience = max(years_of_experience, years)
                except (ValueError, IndexError):
                    continue
        
        # If no explicit years mentioned, try to calculate from date ranges
        if years_of_experience == 0:
            date_ranges = self._extract_date_ranges(processed_text)
            total_months = 0
            
            for start_date, end_date in date_ranges:
                if start_date and end_date:
                    # Calculate months between dates
                    months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                    total_months += months
            
            # Convert months to years
            if total_months > 0:
                years_of_experience = round(total_months / 12, 1)
        
        # Check for internship/entry-level indicators if no experience is found
        if years_of_experience == 0:
            internship_patterns = [
                r'\bintern\b', r'\binternship\b', r'\btrainee\b', r'\bentry[ -]level\b',
                r'\bjunior\b', r'\bassistant\b', r'\bfresh\b', r'\bnovice\b',
                r'\bbeginner\b', r'\bstarter\b', r'\bapprentice\b'
            ]
            
            for pattern in internship_patterns:
                if re.search(pattern, processed_text):
                    # If internship indicators are found, assume 0-1 years of experience
                    logger.info("Internship/entry-level indicators found")
                    years_of_experience = 0.5
                    break
        
        # Extract positions held
        positions = self.extract_job_roles(text)
        
        # If we found positions but no years, make an educated guess based on the number of positions
        if years_of_experience == 0 and positions:
            # Assume each position represents some experience, but be conservative
            estimated_years = min(len(positions) * 0.5, 2.0)
            logger.info(f"No explicit experience years found, estimating {estimated_years} years based on {len(positions)} positions")
            years_of_experience = estimated_years
        
        return {
            "years": years_of_experience,
            "positions": positions
        }
    
    def _extract_date_ranges(self, text: str) -> List[Tuple[Optional[datetime], Optional[datetime]]]:
        """
        Extract date ranges from text to calculate experience duration.
        
        Args:
            text (str): Text to extract date ranges from
            
        Returns:
            List[Tuple[Optional[datetime], Optional[datetime]]]: List of (start_date, end_date) tuples
        """
        # Common date formats in resumes
        date_patterns = [
            # MM/YYYY or MM-YYYY
            r'(\d{1,2}[/-]\d{4})',
            # Month YYYY (e.g., January 2020)
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
            # Abbreviated month YYYY (e.g., Jan 2020)
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})'
        ]
        
        # Find all dates in the text
        all_dates = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(0)
                try:
                    # Parse the date string to a datetime object
                    if '/' in date_str or '-' in date_str:
                        # MM/YYYY or MM-YYYY
                        parts = re.split(r'[/-]', date_str)
                        month = int(parts[0])
                        year = int(parts[1])
                        date = datetime(year, month, 1)
                    else:
                        # Month YYYY or abbreviated month YYYY
                        parts = date_str.split()
                        month_str = parts[0].lower()
                        year = int(parts[1])
                        
                        month_map = {
                            'january': 1, 'jan': 1,
                            'february': 2, 'feb': 2,
                            'march': 3, 'mar': 3,
                            'april': 4, 'apr': 4,
                            'may': 5,
                            'june': 6, 'jun': 6,
                            'july': 7, 'jul': 7,
                            'august': 8, 'aug': 8,
                            'september': 9, 'sep': 9,
                            'october': 10, 'oct': 10,
                            'november': 11, 'nov': 11,
                            'december': 12, 'dec': 12
                        }
                        
                        month = month_map.get(month_str.lower(), 1)
                        date = datetime(year, month, 1)
                    
                    all_dates.append((date, match.start()))
                except (ValueError, IndexError):
                    continue
        
        # Sort dates by position in text
        all_dates.sort(key=lambda x: x[1])
        
        # Group dates into ranges (start_date, end_date)
        date_ranges = []
        for i in range(0, len(all_dates) - 1, 2):
            start_date = all_dates[i][0]
            
            # Check if there's a next date
            if i + 1 < len(all_dates):
                end_date = all_dates[i + 1][0]
            else:
                # If no end date, use current date
                end_date = datetime.now()
            
            # Ensure start_date is before end_date
            if start_date < end_date:
                date_ranges.append((start_date, end_date))
        
        # Handle case where only one date is found (assume it's a start date)
        if len(all_dates) == 1:
            start_date = all_dates[0][0]
            end_date = datetime.now()
            date_ranges.append((start_date, end_date))
        
        return date_ranges
    
    def extract_projects(self, text: str) -> List[Dict[str, any]]:
        """
        Extract project information from text.
        
        Args:
            text (str): Text to extract projects from
            
        Returns:
            List[Dict[str, any]]: List of project dictionaries with name and description
        """
        if not text:
            return []
        
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        projects = []
        current_project = None
        
        # Project indicators
        project_indicators = [
            r'\bproject\b', r'\bdeveloped\b', r'\bbuilt\b', r'\bcreated\b',
            r'\bimplemented\b', r'\bdesigned\b', r'\bmanaged\b', r'\bled\b',
            r'\bauthored\b', r'\bpublished\b', r'\bresearched\b', r'\banalyzed\b'
        ]
        
        # Data science project indicators
        data_science_project_indicators = [
            r'\bdata\s+analysis\b', r'\bdata\s+visualization\b', r'\bdata\s+mining\b',
            r'\bmachine\s+learning\b', r'\bdeep\s+learning\b', r'\bneural\s+network\b',
            r'\bpredictive\s+model\b', r'\bclassification\b', r'\bregression\b',
            r'\bclustering\b', r'\bsegmentation\b', r'\brecommendation\b',
            r'\bnatural\s+language\s+processing\b', r'\bnlp\b', r'\bcomputer\s+vision\b',
            r'\bimage\s+recognition\b', r'\bspeech\s+recognition\b', r'\bsentiment\s+analysis\b',
            r'\btext\s+classification\b', r'\btopic\s+modeling\b', r'\bfeature\s+engineering\b',
            r'\bhyperparameter\s+tuning\b', r'\bcross-validation\b', r'\bmodel\s+evaluation\b',
            r'\bdata\s+preprocessing\b', r'\bdata\s+cleaning\b', r'\bdata\s+wrangling\b',
            r'\bexploratory\s+data\s+analysis\b', r'\beda\b', r'\bstatistical\s+analysis\b',
            r'\bhypothesis\s+testing\b', r'\ba/b\s+testing\b', r'\btime\s+series\b',
            r'\bforecasting\b', r'\banomaly\s+detection\b', r'\boutlier\s+detection\b',
            r'\bdimensionality\s+reduction\b', r'\bpca\b', r'\bt-sne\b', r'\bumap\b',
            r'\btransfer\s+learning\b', r'\bfine-tuning\b', r'\breinforcement\s+learning\b',
            r'\bsupervised\s+learning\b', r'\bunsupervised\s+learning\b', r'\bsemi-supervised\b',
            r'\bdeep\s+neural\s+network\b', r'\bcnn\b', r'\brnn\b', r'\blstm\b', r'\bgru\b',
            r'\btransformer\b', r'\bbert\b', r'\bgpt\b', r'\bword2vec\b', r'\bglove\b',
            r'\bfasttext\b', r'\bdoc2vec\b', r'\bsentence-bert\b', r'\buse\b',
            r'\bdata\s+pipeline\b', r'\betl\b', r'\bextract\s+transform\s+load\b',
            r'\bdata\s+warehouse\b', r'\bdata\s+lake\b', r'\bdata\s+mart\b',
            r'\bbig\s+data\b', r'\bhadoop\b', r'\bspark\b', r'\bhive\b', r'\bpig\b',
            r'\bkafka\b', r'\bstreaming\b', r'\bbatch\s+processing\b', r'\breal-time\b',
            r'\bdashboard\b', r'\bvisualization\b', r'\btableau\b', r'\bpower\s+bi\b',
            r'\bd3\.js\b', r'\bmatplotlib\b', r'\bseaborn\b', r'\bplotly\b', r'\bbokeh\b',
            r'\bggplot\b', r'\bdata\s+storytelling\b', r'\binfographic\b', r'\bcharts?\b',
            r'\bgraphs?\b', r'\bplots?\b', r'\bfigures?\b', r'\bdiagrams?\b', r'\bmaps?\b',
            r'\bheatmaps?\b', r'\bscatterplots?\b', r'\bhistograms?\b', r'\bbar\s+charts?\b',
            r'\bline\s+charts?\b', r'\bpie\s+charts?\b', r'\barea\s+charts?\b', r'\btreemaps?\b',
            r'\bchoropleth\b', r'\bcartogram\b', r'\bnetwork\s+graph\b', r'\bsankey\s+diagram\b',
            r'\bword\s+cloud\b', r'\btag\s+cloud\b', r'\bboxplot\b', r'\bviolin\s+plot\b',
            r'\bswarm\s+plot\b', r'\bpair\s+plot\b', r'\bjoint\s+plot\b', r'\bfacet\s+grid\b',
            r'\bfacet\s+wrap\b', r'\bcontour\s+plot\b', r'\bsurface\s+plot\b', r'\bgantt\s+chart\b',
            r'\btimeline\b', r'\bwaffle\s+chart\b', r'\bradial\s+chart\b', r'\bradar\s+chart\b',
            r'\bspider\s+chart\b', r'\bpolar\s+chart\b', r'\brose\s+chart\b', r'\bsunburst\s+chart\b',
            r'\bicicle\s+chart\b', r'\bparallel\s+coordinates\b', r'\bparallel\s+sets\b',
            r'\balluvial\s+diagram\b', r'\bcord\s+diagram\b', r'\barc\s+diagram\b',
            r'\bchord\s+diagram\b', r'\bdendrogram\b', r'\btree\s+diagram\b', r'\btree\s+map\b',
            r'\bvenn\s+diagram\b', r'\beuler\s+diagram\b', r'\bupset\s+plot\b', r'\bset\s+diagram\b'
        ]
        
        # Academic project indicators
        academic_project_indicators = [
            r'\bcourse\s+project\b', r'\bclass\s+project\b', r'\bacademic\s+project\b',
            r'\bsemester\s+project\b', r'\bfinal\s+year\s+project\b', r'\bcapstone\s+project\b',
            r'\bthesis\s+project\b', r'\bdissertation\s+project\b', r'\bresearch\s+project\b',
            r'\blab\s+project\b', r'\bgroup\s+project\b', r'\bteam\s+project\b',
            r'\bschool\s+project\b', r'\bcollege\s+project\b', r'\buniversity\s+project\b',
            r'\bgraduate\s+project\b', r'\bundergraduate\s+project\b', r'\bstudent\s+project\b',
            r'\bcoursework\b', r'\bassignment\b', r'\bcase\s+study\b', r'\bpracticum\b',
            r'\bfield\s+work\b', r'\bindependent\s+study\b', r'\bguided\s+research\b',
            r'\bsupervised\s+research\b', r'\bmentored\s+research\b', r'\badvised\s+research\b',
            r'\bfaculty\s+research\b', r'\bprofessor\s+research\b', r'\blab\s+research\b',
            r'\bresearch\s+assistant\b', r'\bresearch\s+associate\b', r'\bresearch\s+fellow\b',
            r'\bresearch\s+intern\b', r'\bresearch\s+scholar\b', r'\bresearch\s+student\b'
        ]
        
        # Educational institution indicators to filter out
        educational_institution_indicators = [
            r'\buniversity\b', r'\bcollege\b', r'\bschool\b', r'\binstitute\b', 
            r'\bacademy\b', r'\bpolytechnic\b', r'\bcampus\b', r'\balma\s+mater\b',
            r'\bgraduate\s+school\b', r'\bfaculty\b', r'\bdepartment\b', r'\bprogram\b',
            r'\bdegree\b', r'\bbachelor\b', r'\bmaster\b', r'\bphd\b', r'\bdoctorate\b',
            r'\bundergraduate\b', r'\bgraduate\b', r'\balumni\b', r'\bstudent\b',
            r'\bprofessor\b', r'\blecturer\b', r'\binstructor\b', r'\bteacher\b',
            r'\beducation\b', r'\bstudies\b', r'\blearning\b', r'\bteaching\b',
            r'\bcourse\b', r'\bclass\b', r'\bsemester\b', r'\bterm\b', r'\bacademic\b',
            r'\bscholastic\b', r'\bscholarly\b', r'\bintellectual\b', r'\berudite\b',
            r'\bprofessional\s+university\b', r'\btechnical\s+university\b', r'\bstate\s+university\b',
            r'\bnational\s+university\b', r'\binternational\s+university\b', r'\bprivate\s+university\b',
            r'\bpublic\s+university\b', r'\bcommunity\s+college\b', r'\bjunior\s+college\b',
            r'\btechnical\s+college\b', r'\bvocational\s+college\b', r'\blaw\s+school\b',
            r'\bmedical\s+school\b', r'\bbusiness\s+school\b', r'\bengineering\s+school\b',
            r'\bart\s+school\b', r'\bdesign\s+school\b', r'\bmusic\s+school\b', r'\bdrama\s+school\b',
            r'\bculinary\s+school\b', r'\bnursing\s+school\b', r'\bdental\s+school\b',
            r'\bpharmacy\s+school\b', r'\bveterinary\s+school\b', r'\barchitecture\s+school\b',
            r'\blovely\s+professional\s+university\b', r'\blpu\b'  # Add specific university names
        ]
        
        # Combine all indicators
        all_indicators = project_indicators + data_science_project_indicators + academic_project_indicators
        
        for sentence in sentences:
            # Check if sentence might be about a project
            is_project_sentence = False
            for indicator in all_indicators:
                if re.search(indicator, sentence.lower()):
                    is_project_sentence = True
                    break
            
            # Check if sentence is about an educational institution
            is_educational_institution = False
            for indicator in educational_institution_indicators:
                if re.search(indicator, sentence.lower()):
                    is_educational_institution = True
                    break
            
            # Skip if it's an educational institution and not clearly a project
            if is_educational_institution and not is_project_sentence:
                continue
            
            if is_project_sentence:
                # Try to extract project name
                project_name_patterns = [
                    r'project(?:\s+called|\s+named|\s+titled)?\s+"([^"]+)"',
                    r'project(?:\s+called|\s+named|\s+titled)?\s+\'([^\']+)\'',
                    r'project(?:\s+called|\s+named|\s+titled)?\s+([A-Z][a-zA-Z0-9]+)',
                    r'(?:developed|built|created|implemented|designed)\s+(?:a|an)\s+([^,.]+)',
                    r'(?:thesis|dissertation|capstone|research)\s+(?:on|about|titled)\s+([^,.]+)',
                    r'(?:project|work)\s+(?:on|for)\s+([^,.]+)'
                ]
                
                project_name = None
                for pattern in project_name_patterns:
                    match = re.search(pattern, sentence.lower())
                    if match:
                        project_name = match.group(1).strip()
                        break
                
                if not project_name:
                    # If no name found, use first few words as name
                    words = sentence.split()
                    if len(words) > 3:
                        project_name = " ".join(words[:3]) + "..."
                    else:
                        project_name = sentence
                
                # Skip if the project name is likely an educational institution
                if any(re.search(r'\b' + re.escape(edu_word) + r'\b', project_name.lower()) 
                       for edu_word in ['university', 'college', 'school', 'institute', 'academy', 'lpu']):
                    continue
                
                # Extract skills used in the project
                project_skills = self.extract_skills(sentence)
                
                # Determine if it's a data science project
                is_data_science_project = False
                for indicator in data_science_project_indicators:
                    if re.search(indicator, sentence.lower()):
                        is_data_science_project = True
                        break
                
                # If it's a data science project, add data science as a skill if not already present
                if is_data_science_project and 'data science' not in project_skills:
                    project_skills.add('data science')
                
                # Create project entry
                current_project = {
                    "name": project_name,
                    "description": sentence,
                    "skills": list(project_skills),
                    "is_data_science": is_data_science_project
                }
                
                projects.append(current_project)
            elif current_project:
                # If this sentence might be a continuation of the previous project description
                current_project["description"] += " " + sentence
                
                # Update skills
                additional_skills = self.extract_skills(sentence)
                current_project["skills"] = list(set(current_project["skills"]).union(additional_skills))
                
                # Check if this sentence contains data science indicators
                if not current_project.get("is_data_science"):
                    for indicator in data_science_project_indicators:
                        if re.search(indicator, sentence.lower()):
                            current_project["is_data_science"] = True
                            # Add data science as a skill if not already present
                            if 'data science' not in current_project["skills"]:
                                current_project["skills"].append('data science')
                            break
        
        # If no projects were found using the above method, try to find project sections
        if not projects:
            # Look for project sections
            project_section_patterns = [
                r'(?:projects?|portfolio|work)(?:\s+experience)?(?:\s*:|\s*-|\s*–|\n)([\s\S]+?)(?=\n\s*(?:[A-Z][A-Za-z\s]+:|\Z))',
                r'(?:PROJECTS?|PORTFOLIO|WORK)(?:\s+EXPERIENCE)?(?:\s*:|\s*-|\s*–|\n)([\s\S]+?)(?=\n\s*(?:[A-Z][A-Z\s]+:|\Z))'
            ]
            
            for pattern in project_section_patterns:
                match = re.search(pattern, text)
                if match:
                    project_section = match.group(1).strip()
                    # Split the project section into individual projects
                    project_items = re.split(r'\n\s*[-•●■◆▪▫]\s+', project_section)
                    
                    for item in project_items:
                        if item.strip():
                            # Extract project name (first line or first few words)
                            lines = item.strip().split('\n')
                            if lines:
                                project_name = lines[0].strip()
                                # If the name is too long, use just the first few words
                                if len(project_name.split()) > 5:
                                    project_name = " ".join(project_name.split()[:5]) + "..."
                                
                                # Skip if the project name is likely an educational institution
                                if any(re.search(r'\b' + re.escape(edu_word) + r'\b', project_name.lower()) 
                                       for edu_word in ['university', 'college', 'school', 'institute', 'academy', 'lpu']):
                                    continue
                                
                                # Extract skills
                                project_skills = self.extract_skills(item)
                                
                                # Determine if it's a data science project
                                is_data_science_project = False
                                for indicator in data_science_project_indicators:
                                    if re.search(indicator, item.lower()):
                                        is_data_science_project = True
                                        break
                                
                                # If it's a data science project, add data science as a skill if not already present
                                if is_data_science_project and 'data science' not in project_skills:
                                    project_skills.add('data science')
                                
                                projects.append({
                                    "name": project_name,
                                    "description": item,
                                    "skills": list(project_skills),
                                    "is_data_science": is_data_science_project
                                })
        
        # Final filter to remove any educational institutions that might have slipped through
        filtered_projects = []
        for project in projects:
            # Skip if the project name contains educational institution keywords
            if any(re.search(r'\b' + re.escape(edu_word) + r'\b', project["name"].lower()) 
                   for edu_word in ['university', 'college', 'school', 'institute', 'academy', 'lpu', 'lovely professional']):
                continue
            
            # Skip if the project doesn't have any skills (likely not a real project)
            if not project["skills"]:
                continue
            
            filtered_projects.append(project)
        
        return filtered_projects
    
    def analyze_resume(self, resume_text: str) -> Dict[str, any]:
        """
        Analyze resume text to extract all relevant information.
        
        Args:
            resume_text (str): Resume text to analyze
            
        Returns:
            Dict[str, any]: Dictionary with extracted information
        """
        if not resume_text:
            return {
                "skills": [],
                "experience": {"years": 0, "positions": []},
                "job_roles": [],
                "projects": []
            }
        
        # Extract all information
        skills = self.extract_skills(resume_text)
        experience = self.extract_experience(resume_text)
        job_roles = self.extract_job_roles(resume_text)
        projects = self.extract_projects(resume_text)
        
        return {
            "skills": list(skills),
            "experience": experience,
            "job_roles": job_roles,
            "projects": projects
        }
    
    def analyze_job_description(self, jd_text: str) -> Dict[str, any]:
        """
        Analyze job description text to extract all relevant information.
        
        Args:
            jd_text (str): Job description text to analyze
            
        Returns:
            Dict[str, any]: Dictionary with extracted information
        """
        if not jd_text:
            return {
                "required_skills": [],
                "preferred_skills": [],
                "required_experience": {"years": 0, "positions": []},
                "job_roles": []
            }
        
        # Split text into sections
        sections = self._split_into_sections(jd_text)
        
        # Extract required skills
        required_skills = set()
        preferred_skills = set()
        
        # Look for required and preferred skills sections
        for section_title, section_text in sections:
            if any(keyword in section_title.lower() for keyword in ['requirement', 'qualification', 'skill', 'must have']):
                required_skills.update(self.extract_skills(section_text))
            elif any(keyword in section_title.lower() for keyword in ['preferred', 'nice to have', 'plus', 'bonus']):
                preferred_skills.update(self.extract_skills(section_text))
        
        # If no clear sections, extract from full text
        if not required_skills:
            # Look for required skills patterns
            required_patterns = [
                r'required skills[:\s]+([\s\S]+?)(?=\n\n|\Z)',
                r'requirements[:\s]+([\s\S]+?)(?=\n\n|\Z)',
                r'qualifications[:\s]+([\s\S]+?)(?=\n\n|\Z)',
                r'must have[:\s]+([\s\S]+?)(?=\n\n|\Z)'
            ]
            
            for pattern in required_patterns:
                match = re.search(pattern, jd_text, re.IGNORECASE)
                if match:
                    required_skills.update(self.extract_skills(match.group(1)))
            
            # If still no required skills, extract from full text
            if not required_skills:
                required_skills.update(self.extract_skills(jd_text))
        
        # Extract experience requirements
        experience_req = {"years": 0, "positions": []}
        
        # Look for experience patterns
        exp_patterns = [
            r'(\d+)[\+]?\s*(?:years|yrs)(?:\s+of)?\s+experience',
            r'experience\s+(?:of|for)?\s+(\d+)[\+]?\s*(?:years|yrs)',
            r'minimum\s+(?:of)?\s+(\d+)[\+]?\s*(?:years|yrs)'
        ]
        
        for pattern in exp_patterns:
            matches = re.finditer(pattern, jd_text.lower())
            for match in matches:
                try:
                    years = int(match.group(1))
                    experience_req["years"] = max(experience_req["years"], years)
                except (ValueError, IndexError):
                    continue
        
        # Extract job roles
        job_roles = self.extract_job_roles(jd_text)
        
        return {
            "required_skills": list(required_skills),
            "preferred_skills": list(preferred_skills),
            "required_experience": experience_req,
            "job_roles": job_roles
        }
    
    def _split_into_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text into sections based on headings.
        
        Args:
            text (str): Text to split into sections
            
        Returns:
            List[Tuple[str, str]]: List of (section_title, section_content) tuples
        """
        if not text:
            return []
        
        # Common section heading patterns
        heading_patterns = [
            # All caps headings
            r'^([A-Z][A-Z\s]+[A-Z])[\s:]*$',
            # Title case headings with colon
            r'^([A-Z][a-zA-Z\s]+):',
            # Numbered or bulleted headings
            r'^(\d+\.\s+[A-Z][a-zA-Z\s]+)[\s:]*$',
            r'^(•\s+[A-Z][a-zA-Z\s]+)[\s:]*$'
        ]
        
        # Split text into lines
        lines = text.split('\n')
        
        sections = []
        current_section = ("", "")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a heading
            is_heading = False
            for pattern in heading_patterns:
                match = re.match(pattern, line)
                if match:
                    # If we have content in the current section, add it to sections
                    if current_section[1]:
                        sections.append(current_section)
                    
                    # Start a new section
                    current_section = (match.group(1), "")
                    is_heading = True
                    break
            
            if not is_heading:
                # Add line to current section content
                current_section = (current_section[0], current_section[1] + " " + line)
        
        # Add the last section
        if current_section[1]:
            sections.append(current_section)
        
        # If no sections were found, use the entire text as one section
        if not sections:
            sections = [("Text", text)]
        
        return sections


# Example usage
if __name__ == "__main__":
    analyzer = TextAnalyzer()
    
    # Example resume text
    sample_resume = """
    John Doe
    Software Engineer
    
    EXPERIENCE
    Senior Software Engineer, ABC Tech (2018-2022)
    - Developed scalable web applications using React and Node.js
    - Led a team of 5 developers on a project that increased user engagement by 30%
    - Implemented CI/CD pipelines using Jenkins and Docker
    
    Software Developer, XYZ Corp (2015-2018)
    - Built RESTful APIs using Python and Flask
    - Worked with PostgreSQL and MongoDB databases
    - Collaborated with UX designers to improve user experience
    
    SKILLS
    Programming: Python, JavaScript, Java, SQL
    Frameworks: React, Node.js, Flask, Django
    Tools: Git, Docker, Jenkins, AWS
    
    PROJECTS
    E-commerce Platform
    - Built a full-stack e-commerce platform using MERN stack
    - Implemented payment processing with Stripe API
    - Deployed on AWS using Kubernetes
    """
    
    # Example job description
    sample_jd = """
    Senior Full Stack Developer
    
    REQUIREMENTS:
    - 5+ years of experience in web development
    - Strong proficiency in JavaScript, React, and Node.js
    - Experience with database design and management (SQL and NoSQL)
    - Knowledge of CI/CD pipelines and containerization
    
    PREFERRED SKILLS:
    - Experience with AWS or other cloud platforms
    - Knowledge of TypeScript
    - Experience with microservices architecture
    
    RESPONSIBILITIES:
    - Develop and maintain web applications
    - Collaborate with cross-functional teams
    - Mentor junior developers
    - Participate in code reviews
    """
    
    # Analyze resume
    resume_analysis = analyzer.analyze_resume(sample_resume)
    print("Resume Analysis:")
    print(f"Skills: {resume_analysis['skills']}")
    print(f"Experience: {resume_analysis['experience']}")
    print(f"Job Roles: {resume_analysis['job_roles']}")
    print(f"Projects: {len(resume_analysis['projects'])}")
    
    # Analyze job description
    jd_analysis = analyzer.analyze_job_description(sample_jd)
    print("\nJob Description Analysis:")
    print(f"Required Skills: {jd_analysis['required_skills']}")
    print(f"Preferred Skills: {jd_analysis['preferred_skills']}")
    print(f"Required Experience: {jd_analysis['required_experience']}")
    print(f"Job Roles: {jd_analysis['job_roles']}") 