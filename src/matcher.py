"""
Matcher Module

This module compares resume data with job description requirements
and calculates a compatibility score.
"""

import logging
import spacy
from typing import Dict, List, Any, Tuple, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy model for semantic similarity
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    logger.warning("spaCy model not found. Using smaller model as fallback.")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.error("No spaCy model available. Semantic matching will be limited.")
        nlp = None

class Matcher:
    """
    A class to match resume data with job description requirements
    and calculate a compatibility score.
    """
    
    def __init__(self):
        """Initialize the matcher."""
        logger.info("Initializing Matcher")
        
        # Define weights for different matching components
        self.weights = {
            "skills": 0.4,
            "experience": 0.3,
            "job_roles": 0.2,
            "projects": 0.1
        }
        
        # Define data science specific skill categories for better matching
        self.data_science_skill_categories = {
            "core_ml": [
                "machine learning", "deep learning", "neural networks", "artificial intelligence",
                "ai", "ml", "dl", "supervised learning", "unsupervised learning", 
                "reinforcement learning", "transfer learning"
            ],
            "data_processing": [
                "data preprocessing", "data cleaning", "data wrangling", "data mining",
                "etl", "extract transform load", "data integration", "data transformation"
            ],
            "statistics": [
                "statistical analysis", "hypothesis testing", "a/b testing", "bayesian statistics",
                "regression", "classification", "clustering", "dimensionality reduction"
            ],
            "programming": [
                "python", "r", "sql", "java", "scala", "c++", "julia"
            ],
            "ml_frameworks": [
                "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "xgboost",
                "lightgbm", "catboost", "h2o", "mxnet", "caffe"
            ],
            "data_visualization": [
                "data visualization", "tableau", "power bi", "matplotlib", "seaborn",
                "plotly", "bokeh", "d3.js", "ggplot"
            ],
            "big_data": [
                "hadoop", "spark", "hive", "pig", "kafka", "big data", "data lake",
                "data warehouse", "nosql", "mongodb", "cassandra", "hbase"
            ],
            "nlp": [
                "natural language processing", "nlp", "text mining", "sentiment analysis",
                "named entity recognition", "topic modeling", "word embeddings", "bert",
                "gpt", "transformers", "word2vec", "glove", "fasttext"
            ],
            "computer_vision": [
                "computer vision", "image processing", "object detection", "image classification",
                "semantic segmentation", "face recognition", "opencv", "cnn"
            ],
            "time_series": [
                "time series analysis", "forecasting", "arima", "prophet", "lstm", "rnn",
                "gru", "seasonal decomposition", "trend analysis"
            ],
            "cloud": [
                "aws", "azure", "gcp", "google cloud", "amazon web services",
                "cloud computing", "s3", "ec2", "lambda", "sagemaker"
            ],
            "mlops": [
                "mlops", "model deployment", "model monitoring", "ci/cd", "docker",
                "kubernetes", "kubeflow", "airflow", "luigi", "mlflow", "dvc"
            ]
        }
    
    def calculate_skills_match(self, resume_skills: List[str], 
                              required_skills: List[str], 
                              preferred_skills: List[str]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the match score for skills.
        
        Args:
            resume_skills (List[str]): Skills from the resume
            required_skills (List[str]): Required skills from the job description
            preferred_skills (List[str]): Preferred skills from the job description
            
        Returns:
            Tuple[float, Dict[str, Any]]: Score (0-1) and detailed match information
        """
        if not resume_skills or (not required_skills and not preferred_skills):
            return 0.0, {"matched": [], "missing": [], "preferred_matched": [], "semantic_matches": {}, "required_score": 0.0, "preferred_score": 0.0}
        
        # Convert to sets for easier operations
        resume_skills_set = set(s.lower() for s in resume_skills)
        required_skills_set = set(s.lower() for s in required_skills)
        preferred_skills_set = set(s.lower() for s in preferred_skills)
        
        # Check if this is likely a data science role
        is_data_science_role = False
        data_science_keywords = ["data scientist", "machine learning", "data science", "ml engineer", "ai", "deep learning"]
        for keyword in data_science_keywords:
            if any(keyword in skill.lower() for skill in required_skills + preferred_skills):
                is_data_science_role = True
                break
        
        # Calculate exact matches
        matched_required = resume_skills_set.intersection(required_skills_set)
        matched_preferred = resume_skills_set.intersection(preferred_skills_set)
        missing_required = required_skills_set - resume_skills_set
        
        # Initialize semantic_matches before any conditional blocks
        semantic_matches = {}
        
        # For data science roles, use category-based matching for better results
        if is_data_science_role:
            logger.info("Detected data science role, using category-based skill matching")
            
            # Create a mapping of skills to categories
            skill_to_category = {}
            for category, skills in self.data_science_skill_categories.items():
                for skill in skills:
                    skill_to_category[skill] = category
            
            # Check if missing skills are covered by category equivalents
            for missing_skill in list(missing_required):
                # Skip if the skill is too short
                if len(missing_skill) < 3:
                    continue
                
                # Check if the missing skill belongs to a category
                missing_skill_category = None
                for category, skills in self.data_science_skill_categories.items():
                    if any(skill in missing_skill for skill in skills):
                        missing_skill_category = category
                        break
                
                if missing_skill_category:
                    # Check if the resume has any skills from the same category
                    category_skills_in_resume = []
                    for resume_skill in resume_skills_set:
                        for skill in self.data_science_skill_categories[missing_skill_category]:
                            if skill in resume_skill:
                                category_skills_in_resume.append(resume_skill)
                    
                    if category_skills_in_resume:
                        # Find the best match from the category
                        best_match = category_skills_in_resume[0]
                        semantic_matches[missing_skill] = (best_match, 0.85)  # High confidence for category matches
        
        # Calculate semantic matches for remaining missing skills if spaCy is available
        if nlp and missing_required:
            for missing_skill in missing_required:
                # Skip if already matched by category
                if missing_skill in semantic_matches:
                    continue
                
                best_match = None
                best_score = 0
                
                # Skip if the skill is too short (less than 3 characters)
                if len(missing_skill) < 3:
                    continue
                
                missing_doc = nlp(missing_skill)
                
                for resume_skill in resume_skills_set:
                    # Skip if the skill is too short
                    if len(resume_skill) < 3:
                        continue
                    
                    resume_doc = nlp(resume_skill)
                    similarity = missing_doc.similarity(resume_doc)
                    
                    if similarity > 0.8 and similarity > best_score:  # Threshold for semantic similarity
                        best_match = resume_skill
                        best_score = similarity
                
                if best_match:
                    semantic_matches[missing_skill] = (best_match, best_score)
        
        # Initialize scores
        required_score = 0.0
        preferred_score = 0.0
        
        # Calculate scores
        if required_skills_set:
            # Count semantic matches as partial matches (0.8 weight)
            effective_matches = len(matched_required) + 0.8 * len(semantic_matches)
            required_score = effective_matches / len(required_skills_set)
        else:
            required_score = 1.0  # No required skills means perfect match
        
        if preferred_skills_set:
            preferred_score = len(matched_preferred) / len(preferred_skills_set)
        else:
            preferred_score = 0.0  # No preferred skills means no bonus
        
        # Combined score: 80% required skills, 20% preferred skills
        combined_score = 0.8 * required_score + 0.2 * preferred_score
        
        # Prepare detailed match information
        match_details = {
            "matched": list(matched_required),
            "missing": [skill for skill in missing_required if skill not in semantic_matches],
            "semantic_matches": semantic_matches,
            "preferred_matched": list(matched_preferred),
            "required_score": required_score,
            "preferred_score": preferred_score,
            "is_data_science_role": is_data_science_role
        }
        
        return combined_score, match_details
    
    def calculate_experience_match(self, resume_experience: Dict[str, Any], 
                                  required_experience: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the match score for experience.
        
        Args:
            resume_experience (Dict[str, Any]): Experience from the resume
            required_experience (Dict[str, Any]): Required experience from the job description
            
        Returns:
            Tuple[float, Dict[str, Any]]: Score (0-1) and detailed match information
        """
        # Extract years of experience
        resume_years = resume_experience.get("years", 0)
        required_years = required_experience.get("years", 0)
        
        # Calculate experience score
        if required_years == 0:
            years_score = 1.0  # No experience requirement means perfect match
        else:
            # If resume has more experience than required, it's a perfect match
            # Otherwise, calculate the ratio
            years_score = min(resume_years / required_years, 1.0) if resume_years > 0 else 0.0
        
        # Extract positions
        resume_positions = set(p.lower() for p in resume_experience.get("positions", []))
        required_positions = set(p.lower() for p in required_experience.get("positions", []))
        
        # Initialize positions_score and semantic_matches
        positions_score = 1.0
        semantic_matches = {}
        
        # Calculate position match score
        if required_positions:
            # Check for exact matches
            matched_positions = resume_positions.intersection(required_positions)
            
            # Calculate semantic matches for positions if spaCy is available
            if nlp and (required_positions - resume_positions):
                for req_position in required_positions - resume_positions:
                    best_match = None
                    best_score = 0
                    
                    req_doc = nlp(req_position)
                    
                    for resume_position in resume_positions:
                        resume_doc = nlp(resume_position)
                        similarity = req_doc.similarity(resume_doc)
                        
                        if similarity > 0.7 and similarity > best_score:  # Threshold for position similarity
                            best_match = resume_position
                            best_score = similarity
                    
                    if best_match:
                        semantic_matches[req_position] = (best_match, best_score)
            
            # Count exact matches and semantic matches (with 0.7 weight)
            effective_matches = len(matched_positions) + 0.7 * len(semantic_matches)
            positions_score = effective_matches / len(required_positions)
        
        # Combined score: 70% years, 30% positions
        combined_score = 0.7 * years_score + 0.3 * positions_score
        
        # Prepare detailed match information
        match_details = {
            "years": {
                "resume": resume_years,
                "required": required_years,
                "score": years_score
            },
            "positions": {
                "matched": list(resume_positions.intersection(required_positions)),
                "missing": [pos for pos in required_positions - resume_positions 
                           if pos not in semantic_matches],
                "semantic_matches": semantic_matches,
                "score": positions_score
            }
        }
        
        return combined_score, match_details
    
    def calculate_job_roles_match(self, resume_roles: List[str], 
                                 jd_roles: List[str]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the match score for job roles.
        
        Args:
            resume_roles (List[str]): Job roles from the resume
            jd_roles (List[str]): Job roles from the job description
            
        Returns:
            Tuple[float, Dict[str, Any]]: Score (0-1) and detailed match information
        """
        if not resume_roles or not jd_roles:
            return 0.0, {"matched": [], "missing": [], "semantic_matches": {}}
        
        # Convert to sets for easier operations
        resume_roles_set = set(r.lower() for r in resume_roles)
        jd_roles_set = set(r.lower() for r in jd_roles)
        
        # Calculate exact matches
        matched_roles = resume_roles_set.intersection(jd_roles_set)
        missing_roles = jd_roles_set - resume_roles_set
        
        # Initialize semantic_matches
        semantic_matches = {}
        
        # Calculate semantic matches for roles if spaCy is available
        if nlp and missing_roles:
            for jd_role in missing_roles:
                best_match = None
                best_score = 0
                
                jd_doc = nlp(jd_role)
                
                for resume_role in resume_roles_set:
                    resume_doc = nlp(resume_role)
                    similarity = jd_doc.similarity(resume_doc)
                    
                    if similarity > 0.75 and similarity > best_score:  # Threshold for role similarity
                        best_match = resume_role
                        best_score = similarity
                
                if best_match:
                    semantic_matches[jd_role] = (best_match, best_score)
        
        # Initialize score
        score = 0.0
        
        # Calculate score
        if jd_roles_set:
            # Count exact matches and semantic matches (with 0.75 weight)
            effective_matches = len(matched_roles) + 0.75 * len(semantic_matches)
            score = effective_matches / len(jd_roles_set)
        else:
            score = 1.0  # No job roles in JD means perfect match
        
        # Prepare detailed match information
        match_details = {
            "matched": list(matched_roles),
            "missing": [role for role in missing_roles if role not in semantic_matches],
            "semantic_matches": semantic_matches
        }
        
        return score, match_details
    
    def calculate_projects_match(self, resume_projects: List[Dict[str, Any]], 
                               jd_text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the match score for projects by comparing project descriptions
        with the job description text.
        
        Args:
            resume_projects (List[Dict[str, Any]]): Projects from the resume
            jd_text (str): Full job description text
            
        Returns:
            Tuple[float, Dict[str, Any]]: Score (0-1) and detailed match information
        """
        if not resume_projects or not jd_text:
            return 0.0, {"relevant_projects": [], "top_project_indices": []}
        
        # Extract project descriptions and skills
        project_descriptions = [p.get("description", "") for p in resume_projects]
        project_skills = [p.get("skills", []) for p in resume_projects]
        
        # Initialize variables
        text_similarities = []
        skill_relevance = []
        project_scores = []
        top_indices = []
        overall_score = 0.0
        
        # Calculate text similarity between project descriptions and job description
        # Use TF-IDF and cosine similarity
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            all_texts = project_descriptions + [jd_text]
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Calculate cosine similarity between each project and the job description
            for i in range(len(project_descriptions)):
                similarity = cosine_similarity(tfidf_matrix[i], tfidf_matrix[-1])[0][0]
                text_similarities.append(similarity)
        except Exception as e:
            logger.error(f"Error calculating text similarity: {e}")
            # Fallback to simple word overlap
            for desc in project_descriptions:
                jd_words = set(jd_text.lower().split())
                desc_words = set(desc.lower().split())
                overlap = len(jd_words.intersection(desc_words)) / len(jd_words) if jd_words else 0
                text_similarities.append(overlap)
        
        # Calculate skill relevance for each project
        for skills in project_skills:
            if not skills:
                skill_relevance.append(0.0)
                continue
            
            # Count how many project skills appear in the job description
            skill_count = sum(1 for skill in skills if skill.lower() in jd_text.lower())
            relevance = skill_count / len(skills) if skills else 0
            skill_relevance.append(relevance)
        
        # Combine text similarity and skill relevance for each project
        for i in range(len(resume_projects)):
            # 60% text similarity, 40% skill relevance
            combined_score = 0.6 * text_similarities[i] + 0.4 * skill_relevance[i]
            project_scores.append(combined_score)
        
        # Overall project match score is the average of the top 3 project scores (or all if less than 3)
        top_projects_count = min(3, len(project_scores))
        if top_projects_count > 0:
            top_indices = np.argsort(project_scores)[-top_projects_count:]
            top_scores = [project_scores[i] for i in top_indices]
            overall_score = sum(top_scores) / top_projects_count
        
        # Prepare detailed match information
        relevant_projects = []
        for i, score in enumerate(project_scores):
            if score > 0.3:  # Only include somewhat relevant projects
                relevant_projects.append({
                    "name": resume_projects[i].get("name", f"Project {i+1}"),
                    "relevance_score": score,
                    "text_similarity": text_similarities[i],
                    "skill_relevance": skill_relevance[i]
                })
        
        # Sort by relevance score
        relevant_projects.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        match_details = {
            "relevant_projects": relevant_projects,
            "top_project_indices": list(top_indices) if top_projects_count > 0 else []
        }
        
        return overall_score, match_details
    
    def calculate_match(self, resume_data: Dict[str, Any], 
                       jd_data: Dict[str, Any], 
                       jd_text: str) -> Dict[str, Any]:
        """
        Calculate the overall match score between a resume and a job description.
        
        Args:
            resume_data (Dict[str, Any]): Extracted data from the resume
            jd_data (Dict[str, Any]): Extracted data from the job description
            jd_text (str): Full job description text
            
        Returns:
            Dict[str, Any]: Match results including overall score and detailed component scores
        """
        # Detect profession mismatch
        profession_mismatch = self._detect_profession_mismatch(resume_data, jd_data)
        
        # Calculate component scores
        skills_score, skills_details = self.calculate_skills_match(
            resume_data.get("skills", []),
            jd_data.get("required_skills", []),
            jd_data.get("preferred_skills", [])
        )
        
        experience_score, experience_details = self.calculate_experience_match(
            resume_data.get("experience", {"years": 0, "positions": []}),
            jd_data.get("required_experience", {"years": 0, "positions": []})
        )
        
        roles_score, roles_details = self.calculate_job_roles_match(
            resume_data.get("job_roles", []),
            jd_data.get("job_roles", [])
        )
        
        projects_score, projects_details = self.calculate_projects_match(
            resume_data.get("projects", []),
            jd_text
        )
        
        # Apply profession mismatch penalty if detected
        if profession_mismatch:
            # Apply stronger penalties to skills and experience scores
            skills_score *= 0.5  # 50% penalty
            experience_score *= 0.4  # 60% penalty
            projects_score *= 0.5  # 50% penalty
            
            # Add profession mismatch information to the details
            skills_details["profession_mismatch"] = True
            experience_details["profession_mismatch"] = True
            roles_details["profession_mismatch"] = True
            projects_details["profession_mismatch"] = True
        
        # Calculate weighted overall score
        overall_score = (
            self.weights["skills"] * skills_score +
            self.weights["experience"] * experience_score +
            self.weights["job_roles"] * roles_score +
            self.weights["projects"] * projects_score
        )
        
        # Scale to 0-100
        overall_score_percentage = round(overall_score * 100)
        
        # Prepare match results
        match_results = {
            "overall_score": overall_score_percentage,
            "profession_mismatch": profession_mismatch,
            "component_scores": {
                "skills": {
                    "score": round(skills_score * 100),
                    "weight": self.weights["skills"],
                    "details": skills_details
                },
                "experience": {
                    "score": round(experience_score * 100),
                    "weight": self.weights["experience"],
                    "details": experience_details
                },
                "job_roles": {
                    "score": round(roles_score * 100),
                    "weight": self.weights["job_roles"],
                    "details": roles_details
                },
                "projects": {
                    "score": round(projects_score * 100),
                    "weight": self.weights["projects"],
                    "details": projects_details
                }
            }
        }
        
        # Generate feedback
        feedback = self._generate_feedback(match_results)
        match_results["feedback"] = feedback
        
        return match_results
    
    def _detect_profession_mismatch(self, resume_data: Dict[str, Any], jd_data: Dict[str, Any]) -> bool:
        """
        Detect if there's a significant mismatch between the resume profession and job description profession.
        
        Args:
            resume_data (Dict[str, Any]): Extracted data from the resume
            jd_data (Dict[str, Any]): Extracted data from the job description
            
        Returns:
            bool: True if a profession mismatch is detected, False otherwise
        """
        # Extract job roles from resume and job description
        resume_roles = resume_data.get("job_roles", [])
        jd_roles = jd_data.get("job_roles", [])
        
        # Define profession categories
        profession_categories = {
            "data_science": ["data scientist", "machine learning", "ml engineer", "ai engineer", 
                            "data analyst", "research scientist", "data science", "analytics"],
            "software_dev": ["software engineer", "developer", "programmer", "coder", "full stack", 
                            "backend", "frontend", "web developer", "mobile developer", "app developer"],
            "design": ["designer", "ux", "ui", "user experience", "user interface", "graphic", "visual"],
            "legal": ["lawyer", "attorney", "legal", "paralegal", "counsel"],
            "finance": ["financial", "accountant", "finance", "banking", "investment", "trader"],
            "healthcare": ["doctor", "nurse", "physician", "medical", "healthcare", "clinical"],
            "marketing": ["marketing", "seo", "content", "social media", "brand", "advertising"],
            "hr": ["hr", "human resources", "recruiter", "talent", "hiring", "personnel"],
            "sales": ["sales", "account executive", "business development", "customer success"],
            "operations": ["operations", "project manager", "product manager", "program manager"]
        }
        
        # Determine resume profession category
        resume_profession = None
        for category, keywords in profession_categories.items():
            if any(any(keyword in role.lower() for keyword in keywords) for role in resume_roles):
                resume_profession = category
                break
        
        # Determine job description profession category
        jd_profession = None
        for category, keywords in profession_categories.items():
            if any(any(keyword in role.lower() for keyword in keywords) for role in jd_roles):
                jd_profession = category
                break
        
        # Check if there's a mismatch
        if resume_profession and jd_profession and resume_profession != jd_profession:
            logger.info(f"Detected profession mismatch: Resume: {resume_profession}, JD: {jd_profession}")
            return True
        
        # Check for significant skill mismatch
        resume_skills = set(s.lower() for s in resume_data.get("skills", []))
        required_skills = set(s.lower() for s in jd_data.get("required_skills", []))
        
        # If there are required skills and very few matches, it's likely a profession mismatch
        if required_skills:
            match_ratio = len(resume_skills.intersection(required_skills)) / len(required_skills)
            if match_ratio < 0.2:  # Less than 20% of required skills match
                logger.info(f"Detected significant skill mismatch: match ratio {match_ratio:.2f}")
                return True
        
        # Check for significant experience gap
        resume_years = resume_data.get("experience", {}).get("years", 0)
        required_years = jd_data.get("required_experience", {}).get("years", 0)
        
        # If there's a large experience gap (more than 5 years), it's likely a mismatch
        if required_years > 0 and resume_years < required_years / 2 and required_years - resume_years > 5:
            logger.info(f"Detected significant experience gap: Resume: {resume_years} years, Required: {required_years} years")
            return True
        
        return False
    
    def _generate_feedback(self, match_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate feedback based on match results.
        
        Args:
            match_results (Dict[str, Any]): Match results
            
        Returns:
            Dict[str, Any]: Feedback including strengths, gaps, and suggestions
        """
        overall_score = match_results["overall_score"]
        component_scores = match_results["component_scores"]
        profession_mismatch = match_results.get("profession_mismatch", False)
        
        strengths = []
        gaps = []
        suggestions = []
        
        # If there's a profession mismatch, provide specific feedback
        if profession_mismatch:
            gaps.append("Your background appears to be in a different profession than what this job requires.")
            suggestions.append("This job may require a significant career change. Consider roles more aligned with your current profession or build a transition plan to acquire the necessary skills and experience.")
        
        # Check if this is a data science role
        is_data_science_role = component_scores["skills"]["details"].get("is_data_science_role", False)
        
        # Analyze skills
        skills_score = component_scores["skills"]["score"]
        skills_details = component_scores["skills"]["details"]
        
        if skills_score >= 80:
            strengths.append("Strong skills match with the job requirements.")
        elif skills_score >= 50:
            strengths.append("Decent skills match, but some improvements possible.")
        else:
            gaps.append("Significant skills gap compared to job requirements.")
        
        # Add missing skills to gaps
        missing_skills = skills_details.get("missing", [])
        if missing_skills:
            # Limit to top 5 missing skills if there are too many
            if len(missing_skills) > 5 and profession_mismatch:
                missing_skills_sample = missing_skills[:5]
                gaps.append(f"Missing required skills: {', '.join(missing_skills_sample)} and {len(missing_skills) - 5} more")
            else:
                gaps.append(f"Missing required skills: {', '.join(missing_skills)}")
            
            if is_data_science_role and not profession_mismatch:
                # Group missing skills by category for data science roles
                missing_by_category = {}
                for skill in missing_skills:
                    category = None
                    for cat_name, cat_skills in self.data_science_skill_categories.items():
                        if any(cat_skill in skill for cat_skill in cat_skills):
                            category = cat_name
                            break
                    
                    if category:
                        if category not in missing_by_category:
                            missing_by_category[category] = []
                        missing_by_category[category].append(skill)
                
                # Generate category-specific suggestions
                for category, skills in missing_by_category.items():
                    if category == "core_ml":
                        suggestions.append(f"Consider strengthening your machine learning fundamentals: {', '.join(skills)}")
                    elif category == "data_processing":
                        suggestions.append(f"Highlight your data preprocessing skills: {', '.join(skills)}")
                    elif category == "statistics":
                        suggestions.append(f"Emphasize your statistical knowledge: {', '.join(skills)}")
                    elif category == "programming":
                        suggestions.append(f"Add these programming languages to your skill set: {', '.join(skills)}")
                    elif category == "ml_frameworks":
                        suggestions.append(f"Consider learning these ML frameworks: {', '.join(skills)}")
                    elif category == "data_visualization":
                        suggestions.append(f"Strengthen your data visualization skills with: {', '.join(skills)}")
                    elif category == "big_data":
                        suggestions.append(f"Add big data technologies to your resume: {', '.join(skills)}")
                    elif category == "nlp":
                        suggestions.append(f"Consider adding NLP skills: {', '.join(skills)}")
                    elif category == "computer_vision":
                        suggestions.append(f"Add computer vision skills: {', '.join(skills)}")
                    elif category == "time_series":
                        suggestions.append(f"Highlight time series analysis skills: {', '.join(skills)}")
                    elif category == "cloud":
                        suggestions.append(f"Add cloud computing skills: {', '.join(skills)}")
                    elif category == "mlops":
                        suggestions.append(f"Consider adding MLOps skills: {', '.join(skills)}")
            elif not profession_mismatch:
                suggestions.append(f"Consider acquiring or highlighting these skills: {', '.join(missing_skills)}")
            elif profession_mismatch and len(missing_skills) <= 10:
                suggestions.append(f"If you're interested in this field, focus on acquiring these key skills: {', '.join(missing_skills)}")
        
        # Analyze experience
        exp_score = component_scores["experience"]["score"]
        exp_details = component_scores["experience"]["details"]
        
        resume_years = exp_details["years"]["resume"]
        required_years = exp_details["years"]["required"]
        
        if resume_years >= required_years:
            strengths.append(f"Experience ({resume_years} years) meets or exceeds the requirement ({required_years} years).")
        else:
            gap_years = required_years - resume_years
            if gap_years > 5 and profession_mismatch:
                gaps.append(f"Significant experience gap of {gap_years} years compared to the requirement.")
            else:
                gaps.append(f"Experience gap of {gap_years} years compared to the requirement.")
            
            if is_data_science_role and not profession_mismatch:
                if resume_years == 0:
                    suggestions.append("As you're starting in data science, focus on building projects that demonstrate your skills and consider internships or entry-level positions.")
                else:
                    suggestions.append("Highlight relevant projects, research, or academic experience to compensate for the experience gap in data science.")
            elif not profession_mismatch:
                suggestions.append("Highlight relevant projects or transferable skills to compensate for the experience gap.")
        
        # Analyze job roles
        roles_score = component_scores["job_roles"]["score"]
        roles_details = component_scores["job_roles"]["details"]
        
        if roles_score >= 70:
            strengths.append("Job roles align well with the position.")
        elif roles_score >= 40:
            strengths.append("Some relevant job roles, but could be better aligned.")
        else:
            gaps.append("Job roles don't align well with the position.")
        
        # Add missing roles to gaps
        missing_roles = roles_details.get("missing", [])
        if missing_roles and not profession_mismatch:
            gaps.append(f"Missing job roles: {', '.join(missing_roles)}")
            
            if is_data_science_role:
                data_science_roles = [role for role in missing_roles if any(kw in role.lower() for kw in ["data", "scientist", "machine", "learning", "ai", "ml"])]
                if data_science_roles:
                    suggestions.append(f"Consider highlighting experience or projects related to these data science roles: {', '.join(data_science_roles)}")
            else:
                suggestions.append("Consider highlighting experience in similar roles or related responsibilities.")
        
        # Analyze projects
        projects_score = component_scores["projects"]["score"]
        projects_details = component_scores["projects"]["details"]
        
        relevant_projects = projects_details.get("relevant_projects", [])
        
        if projects_score >= 70:
            strengths.append("Projects demonstrate relevant experience for the position.")
        elif projects_score >= 40:
            strengths.append("Some projects are relevant to the position.")
        else:
            gaps.append("Projects don't strongly demonstrate relevant experience.")
        
        # Filter out educational institutions from relevant projects
        filtered_relevant_projects = []
        educational_keywords = ['university', 'college', 'school', 'institute', 'academy', 'lpu', 'lovely professional']
        
        for project in relevant_projects:
            project_name = project.get("name", "").lower()
            if not any(edu_keyword in project_name for edu_keyword in educational_keywords):
                filtered_relevant_projects.append(project)
        
        if filtered_relevant_projects and not profession_mismatch:
            most_relevant = filtered_relevant_projects[0]["name"] if filtered_relevant_projects else ""
            if most_relevant:
                # Check if the project name is not an educational institution
                if not any(edu_keyword in most_relevant.lower() for edu_keyword in educational_keywords):
                    strengths.append(f"'{most_relevant}' is particularly relevant to this position.")
        elif not profession_mismatch:
            if is_data_science_role:
                suggestions.append("Consider adding data science projects that demonstrate skills like machine learning, data analysis, or visualization.")
            else:
                suggestions.append("Consider adding projects that demonstrate skills relevant to this position.")
        
        # For data science roles, add specific suggestions based on the overall score
        if is_data_science_role and not profession_mismatch:
            if overall_score < 60:
                # Check if there are any data science projects
                has_ds_projects = any(proj.get("is_data_science", False) for proj in filtered_relevant_projects)
                if not has_ds_projects:
                    suggestions.append("Create a portfolio of data science projects (e.g., on GitHub) demonstrating your skills in machine learning, data analysis, and visualization.")
                
                # Check if there are any ML frameworks in the matched skills
                matched_skills = skills_details.get("matched", [])
                has_ml_frameworks = any(framework in " ".join(matched_skills).lower() for framework in ["tensorflow", "pytorch", "scikit-learn", "sklearn"])
                if not has_ml_frameworks:
                    suggestions.append("Learn and showcase experience with popular ML frameworks like TensorFlow, PyTorch, or scikit-learn.")
                
                # Suggest Kaggle competitions for beginners
                suggestions.append("Participate in Kaggle competitions to gain practical experience and showcase your data science skills.")
        
        # Overall assessment
        if profession_mismatch:
            if overall_score >= 40:
                overall = "Despite some matching skills, your background appears to be in a different profession than what this job requires. Consider roles more aligned with your current profession or develop a transition plan."
            else:
                overall = "Your background is in a significantly different profession than what this job requires. This would likely require a major career change."
        elif overall_score >= 80:
            overall = "Strong match for the position. Consider applying with confidence."
        elif overall_score >= 60:
            overall = "Good match for the position. Some improvements could strengthen your application."
        elif overall_score >= 40:
            overall = "Moderate match. Consider addressing gaps before applying."
        else:
            overall = "Limited match. Significant improvements needed to be competitive for this position."
        
        return {
            "overall": overall,
            "strengths": strengths,
            "gaps": gaps,
            "suggestions": suggestions
        }


# Example usage
if __name__ == "__main__":
    matcher = Matcher()
    
    # Example resume data
    resume_data = {
        "skills": ["python", "javascript", "react", "node.js", "sql", "git"],
        "experience": {
            "years": 3,
            "positions": ["software developer", "web developer"]
        },
        "job_roles": ["software developer", "frontend developer", "web developer"],
        "projects": [
            {
                "name": "E-commerce Platform",
                "description": "Built a full-stack e-commerce platform using MERN stack",
                "skills": ["react", "node.js", "mongodb"]
            }
        ]
    }
    
    # Example job description data
    jd_data = {
        "required_skills": ["python", "javascript", "react", "sql", "aws"],
        "preferred_skills": ["typescript", "docker"],
        "required_experience": {
            "years": 5,
            "positions": ["software engineer", "full stack developer"]
        },
        "job_roles": ["software engineer", "full stack developer"]
    }
    
    # Example job description text
    jd_text = """
    We are looking for a Software Engineer with 5+ years of experience in web development.
    The ideal candidate should have strong skills in Python, JavaScript, React, and SQL.
    Experience with AWS is required. Knowledge of TypeScript and Docker is a plus.
    """
    
    # Calculate match
    match_results = matcher.calculate_match(resume_data, jd_data, jd_text)
    
    # Print results
    print(f"Overall Score: {match_results['overall_score']}/100")
    print("\nComponent Scores:")
    for component, data in match_results["component_scores"].items():
        print(f"- {component.capitalize()}: {data['score']}/100")
    
    print("\nFeedback:")
    feedback = match_results["feedback"]
    print(f"Overall: {feedback['overall']}")
    
    print("\nStrengths:")
    for strength in feedback["strengths"]:
        print(f"- {strength}")
    
    print("\nGaps:")
    for gap in feedback["gaps"]:
        print(f"- {gap}")
    
    print("\nSuggestions:")
    for suggestion in feedback["suggestions"]:
        print(f"- {suggestion}") 