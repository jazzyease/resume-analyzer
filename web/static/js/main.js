/**
 * Resume Analyzer - Main JavaScript
 * Handles animations, form submissions, and interactive elements
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Document loaded');
    
    // Initialize animations
    initAnimations();
    
    // Initialize file upload
    initFileUpload();
    
    // Initialize form submission
    initFormSubmission();
    
    // Initialize results page functionality if on results page
    if (document.querySelector('.score-meter')) {
        console.log('Results page detected');
        initResultsPage();
    }
});

/**
 * Initialize animations for page elements
 */
function initAnimations() {
    console.log('Initializing animations');
    
    // Add animation classes to elements
    const fadeElements = document.querySelectorAll('.fade-in-element');
    fadeElements.forEach((el, index) => {
        el.style.opacity = '1';
        el.style.transform = 'translateY(0)';
        el.style.transition = `opacity 0.5s ease ${index * 0.1}s, transform 0.5s ease ${index * 0.1}s`;
    });
}

/**
 * Initialize file upload functionality
 */
function initFileUpload() {
    console.log('Initializing file upload');
    
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            const fileNameElement = this.parentElement.querySelector('.file-name');
            if (fileNameElement && this.files.length > 0) {
                fileNameElement.textContent = this.files[0].name;
                fileNameElement.parentElement.classList.add('has-file');
            }
        });
    });
}

/**
 * Initialize form submission
 */
function initFormSubmission() {
    console.log('Initializing form submission');
    
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(e) {
            const resumeInput = form.querySelector('input[name="resume"]');
            const jobDescriptionInput = form.querySelector('input[name="job_description"]');
            
            if (!resumeInput.files.length || !jobDescriptionInput.files.length) {
                e.preventDefault();
                showError('Please upload both a resume and a job description.');
                return;
            }
            
            // Show loading indicator
            const loadingOverlay = document.createElement('div');
            loadingOverlay.className = 'loading-overlay';
            loadingOverlay.innerHTML = `
                <div class="spinner"></div>
                <div class="loading-text">Analyzing your resume...</div>
            `;
            document.body.appendChild(loadingOverlay);
        });
    }
}

/**
 * Show error message
 */
function showError(message) {
    console.log('Showing error:', message);
    
    let errorContainer = document.querySelector('.error-message');
    
    if (!errorContainer) {
        errorContainer = document.createElement('div');
        errorContainer.className = 'error-message';
        errorContainer.innerHTML = `<p>${message}</p>`;
        
        const form = document.querySelector('form');
        form.insertBefore(errorContainer, form.firstChild);
    } else {
        errorContainer.querySelector('p').textContent = message;
    }
}

/**
 * Initialize results page functionality
 */
function initResultsPage() {
    console.log('Initializing results page');
    
    // Initialize score meter
    initScoreMeter();
    
    // Initialize component scores animation
    initComponentScores();
    
    // Initialize collapsible sections
    initCollapsibleSections();
    
    // Add debug toggle functionality
    const debugToggle = document.getElementById('toggle-debug');
    if (debugToggle) {
        debugToggle.addEventListener('click', function() {
            const debugInfo = document.querySelector('.debug-info');
            if (debugInfo) {
                debugInfo.style.display = debugInfo.style.display === 'none' ? 'block' : 'none';
            }
        });
    }
}

/**
 * Initialize score meter animation
 */
function initScoreMeter() {
    console.log('Initializing score meter');
    
    const scoreMeter = document.querySelector('.score-meter');
    if (!scoreMeter) return;
    
    const scoreValue = parseInt(scoreMeter.getAttribute('data-score') || '0');
    const fill = scoreMeter.querySelector('.score-meter-fill');
    const text = scoreMeter.querySelector('.score-meter-text');
    
    if (!fill || !text) return;
    
    // Set the text
    text.textContent = scoreValue.toString();
    
    // Animate the fill
    const circumference = 2 * Math.PI * 90; // 90 is the radius
    fill.style.strokeDasharray = `${circumference} ${circumference}`;
    fill.style.strokeDashoffset = circumference;
    
    setTimeout(() => {
        const fillAmount = (scoreValue / 100) * circumference;
        fill.style.strokeDashoffset = circumference - fillAmount;
        
        // Set color based on score
        if (scoreValue < 40) {
            fill.style.stroke = '#ef4444'; // Red
            text.style.color = '#ef4444';
        } else if (scoreValue < 70) {
            fill.style.stroke = '#f59e0b'; // Yellow
            text.style.color = '#f59e0b';
        } else {
            fill.style.stroke = '#10b981'; // Green
            text.style.color = '#10b981';
        }
    }, 100);
}

/**
 * Initialize component scores animation
 */
function initComponentScores() {
    console.log('Initializing component scores');
    
    const componentScores = document.querySelectorAll('.component-score-value');
    componentScores.forEach((scoreElement, index) => {
        const scoreValue = parseInt(scoreElement.getAttribute('data-value') || '0');
        scoreElement.textContent = scoreValue.toString();
        
        // Set color based on score
        if (scoreValue < 40) {
            scoreElement.style.color = '#ef4444'; // Red
        } else if (scoreValue < 70) {
            scoreElement.style.color = '#f59e0b'; // Yellow
        } else {
            scoreElement.style.color = '#10b981'; // Green
        }
    });
}

/**
 * Initialize collapsible sections
 */
function initCollapsibleSections() {
    console.log('Initializing collapsible sections');
    
    document.querySelectorAll('[data-toggle]').forEach(button => {
        button.addEventListener('click', function() {
            const sectionId = this.getAttribute('data-toggle');
            const section = document.getElementById(sectionId);
            
            if (section) {
                const isExpanded = section.classList.contains('expanded');
                section.classList.toggle('expanded');
                
                // Update button text
                if (isExpanded) {
                    this.innerHTML = 'Show More <i class="fas fa-chevron-down"></i>';
                } else {
                    this.innerHTML = 'Show Less <i class="fas fa-chevron-up"></i>';
                }
            }
        });
    });
} 