/**
 * Student Performance Predictor - JavaScript
 * Handles form validation, AJAX requests, and UI interactions
 */

// DOM Elements
const form = document.getElementById('predictionForm');
const submitBtn = document.getElementById('submitBtn');
const btnText = document.querySelector('.btn-text');
const btnLoading = document.querySelector('.btn-loading');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const newPredictionBtn = document.getElementById('newPredictionBtn');
const retryBtn = document.getElementById('retryBtn');

// Result elements
const predictionResult = document.getElementById('predictionResult');
const resultIcon = document.getElementById('resultIcon');
const resultPrediction = document.getElementById('resultPrediction');
const resultConfidence = document.getElementById('resultConfidence');
const passProb = document.getElementById('passProb');
const failProb = document.getElementById('failProb');
const passFill = document.getElementById('passFill');
const failFill = document.getElementById('failFill');
const inputSummary = document.getElementById('inputSummary');
const errorMessage = document.getElementById('errorMessage');

// Form validation rules
const validationRules = {
    study_hours: { min: 0, max: 24, step: 0.5 },
    attendance: { min: 0, max: 100, step: 1 },
    previous_marks: { min: 0, max: 100, step: 1 },
    assignment_completion: { min: 0, max: 100, step: 1 }
};

/**
 * Initialize the application
 */
function init() {
    console.log('Student Performance Predictor initialized');
    
    // Add event listeners
    form.addEventListener('submit', handleFormSubmit);
    newPredictionBtn.addEventListener('click', resetForm);
    retryBtn.addEventListener('click', hideError);
    
    // Add input validation
    addInputValidation();
    
    // Add smooth scrolling for better UX
    addSmoothScrolling();
}

/**
 * Add real-time input validation
 */
function addInputValidation() {
    const inputs = form.querySelectorAll('input[type="number"]');
    
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            validateInput(this);
        });
        
        input.addEventListener('blur', function() {
            validateInput(this);
        });
    });
}

/**
 * Validate individual input field
 * @param {HTMLInputElement} input - The input element to validate
 */
function validateInput(input) {
    const value = parseFloat(input.value);
    const rules = validationRules[input.name];
    
    if (!rules) return;
    
    // Remove existing validation classes
    input.classList.remove('valid', 'invalid');
    
    // Check if value is within range
    if (isNaN(value) || value < rules.min || value > rules.max) {
        input.classList.add('invalid');
        return false;
    } else {
        input.classList.add('valid');
        return true;
    }
}

/**
 * Validate entire form
 * @returns {boolean} - True if form is valid
 */
function validateForm() {
    const inputs = form.querySelectorAll('input[type="number"]');
    let isValid = true;
    
    inputs.forEach(input => {
        if (!validateInput(input)) {
            isValid = false;
        }
    });
    
    return isValid;
}

/**
 * Handle form submission
 * @param {Event} e - Form submit event
 */
async function handleFormSubmit(e) {
    e.preventDefault();
    
    // Validate form
    if (!validateForm()) {
        showError('Please check your input values and ensure they are within the valid ranges.');
        return;
    }
    
    // Show loading state
    setLoadingState(true);
    hideError();
    hideResults();
    
    // Collect form data
    const formData = new FormData(form);
    const data = {
        study_hours: parseFloat(formData.get('study_hours')),
        attendance: parseFloat(formData.get('attendance')),
        previous_marks: parseFloat(formData.get('previous_marks')),
        assignment_completion: parseFloat(formData.get('assignment_completion'))
    };
    
    try {
        // Make prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'Prediction failed');
        }
        
        // Display results
        displayResults(result, data);
        
    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message || 'An error occurred while making the prediction. Please try again.');
    } finally {
        setLoadingState(false);
    }
}

/**
 * Set loading state for submit button
 * @param {boolean} loading - Whether to show loading state
 */
function setLoadingState(loading) {
    submitBtn.disabled = loading;
    
    if (loading) {
        btnText.style.display = 'none';
        btnLoading.style.display = 'flex';
    } else {
        btnText.style.display = 'block';
        btnLoading.style.display = 'none';
    }
}

/**
 * Display prediction results
 * @param {Object} result - Prediction result from API
 * @param {Object} inputData - Original input data
 */
function displayResults(result, inputData) {
    // Set prediction result
    const isPassing = result.prediction === 'Pass';
    
    predictionResult.className = `prediction-result ${isPassing ? 'pass' : 'fail'}`;
    resultIcon.textContent = isPassing ? '🎉' : '😔';
    resultPrediction.textContent = `Prediction: ${result.prediction}`;
    resultConfidence.textContent = `Confidence: ${result.confidence}%`;
    
    // Set probability bars
    passProb.textContent = `${result.probability_pass}%`;
    failProb.textContent = `${result.probability_fail}%`;
    
    // Animate probability bars
    setTimeout(() => {
        passFill.style.width = `${result.probability_pass}%`;
        failFill.style.width = `${result.probability_fail}%`;
    }, 100);
    
    // Display input summary
    displayInputSummary(inputData);
    
    // Show results section
    showResults();
}

/**
 * Display input summary
 * @param {Object} inputData - Original input data
 */
function displayInputSummary(inputData) {
    const summaryItems = [
        { label: 'Study Hours', value: `${inputData.study_hours} hours/day`, icon: '📚' },
        { label: 'Attendance', value: `${inputData.attendance}%`, icon: '📅' },
        { label: 'Previous Marks', value: `${inputData.previous_marks}/100`, icon: '📊' },
        { label: 'Assignment Completion', value: `${inputData.assignment_completion}%`, icon: '✅' }
    ];
    
    inputSummary.innerHTML = summaryItems.map(item => `
        <div class="summary-item">
            <div class="summary-label">${item.icon} ${item.label}</div>
            <div class="summary-value">${item.value}</div>
        </div>
    `).join('');
}

/**
 * Show results section with animation
 */
function showResults() {
    resultsSection.style.display = 'block';
    
    // Smooth scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }, 100);
}

/**
 * Hide results section
 */
function hideResults() {
    resultsSection.style.display = 'none';
}

/**
 * Show error message
 * @param {string} message - Error message to display
 */
function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    
    // Smooth scroll to error
    setTimeout(() => {
        errorSection.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }, 100);
}

/**
 * Hide error section
 */
function hideError() {
    errorSection.style.display = 'none';
}

/**
 * Reset form and hide results
 */
function resetForm() {
    form.reset();
    hideResults();
    hideError();
    
    // Remove validation classes
    const inputs = form.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.classList.remove('valid', 'invalid');
    });
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/**
 * Add smooth scrolling behavior
 */
function addSmoothScrolling() {
    // Add smooth scrolling for any anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

/**
 * Utility function to format numbers
 * @param {number} num - Number to format
 * @param {number} decimals - Number of decimal places
 * @returns {string} - Formatted number
 */
function formatNumber(num, decimals = 1) {
    return parseFloat(num).toFixed(decimals);
}

/**
 * Utility function to get random tip for better performance
 * @returns {string} - Random tip
 */
function getRandomTip() {
    const tips = [
        "Regular study habits lead to better academic performance!",
        "Consistent attendance is key to understanding course material.",
        "Completing assignments on time helps reinforce learning.",
        "Previous performance is a good indicator, but improvement is always possible!",
        "Balanced study schedule with adequate rest improves retention.",
        "Active participation in class discussions enhances understanding."
    ];
    
    return tips[Math.floor(Math.random() * tips.length)];
}

/**
 * Add educational tips (could be used for future enhancements)
 */
function addEducationalTips() {
    // This function could be used to show tips based on prediction results
    // For now, it's a placeholder for future enhancements
    console.log('Educational tip:', getRandomTip());
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', init);

// Add some console styling for development
console.log('%c🎓 Student Performance Predictor', 'color: #667eea; font-size: 16px; font-weight: bold;');
console.log('%cBuilt with Flask & Naive Bayes ML', 'color: #718096; font-size: 12px;');

// Export functions for potential testing (if needed)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        validateInput,
        validateForm,
        formatNumber,
        getRandomTip
    };
}