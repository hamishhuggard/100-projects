// Dummy data for the IQ Cameo platform
const dummyData = {
    questions: [
        {
            id: 1,
            title: "How can I optimize my machine learning model for better performance?",
            category: "Technology",
            budget: 75,
            timeAgo: "2 hours ago",
            preview: "I'm working on a classification problem and my model is achieving 85% accuracy. I've tried different algorithms but can't seem to break through this plateau. What advanced techniques should I consider?",
            answers: 3,
            views: 156,
            author: "DataScientist2024",
            minIQ: 130
        },
        {
            id: 2,
            title: "What's the most efficient way to structure a startup's equity distribution?",
            category: "Business",
            budget: 100,
            timeAgo: "5 hours ago",
            preview: "I'm founding a tech startup with 3 co-founders and planning to raise funding. How should we structure equity to be fair while maintaining control and attracting investors?",
            answers: 1,
            views: 89,
            author: "StartupFounder",
            minIQ: 125
        },
        {
            id: 3,
            title: "Can you explain the philosophical implications of quantum mechanics?",
            category: "Philosophy",
            budget: 60,
            timeAgo: "1 day ago",
            preview: "I'm fascinated by how quantum mechanics challenges our classical understanding of reality. What does the Copenhagen interpretation really mean for our concept of consciousness and free will?",
            answers: 2,
            views: 234,
            author: "PhilosophyStudent",
            minIQ: 140
        },
        {
            id: 4,
            title: "What's the best approach to learn advanced mathematics as an adult?",
            category: "Science",
            budget: 45,
            timeAgo: "2 days ago",
            preview: "I have a basic understanding of calculus and linear algebra, but I want to dive deeper into abstract algebra, topology, and analysis. What's the most effective learning path?",
            answers: 4,
            views: 312,
            author: "MathEnthusiast",
            minIQ: 120
        },
        {
            id: 5,
            title: "How do I create a compelling narrative arc in my novel?",
            category: "Arts",
            budget: 55,
            timeAgo: "3 days ago",
            preview: "I'm writing a science fiction novel and struggling with pacing. My story has interesting characters and world-building, but the plot feels flat. How can I create more tension and engagement?",
            answers: 2,
            views: 178,
            author: "AspiringAuthor",
            minIQ: 115
        },
        {
            id: 6,
            title: "What's the future of renewable energy storage technology?",
            category: "Science",
            budget: 80,
            timeAgo: "4 days ago",
            preview: "With the push for renewable energy, what emerging storage technologies show the most promise? I'm particularly interested in grid-scale solutions and their economic viability.",
            answers: 3,
            views: 445,
            author: "EnergyAnalyst",
            minIQ: 135
        }
    ],
    
    experts: [
        {
            id: 1,
            name: "Dr. Sarah Chen",
            iq: 145,
            rate: 50,
            specialty: "Machine Learning & AI",
            avatar: "👩‍🔬",
            rating: 4.9,
            questionsAnswered: 127
        },
        {
            id: 2,
            name: "Prof. Marcus Johnson",
            iq: 142,
            rate: 75,
            specialty: "Business Strategy",
            avatar: "👨‍💼",
            rating: 4.8,
            questionsAnswered: 89
        },
        {
            id: 3,
            name: "Dr. Elena Rodriguez",
            iq: 138,
            rate: 40,
            specialty: "Philosophy & Ethics",
            avatar: "👩‍🏫",
            rating: 4.7,
            questionsAnswered: 156
        },
        {
            id: 4,
            name: "Dr. James Wilson",
            iq: 135,
            rate: 65,
            specialty: "Mathematics",
            avatar: "👨‍🔬",
            rating: 4.9,
            questionsAnswered: 203
        },
        {
            id: 5,
            name: "Prof. Lisa Thompson",
            iq: 132,
            rate: 45,
            specialty: "Creative Writing",
            avatar: "👩‍🎨",
            rating: 4.6,
            questionsAnswered: 78
        }
    ],
    
    answers: [
        {
            id: 1,
            questionId: 1,
            author: "Dr. Sarah Chen",
            content: "For breaking through the 85% accuracy plateau, I'd recommend trying ensemble methods like Random Forest or Gradient Boosting. Also, consider feature engineering and hyperparameter tuning using techniques like Bayesian optimization. The key is often in the data preprocessing rather than just the algorithm choice.",
            timeAgo: "1 hour ago",
            rating: 5,
            isExpert: true
        },
        {
            id: 2,
            questionId: 1,
            author: "ML_Researcher",
            content: "Have you tried deep learning approaches? CNNs or RNNs might capture patterns that traditional ML algorithms miss. Also, check if your data is balanced - sometimes accuracy can be misleading.",
            timeAgo: "30 minutes ago",
            rating: 4,
            isExpert: false
        },
        {
            id: 3,
            questionId: 2,
            author: "Prof. Marcus Johnson",
            content: "For a 4-founder startup, I recommend: Founders get 60-70% total, employees 15-20%, and investors 10-25%. Use vesting schedules (4 years with 1-year cliff) to protect against early departures. Consider different classes of shares for voting vs. economic rights.",
            timeAgo: "3 hours ago",
            rating: 5,
            isExpert: true
        }
    ]
};

// DOM elements
const questionList = document.getElementById('questionList');
const askQuestionModal = document.getElementById('askQuestionModal');
const questionDetailModal = document.getElementById('questionDetailModal');
const questionForm = document.getElementById('questionForm');
const answerForm = document.getElementById('answerForm');

// Initialize the app
function init() {
    renderQuestions();
    setupEventListeners();
}

// Render questions in the main feed
function renderQuestions() {
    questionList.innerHTML = '';
    
    dummyData.questions.forEach(question => {
        const questionElement = createQuestionElement(question);
        questionList.appendChild(questionElement);
    });
}

// Create a question element
function createQuestionElement(question) {
    const questionDiv = document.createElement('div');
    questionDiv.className = 'question-item';
    questionDiv.onclick = () => openQuestionDetail(question);
    
    questionDiv.innerHTML = `
        <div class="question-header">
            <div class="question-title">${question.title}</div>
        </div>
        <div class="question-meta">
            <span class="category-badge">${question.category}</span>
            <span class="budget-badge">$${question.budget}</span>
            <span class="time-ago">${question.timeAgo}</span>
        </div>
        <div class="question-preview">${question.preview}</div>
        <div class="question-footer">
            <div class="question-stats">
                <span><i class="fas fa-comments"></i> ${question.answers} answers</span>
                <span><i class="fas fa-eye"></i> ${question.views} views</span>
                <span><i class="fas fa-user"></i> ${question.author}</span>
            </div>
            <div class="question-stats">
                <span><i class="fas fa-brain"></i> IQ ${question.minIQ}+</span>
            </div>
        </div>
    `;
    
    return questionDiv;
}

// Open question detail modal
function openQuestionDetail(question) {
    const modal = document.getElementById('questionDetailModal');
    const title = document.getElementById('modalQuestionTitle');
    const category = document.getElementById('modalQuestionCategory');
    const budget = document.getElementById('modalQuestionBudget');
    const time = document.getElementById('modalQuestionTime');
    const body = document.getElementById('modalQuestionBody');
    const answersList = document.getElementById('modalAnswersList');
    
    title.textContent = question.title;
    category.textContent = question.category;
    budget.textContent = `$${question.budget}`;
    time.textContent = question.timeAgo;
    body.textContent = question.preview;
    
    // Render answers
    renderAnswers(question.id, answersList);
    
    modal.style.display = 'block';
}

// Render answers for a question
function renderAnswers(questionId, container) {
    const answers = dummyData.answers.filter(answer => answer.questionId === questionId);
    
    if (answers.length === 0) {
        container.innerHTML = '<p>No answers yet. Be the first to answer!</p>';
        return;
    }
    
    container.innerHTML = answers.map(answer => `
        <div class="answer-item">
            <div class="answer-header">
                <span class="answer-author">${answer.author}</span>
                <span class="answer-time">${answer.timeAgo}</span>
            </div>
            <div class="answer-content">${answer.content}</div>
        </div>
    `).join('');
}

// Open ask question modal
function openAskQuestionModal() {
    askQuestionModal.style.display = 'block';
}

// Close ask question modal
function closeAskQuestionModal() {
    askQuestionModal.style.display = 'none';
    questionForm.reset();
}

// Close question detail modal
function closeQuestionDetailModal() {
    questionDetailModal.style.display = 'none';
}

// Setup event listeners
function setupEventListeners() {
    // Close modals when clicking outside
    window.onclick = function(event) {
        if (event.target === askQuestionModal) {
            closeAskQuestionModal();
        }
        if (event.target === questionDetailModal) {
            closeQuestionDetailModal();
        }
    };
    
    // Handle question form submission
    questionForm.addEventListener('submit', handleQuestionSubmit);
    
    // Handle answer form submission
    answerForm.addEventListener('submit', handleAnswerSubmit);
    
    // Filter buttons
    const filterBtns = document.querySelectorAll('.filter-btn');
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            // In a real app, this would filter the questions
        });
    });
}

// Handle question form submission
function handleQuestionSubmit(e) {
    e.preventDefault();
    
    const formData = new FormData(questionForm);
    const questionData = {
        title: formData.get('title'),
        category: formData.get('category'),
        details: formData.get('details'),
        minIQ: formData.get('minIQ'),
        budget: formData.get('budget')
    };
    
    // In a real app, this would send data to backend
    console.log('New question:', questionData);
    
    // Add to dummy data
    const newQuestion = {
        id: dummyData.questions.length + 1,
        title: questionData.title,
        category: questionData.category,
        budget: parseInt(questionData.budget),
        timeAgo: 'Just now',
        preview: questionData.details,
        answers: 0,
        views: 0,
        author: 'CurrentUser',
        minIQ: parseInt(questionData.minIQ)
    };
    
    dummyData.questions.unshift(newQuestion);
    renderQuestions();
    
    // Show success message
    alert('Question submitted successfully!');
    closeAskQuestionModal();
}

// Handle answer form submission
function handleAnswerSubmit(e) {
    e.preventDefault();
    
    const formData = new FormData(answerForm);
    const answerText = formData.get('answer');
    
    if (!answerText.trim()) {
        alert('Please enter an answer');
        return;
    }
    
    // In a real app, this would send data to backend
    console.log('New answer:', answerText);
    
    // Show success message
    alert('Answer submitted successfully!');
    answerForm.reset();
}

// Filter questions by category
function filterByCategory(category) {
    const filteredQuestions = category === 'all' 
        ? dummyData.questions 
        : dummyData.questions.filter(q => q.category.toLowerCase() === category.toLowerCase());
    
    questionList.innerHTML = '';
    filteredQuestions.forEach(question => {
        const questionElement = createQuestionElement(question);
        questionList.appendChild(questionElement);
    });
}

// Search questions
function searchQuestions(query) {
    const searchResults = dummyData.questions.filter(question => 
        question.title.toLowerCase().includes(query.toLowerCase()) ||
        question.preview.toLowerCase().includes(query.toLowerCase())
    );
    
    questionList.innerHTML = '';
    searchResults.forEach(question => {
        const questionElement = createQuestionElement(question);
        questionList.appendChild(questionElement);
    });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init); 