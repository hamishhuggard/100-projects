// Dummy data for Genius Coach platform
const dummyData = {
    geniuses: [
        {
            id: 1,
            name: "Dr. Sarah Chen",
            category: "technology",
            specialty: "AI & Machine Learning",
            iq: 145,
            dmRate: 25,
            voiceRate: 50,
            callRate: 200,
            rating: 4.9,
            dmRating: 4.8,
            questionsAnswered: 127,
            avatar: "👩‍🔬",
            verified: true
        },
        {
            id: 2,
            name: "Prof. Marcus Johnson",
            category: "business",
            specialty: "Strategic Consulting",
            iq: 142,
            dmRate: 35,
            voiceRate: 75,
            callRate: 300,
            rating: 4.8,
            dmRating: 4.7,
            questionsAnswered: 89,
            avatar: "👨‍💼",
            verified: true
        },
        {
            id: 3,
            name: "Dr. Elena Rodriguez",
            category: "philosophy",
            specialty: "Ethics & Logic",
            iq: 138,
            dmRate: 20,
            voiceRate: 40,
            callRate: 150,
            rating: 4.7,
            dmRating: 4.6,
            questionsAnswered: 156,
            avatar: "👩‍🏫",
            verified: true
        },
        {
            id: 4,
            name: "Dr. James Wilson",
            category: "mathematics",
            specialty: "Pure Mathematics",
            iq: 135,
            dmRate: 30,
            voiceRate: 60,
            callRate: 250,
            rating: 4.9,
            dmRating: 4.8,
            questionsAnswered: 203,
            avatar: "👨‍🔬",
            verified: true
        },
        {
            id: 5,
            name: "Prof. Lisa Thompson",
            category: "arts",
            specialty: "Creative Writing",
            iq: 132,
            dmRate: 15,
            voiceRate: 35,
            callRate: 120,
            rating: 4.6,
            dmRating: 4.5,
            questionsAnswered: 78,
            avatar: "👩‍🎨",
            verified: true
        },
        {
            id: 6,
            name: "Dr. Alex Kim",
            category: "technology",
            specialty: "Cybersecurity",
            iq: 140,
            dmRate: 30,
            voiceRate: 65,
            callRate: 250,
            rating: 4.8,
            dmRating: 4.7,
            questionsAnswered: 95,
            avatar: "👨‍💻",
            verified: true
        },
        {
            id: 7,
            name: "Dr. Maria Santos",
            category: "science",
            specialty: "Quantum Physics",
            iq: 143,
            dmRate: 40,
            voiceRate: 80,
            callRate: 300,
            rating: 4.9,
            dmRating: 4.8,
            questionsAnswered: 112,
            avatar: "👩‍🔬",
            verified: true
        },
        {
            id: 8,
            name: "Prof. David Brown",
            category: "business",
            specialty: "Financial Markets",
            iq: 136,
            dmRate: 45,
            voiceRate: 90,
            callRate: 350,
            rating: 4.7,
            dmRating: 4.6,
            questionsAnswered: 67,
            avatar: "👨‍💼",
            verified: true
        },
        {
            id: 9,
            name: "Dr. Rachel Green",
            category: "arts",
            specialty: "Digital Design",
            iq: 130,
            dmRate: 20,
            voiceRate: 45,
            callRate: 180,
            rating: 4.5,
            dmRating: 4.4,
            questionsAnswered: 45,
            avatar: "👩‍🎨",
            verified: true
        },
        {
            id: 10,
            name: "Dr. Michael Chang",
            category: "mathematics",
            specialty: "Statistics",
            iq: 137,
            dmRate: 25,
            voiceRate: 55,
            callRate: 220,
            rating: 4.8,
            dmRating: 4.7,
            questionsAnswered: 134,
            avatar: "👨‍🔬",
            verified: true
        }
    ],
    
    questions: [
        {
            id: 1,
            title: "How can I optimize my machine learning model for better performance?",
            category: "technology",
            budget: 75,
            timeAgo: "2 hours ago",
            preview: "I'm working on a classification problem and my model is achieving 85% accuracy. I've tried different algorithms but can't seem to break through this plateau. What advanced techniques should I consider?",
            answers: [
                {
                    id: 1,
                    geniusId: 1,
                    geniusName: "Dr. Sarah Chen",
                    geniusIQ: 145,
                    geniusRate: 25,
                    content: "For breaking through the 85% accuracy plateau, I'd recommend trying ensemble methods like Random Forest or Gradient Boosting. Also, consider feature engineering and hyperparameter tuning using techniques like Bayesian optimization. The key is often in the data preprocessing rather than just the algorithm choice.",
                    rating: 5,
                    timeAgo: "1 hour ago"
                }
            ],
            author: "DataScientist2024",
            minIQ: 130
        },
        {
            id: 2,
            title: "What's the most efficient way to structure a startup's equity distribution?",
            category: "business",
            budget: 100,
            timeAgo: "5 hours ago",
            preview: "I'm founding a tech startup with 3 co-founders and planning to raise funding. How should we structure equity to be fair while maintaining control and attracting investors?",
            answers: [
                {
                    id: 2,
                    geniusId: 2,
                    geniusName: "Prof. Marcus Johnson",
                    geniusIQ: 142,
                    geniusRate: 35,
                    content: "For a 4-founder startup, I recommend: Founders get 60-70% total, employees 15-20%, and investors 10-25%. Use vesting schedules (4 years with 1-year cliff) to protect against early departures. Consider different classes of shares for voting vs. economic rights.",
                    rating: 5,
                    timeAgo: "3 hours ago"
                }
            ],
            author: "StartupFounder",
            minIQ: 125
        },
        {
            id: 3,
            title: "Can you explain the philosophical implications of quantum mechanics?",
            category: "philosophy",
            budget: 60,
            timeAgo: "1 day ago",
            preview: "I'm fascinated by how quantum mechanics challenges our classical understanding of reality. What does the Copenhagen interpretation really mean for our concept of consciousness and free will?",
            answers: [
                {
                    id: 3,
                    geniusId: 3,
                    geniusName: "Dr. Elena Rodriguez",
                    geniusIQ: 138,
                    geniusRate: 20,
                    content: "The Copenhagen interpretation suggests that reality is not objective until observed, which challenges our notions of consciousness and free will. It implies that consciousness plays a fundamental role in creating reality, raising questions about whether free will is an illusion or if consciousness itself is the source of reality.",
                    rating: 4,
                    timeAgo: "12 hours ago"
                }
            ],
            author: "PhilosophyStudent",
            minIQ: 140
        },
        {
            id: 4,
            title: "What's the best approach to learn advanced mathematics as an adult?",
            category: "mathematics",
            budget: 45,
            timeAgo: "2 days ago",
            preview: "I have a basic understanding of calculus and linear algebra, but I want to dive deeper into abstract algebra, topology, and analysis. What's the most effective learning path?",
            answers: [
                {
                    id: 4,
                    geniusId: 4,
                    geniusName: "Dr. James Wilson",
                    geniusIQ: 135,
                    geniusRate: 30,
                    content: "Start with Real Analysis to build rigorous thinking, then move to Abstract Algebra for algebraic structures. Topology should come after as it requires both analytical and algebraic thinking. Use resources like MIT OpenCourseWare and work through problem sets systematically.",
                    rating: 5,
                    timeAgo: "1 day ago"
                }
            ],
            author: "MathEnthusiast",
            minIQ: 120
        },
        {
            id: 5,
            title: "How do I create a compelling narrative arc in my novel?",
            category: "arts",
            budget: 55,
            timeAgo: "3 days ago",
            preview: "I'm writing a science fiction novel and struggling with pacing. My story has interesting characters and world-building, but the plot feels flat. How can I create more tension and engagement?",
            answers: [
                {
                    id: 5,
                    geniusId: 5,
                    geniusName: "Prof. Lisa Thompson",
                    geniusIQ: 132,
                    geniusRate: 15,
                    content: "Focus on creating clear stakes and obstacles. Every scene should either advance the plot or develop character. Use the three-act structure: setup (25%), confrontation (50%), resolution (25%). Introduce a ticking clock or deadline to create urgency.",
                    rating: 4,
                    timeAgo: "2 days ago"
                }
            ],
            author: "AspiringAuthor",
            minIQ: 115
        },
        {
            id: 6,
            title: "What's the future of renewable energy storage technology?",
            category: "science",
            budget: 80,
            timeAgo: "4 days ago",
            preview: "With the push for renewable energy, what emerging storage technologies show the most promise? I'm particularly interested in grid-scale solutions and their economic viability.",
            answers: [
                {
                    id: 6,
                    geniusId: 7,
                    geniusName: "Dr. Maria Santos",
                    geniusIQ: 143,
                    geniusRate: 40,
                    content: "Solid-state batteries and flow batteries show the most promise for grid storage. Hydrogen storage is also emerging as a viable option for seasonal storage. The economics are improving rapidly, with costs expected to drop 50% by 2030. Pumped hydro remains the most cost-effective for large-scale storage.",
                    rating: 5,
                    timeAgo: "3 days ago"
                }
            ],
            author: "EnergyAnalyst",
            minIQ: 135
        }
    ]
};

// DOM elements
let currentPage = window.location.pathname.includes('geniuses.html') ? 'geniuses' : 'home';

// Initialize the app
function init() {
    if (currentPage === 'geniuses') {
        renderGeniusesTable();
        setupGeniusesEventListeners();
    } else {
        renderQuestionsGrid();
        setupHomeEventListeners();
    }
}

// Render geniuses table
function renderGeniusesTable() {
    const tableBody = document.getElementById('geniusesTableBody');
    if (!tableBody) return;
    
    tableBody.innerHTML = '';
    
    dummyData.geniuses.forEach(genius => {
        const row = createGeniusTableRow(genius);
        tableBody.appendChild(row);
    });
}

// Create a genius table row
function createGeniusTableRow(genius) {
    const row = document.createElement('tr');
    
    row.innerHTML = `
        <td>
            <div class="genius-info-cell">
                <div class="genius-avatar">${genius.avatar}</div>
                <div class="genius-details-cell">
                    <h4>${genius.name}</h4>
                    <div class="genius-specialty">${genius.specialty}</div>
                </div>
            </div>
        </td>
        <td>
            <span class="category-badge">${genius.category.charAt(0).toUpperCase() + genius.category.slice(1)}</span>
        </td>
        <td>
            <strong>${genius.iq}</strong>
        </td>
        <td class="rate-cell">
            $${genius.dmRate}
            <div class="rating-cell">
                <span class="stars">${'★'.repeat(Math.floor(genius.dmRating))}</span>
                <span class="rating-number">${genius.dmRating}</span>
            </div>
        </td>
        <td class="rate-cell">$${genius.voiceRate}</td>
        <td class="rate-cell">$${genius.callRate}</td>
        <td>
            <div class="rating-cell">
                <span class="stars">${'★'.repeat(Math.floor(genius.rating))}</span>
                <span class="rating-number">${genius.rating}</span>
            </div>
        </td>
        <td>
            <div class="actions-cell">
                <button class="btn btn-primary-small" onclick="bookSession(${genius.id}, 'dm')">DM</button>
                <button class="btn btn-secondary-small" onclick="bookSession(${genius.id}, 'call')">Call</button>
            </div>
        </td>
    `;
    
    return row;
}

// Render questions grid for homepage
function renderQuestionsGrid() {
    const questionsGrid = document.getElementById('questionsGrid');
    if (!questionsGrid) return;
    
    questionsGrid.innerHTML = '';
    
    dummyData.questions.forEach(question => {
        const questionCard = createQuestionCard(question);
        questionsGrid.appendChild(questionCard);
    });
}

// Create a question card
function createQuestionCard(question) {
    const card = document.createElement('div');
    card.className = 'question-card';
    
    const answer = question.answers[0]; // Get the first answer
    const genius = dummyData.geniuses.find(g => g.id === answer.geniusId);
    
    card.innerHTML = `
        <h3>${question.title}</h3>
        <div class="question-meta">
            <span class="category-badge">${question.category.charAt(0).toUpperCase() + question.category.slice(1)}</span>
            <span class="budget-badge">$${question.budget}</span>
        </div>
        <div class="question-preview">${question.preview}</div>
        
        <div class="answer-section">
            <div class="answer-header">
                <span class="answer-author">${answer.geniusName}</span>
                <span class="answer-rating">${'★'.repeat(answer.rating)}</span>
            </div>
            <div class="answer-content">${answer.content}</div>
        </div>
        
        <div class="genius-info">
            <div class="genius-details">
                <span class="genius-iq">IQ ${genius.iq}</span>
                <span class="genius-rate">$${genius.dmRate}/DM</span>
            </div>
            <a href="geniuses.html#${question.category}" class="btn btn-primary-small">Find Similar Geniuses</a>
        </div>
    `;
    
    return card;
}

// Filter geniuses
function filterGeniuses() {
    const categoryFilter = document.getElementById('categoryFilter').value;
    const iqFilter = document.getElementById('iqFilter').value;
    const serviceFilter = document.getElementById('serviceFilter').value;
    const sortBy = document.getElementById('sortBy').value;
    
    let filteredGeniuses = [...dummyData.geniuses];
    
    // Apply filters
    if (categoryFilter) {
        filteredGeniuses = filteredGeniuses.filter(g => g.category === categoryFilter);
    }
    
    if (iqFilter) {
        filteredGeniuses = filteredGeniuses.filter(g => g.iq >= parseInt(iqFilter));
    }
    
    if (serviceFilter) {
        if (serviceFilter === 'dm') {
            filteredGeniuses = filteredGeniuses.filter(g => g.dmRate <= 50);
        } else if (serviceFilter === 'voice') {
            filteredGeniuses = filteredGeniuses.filter(g => g.voiceRate <= 100);
        } else if (serviceFilter === 'call') {
            filteredGeniuses = filteredGeniuses.filter(g => g.callRate <= 500);
        }
    }
    
    // Apply sorting
    switch (sortBy) {
        case 'rating':
            filteredGeniuses.sort((a, b) => b.rating - a.rating);
            break;
        case 'iq':
            filteredGeniuses.sort((a, b) => b.iq - a.iq);
            break;
        case 'dmRate':
            filteredGeniuses.sort((a, b) => a.dmRate - b.dmRate);
            break;
        case 'callRate':
            filteredGeniuses.sort((a, b) => a.callRate - b.callRate);
            break;
    }
    
    // Re-render table
    const tableBody = document.getElementById('geniusesTableBody');
    if (tableBody) {
        tableBody.innerHTML = '';
        filteredGeniuses.forEach(genius => {
            const row = createGeniusTableRow(genius);
            tableBody.appendChild(row);
        });
    }
}

// Book a session
function bookSession(geniusId, serviceType) {
    const genius = dummyData.geniuses.find(g => g.id === geniusId);
    let rate = 0;
    let serviceName = '';
    
    switch (serviceType) {
        case 'dm':
            rate = genius.dmRate;
            serviceName = 'Direct Message';
            break;
        case 'voice':
            rate = genius.voiceRate;
            serviceName = 'Voice Message';
            break;
        case 'call':
            rate = genius.callRate;
            serviceName = '1-Hour Video Call';
            break;
    }
    
    alert(`Booking ${serviceName} with ${genius.name} for $${rate}. In a real app, this would redirect to payment.`);
}

// Modal functions
function openAskQuestionModal() {
    const modal = document.getElementById('askQuestionModal');
    if (modal) {
        modal.style.display = 'block';
    }
}

function closeAskQuestionModal() {
    const modal = document.getElementById('askQuestionModal');
    if (modal) {
        modal.style.display = 'none';
        const form = document.getElementById('questionForm');
        if (form) form.reset();
    }
}

// Setup event listeners for geniuses page
function setupGeniusesEventListeners() {
    // Close modal when clicking outside
    window.onclick = function(event) {
        const modal = document.getElementById('askQuestionModal');
        if (event.target === modal) {
            closeAskQuestionModal();
        }
    };
    
    // Handle question form submission
    const questionForm = document.getElementById('questionForm');
    if (questionForm) {
        questionForm.addEventListener('submit', handleQuestionSubmit);
    }
}

// Setup event listeners for home page
function setupHomeEventListeners() {
    // Close modal when clicking outside
    window.onclick = function(event) {
        const modal = document.getElementById('askQuestionModal');
        if (event.target === modal) {
            closeAskQuestionModal();
        }
    };
    
    // Handle question form submission
    const questionForm = document.getElementById('questionForm');
    if (questionForm) {
        questionForm.addEventListener('submit', handleQuestionSubmit);
    }
}

// Handle question form submission
function handleQuestionSubmit(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const questionData = {
        title: formData.get('title'),
        category: formData.get('category'),
        details: formData.get('details'),
        serviceType: formData.get('serviceType'),
        budget: formData.get('budget')
    };
    
    // In a real app, this would send data to backend
    console.log('New question:', questionData);
    
    // Show success message
    alert('Question submitted successfully! We\'ll match you with the best genius for your needs.');
    closeAskQuestionModal();
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init); 