// TPOT App JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initApp();
});

function initApp() {
    // Add event listeners for interactive elements
    setupPoastComposer();
    setupActionButtons();
}

function setupPoastComposer() {
    const textarea = document.querySelector('.compose-content textarea');
    const poastBtn = document.querySelector('.tweet-submit');
    
    if (textarea && poastBtn) {
        // Auto-resize textarea
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });
        
        // Character count and poast button state
        textarea.addEventListener('input', function() {
            const charCount = this.value.length;
            const maxChars = 280;
            
            if (charCount > maxChars) {
                this.style.color = '#e0245e';
                poastBtn.disabled = true;
                poastBtn.style.opacity = '0.5';
            } else if (charCount > 0) {
                this.style.color = '#0f1419';
                poastBtn.disabled = false;
                poastBtn.style.opacity = '1';
            } else {
                poastBtn.disabled = true;
                poastBtn.style.opacity = '0.5';
            }
        });
        
        // Poast submission
        poastBtn.addEventListener('click', function() {
            if (textarea.value.trim()) {
                createNewPoast(textarea.value);
                textarea.value = '';
                textarea.style.height = 'auto';
                poastBtn.disabled = true;
                poastBtn.style.opacity = '0.5';
            }
        });
    }
}

function setupActionButtons() {
    // Handle action button clicks
    document.addEventListener('click', function(e) {
        if (e.target.closest('.action-btn')) {
            const button = e.target.closest('.action-btn');
            const action = getActionType(button);
            const poast = button.closest('.tweet');
            
            if (poast && action) {
                handleAction(action, button, poast);
            }
        }
    });
}

function getActionType(button) {
    if (button.classList.contains('comment')) return 'comment';
    if (button.classList.contains('retweet')) return 'retweet';
    if (button.classList.contains('see-more')) return 'see-more';
    if (button.classList.contains('see-less')) return 'see-less';
    return null;
}

function handleAction(action, button, poast) {
    const countSpan = button.querySelector('span');
    let currentCount = parseInt(countSpan.textContent) || 0;
    
    switch (action) {
        case 'comment':
            // Simulate comment action
            button.style.color = '#1da1f2';
            setTimeout(() => {
                button.style.color = '#536471';
            }, 200);
            break;
            
        case 'retweet':
            // Toggle retweet state
            if (button.classList.contains('retweeted')) {
                button.classList.remove('retweeted');
                button.style.color = '#536471';
                currentCount--;
                button.innerHTML = '<i class="fas fa-retweet"></i><span>' + currentCount + '</span>';
            } else {
                button.classList.add('retweeted');
                button.style.color = '#00ba7c';
                currentCount++;
                button.innerHTML = '<i class="fas fa-retweet"></i><span>' + currentCount + '</span>';
            }
            break;
            
        case 'see-more':
            // Toggle see more state
            if (button.classList.contains('active')) {
                button.classList.remove('active');
                button.style.color = '#536471';
                button.innerHTML = '<i class="fas fa-plus"></i>';
            } else {
                button.classList.add('active');
                button.style.color = '#f91880';
                button.innerHTML = '<i class="fas fa-plus"></i>';
            }
            break;
            
        case 'see-less':
            // Toggle see less state
            if (button.classList.contains('active')) {
                button.classList.remove('active');
                button.style.color = '#536471';
                button.innerHTML = '<i class="fas fa-minus"></i>';
            } else {
                button.classList.add('active');
                button.style.color = '#f91880';
                button.innerHTML = '<i class="fas fa-minus"></i>';
            }
            break;
    }
}

function createNewPoast(content) {
    const poastsContainer = document.querySelector('.tweets');
    const currentUser = {
        name: 'You',
        username: '@you',
        avatar: '😊'
    };
    
    const poastHTML = `
        <article class="tweet" style="animation: slideIn 0.3s ease-out;">
            <div class="tweet-avatar">
                <span class="avatar-emoji">${currentUser.avatar}</span>
            </div>
            <div class="tweet-content">
                <div class="tweet-header">
                    <span class="tweet-name">${currentUser.name}</span>
                    <span class="tweet-username">${currentUser.username}</span>
                    <span class="tweet-time">now</span>
                </div>
                <p class="tweet-text">${escapeHTML(content)}</p>
                <div class="tweet-actions">
                    <button class="action-btn comment" title="Comment on this poast">
                        <i class="far fa-comment"></i>
                        <span>0</span>
                    </button>
                    <button class="action-btn retweet" title="Repost this">
                        <i class="fas fa-retweet"></i>
                        <span>0</span>
                    </button>
                    <button class="action-btn see-more" title="Show me more content like this">
                        <i class="fas fa-plus"></i>
                    </button>
                    <button class="action-btn see-less" title="Show me less content like this">
                        <i class="fas fa-minus"></i>
                    </button>
                </div>
            </div>
        </article>
    `;
    
    // Insert at the top of the feed
    poastsContainer.insertAdjacentHTML('afterbegin', poastHTML);
    
    // Add CSS animation
    if (!document.querySelector('#poast-animations')) {
        const style = document.createElement('style');
        style.id = 'poast-animations';
        style.textContent = `
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    // Show success message
    showNotification('Poast brewed successfully!');
}

function escapeHTML(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showNotification(message) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        background: #1da1f2;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 600;
        z-index: 10000;
        animation: slideInRight 0.3s ease-out;
    `;
    
    // Add animation CSS
    if (!document.querySelector('#notification-animations')) {
        const style = document.createElement('style');
        style.id = 'notification-animations';
        style.textContent = `
            @keyframes slideInRight {
                from {
                    opacity: 0;
                    transform: translateX(100%);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    document.body.appendChild(notification);
    
    // Remove notification after 3 seconds
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Export functions for potential external use
window.TPOTApp = {
    createNewPoast,
    showNotification
}; 