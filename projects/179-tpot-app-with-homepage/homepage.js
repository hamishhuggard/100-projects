// Tab switching functionality
document.addEventListener('DOMContentLoaded', function() {
    const navTabs = document.querySelectorAll('.nav-tab');
    const tabPanes = document.querySelectorAll('.tab-pane');

    // Add click event listeners to all nav tabs
    navTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const targetTab = this.getAttribute('data-tab');
            
            // Remove active class from all tabs and panes
            navTabs.forEach(t => t.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));
            
            // Add active class to clicked tab and corresponding pane
            this.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });

    // Add hover effects for interactive elements
    const interactiveElements = document.querySelectorAll('.post-link, .user-link, .archive-link');
    
    interactiveElements.forEach(element => {
        element.addEventListener('mouseenter', function() {
            this.style.transition = 'all 0.2s ease';
        });
    });

    // Add smooth scrolling for anchor links (if any)
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add loading animation for media items (placeholder for future functionality)
    const mediaItems = document.querySelectorAll('.media-item');
    
    mediaItems.forEach(item => {
        item.addEventListener('click', function() {
            // Add a subtle click effect
            this.style.transform = 'scale(0.98)';
            setTimeout(() => {
                this.style.transform = 'scale(1)';
            }, 150);
        });
    });

    // Add keyboard navigation support
    document.addEventListener('keydown', function(e) {
        const activeTab = document.querySelector('.nav-tab.active');
        const tabIndex = Array.from(navTabs).indexOf(activeTab);
        
        if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
            e.preventDefault();
            const prevTab = navTabs[tabIndex - 1] || navTabs[navTabs.length - 1];
            prevTab.click();
        } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
            e.preventDefault();
            const nextTab = navTabs[tabIndex + 1] || navTabs[0];
            nextTab.click();
        }
    });

    // Add focus management for accessibility
    navTabs.forEach(tab => {
        tab.addEventListener('focus', function() {
            this.style.outline = '2px solid #1da1f2';
            this.style.outlineOffset = '2px';
        });
        
        tab.addEventListener('blur', function() {
            this.style.outline = 'none';
        });
    });

    // Add smooth transitions for content changes
    tabPanes.forEach(pane => {
        pane.style.transition = 'opacity 0.3s ease';
    });

    // Optional: Add a subtle animation when the page loads
    window.addEventListener('load', function() {
        const sidebar = document.querySelector('.sidebar');
        const contentArea = document.querySelector('.content-area');
        
        if (sidebar && contentArea) {
            sidebar.style.opacity = '0';
            contentArea.style.opacity = '0';
            
            setTimeout(() => {
                sidebar.style.transition = 'opacity 0.5s ease';
                contentArea.style.transition = 'opacity 0.5s ease';
                sidebar.style.opacity = '1';
                contentArea.style.opacity = '1';
            }, 100);
        }
    });
}); 