class VimEditor {
    constructor() {
        this.editor = null;
        this.currentFileName = 'untitled';
        this.currentLanguage = 'javascript';
        this.currentTheme = 'default';
        
        this.init();
    }
    
    init() {
        this.setupEditor();
        this.setupEventListeners();
        this.setupInitialContent();
    }
    
    setupEditor() {
        // Initialize CodeMirror with vim keybindings
        this.editor = CodeMirror.fromTextArea(document.getElementById('editor'), {
            lineNumbers: true,
            mode: this.currentLanguage,
            theme: this.currentTheme,
            keyMap: 'vim',
            vimMode: true,
            showCursorWhenSelecting: true,
            indentUnit: 4,
            tabSize: 4,
            indentWithTabs: false,
            lineWrapping: true,
            autoCloseBrackets: true,
            matchBrackets: true,
            foldGutter: true,
            gutters: ['CodeMirror-linenumbers', 'CodeMirror-foldgutter'],
            extraKeys: {
                'Ctrl-S': () => this.saveFile(),
                'Ctrl-O': () => this.loadFile(),
                'Ctrl-N': () => this.newFile(),
                'F1': () => this.showHelp(),
                'Esc': () => this.hideHelp()
            }
        });
        
        // Set initial size
        this.editor.setSize('100%', '100%');
        
        // Update status on changes
        this.editor.on('cursorActivity', () => this.updateStatus());
        this.editor.on('change', () => this.updateStatus());
        this.editor.on('vim-mode-change', (mode) => this.updateVimMode(mode));
    }
    
    setupEventListeners() {
        // Language selector
        document.getElementById('language-select').addEventListener('change', (e) => {
            this.changeLanguage(e.target.value);
        });
        
        // Theme selector
        document.getElementById('theme-select').addEventListener('change', (e) => {
            this.changeTheme(e.target.value);
        });
        
        // Control buttons
        document.getElementById('save-btn').addEventListener('click', () => this.saveFile());
        document.getElementById('load-btn').addEventListener('click', () => this.loadFile());
        document.getElementById('new-btn').addEventListener('click', () => this.newFile());
        
        // File input
        document.getElementById('file-input').addEventListener('change', (e) => {
            this.handleFileLoad(e.target.files[0]);
        });
        
        // Help modal
        document.getElementById('close-help').addEventListener('click', () => this.hideHelp());
        
        // Close modal on outside click
        document.getElementById('help-modal').addEventListener('click', (e) => {
            if (e.target.id === 'help-modal') {
                this.hideHelp();
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'F1') {
                e.preventDefault();
                this.showHelp();
            } else if (e.key === 'Escape') {
                this.hideHelp();
            }
        });
    }
    
    setupInitialContent() {
        const sampleContent = `// Welcome to Vim Editor with CodeMirror!
// This editor provides full vim keybindings with syntax highlighting

function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Try these vim commands:
// - Press 'i' to enter insert mode
// - Press 'Esc' to return to normal mode
// - Use h,j,k,l to navigate
// - Use dd to delete a line
// - Use yy to copy a line
// - Use p to paste
// - Use / to search
// - Use :w to save (or Ctrl+S)
// - Use :q to quit

const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(x => x * 2);

console.log("Hello, Vim Editor!");
console.log("Fibonacci(10) =", fibonacci(10));
console.log("Doubled numbers:", doubled);

/*
Multi-line comment
Try visual mode with 'v' or 'Shift+V'
Try different themes and languages from the header
*/
`;
        
        this.editor.setValue(sampleContent);
        this.updateStatus();
    }
    
    changeLanguage(language) {
        this.currentLanguage = language;
        
        // Map language names to CodeMirror modes
        const modeMap = {
            'javascript': 'javascript',
            'python': 'python',
            'html': 'xml',
            'css': 'css',
            'json': 'application/json',
            'xml': 'xml',
            'markdown': 'markdown',
            'sql': 'sql',
            'yaml': 'yaml',
            'text': 'text/plain'
        };
        
        const mode = modeMap[language] || 'text/plain';
        this.editor.setOption('mode', mode);
        
        // Update file extension
        const extensionMap = {
            'javascript': '.js',
            'python': '.py',
            'html': '.html',
            'css': '.css',
            'json': '.json',
            'xml': '.xml',
            'markdown': '.md',
            'sql': '.sql',
            'yaml': '.yml',
            'text': '.txt'
        };
        
        const baseName = this.currentFileName.split('.')[0];
        this.currentFileName = baseName + extensionMap[language];
        this.updateFileInfo();
    }
    
    changeTheme(theme) {
        this.currentTheme = theme;
        this.editor.setOption('theme', theme);
    }
    
    updateStatus() {
        const cursor = this.editor.getCursor();
        const line = cursor.line + 1;
        const ch = cursor.ch + 1;
        const content = this.editor.getValue();
        const charCount = content.length;
        const lineCount = this.editor.lineCount();
        
        // Update cursor position
        document.getElementById('cursor-position').textContent = `${line}:${ch}`;
        
        // Update file size
        document.getElementById('file-size').textContent = `${charCount} chars, ${lineCount} lines`;
        
        // Update file info
        this.updateFileInfo();
    }
    
    updateVimMode(mode) {
        const modeIndicator = document.getElementById('mode-indicator');
        const vimStatus = document.getElementById('vim-status');
        
        if (mode && mode.mode) {
            const modeName = mode.mode.toUpperCase();
            modeIndicator.textContent = modeName;
            modeIndicator.className = modeName.toLowerCase();
            
            if (mode.subMode) {
                vimStatus.textContent = `Vim ${modeName} mode (${mode.subMode})`;
            } else {
                vimStatus.textContent = `Vim ${modeName} mode`;
            }
        } else {
            modeIndicator.textContent = 'NORMAL';
            modeIndicator.className = 'normal';
            vimStatus.textContent = 'Vim mode enabled';
        }
    }
    
    updateFileInfo() {
        document.getElementById('file-info').textContent = this.currentFileName;
    }
    
    saveFile() {
        const content = this.editor.getValue();
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = this.currentFileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        // Show feedback
        this.showNotification('File saved successfully!');
    }
    
    loadFile() {
        document.getElementById('file-input').click();
    }
    
    handleFileLoad(file) {
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            this.editor.setValue(e.target.result);
            this.currentFileName = file.name;
            this.updateFileInfo();
            this.updateStatus();
            this.showNotification(`File "${file.name}" loaded successfully!`);
        };
        reader.readAsText(file);
    }
    
    newFile() {
        if (this.editor.getValue().trim() && !confirm('Are you sure you want to create a new file? Unsaved changes will be lost.')) {
            return;
        }
        
        this.editor.setValue('');
        this.currentFileName = 'untitled';
        this.updateFileInfo();
        this.updateStatus();
        this.showNotification('New file created!');
    }
    
    showHelp() {
        document.getElementById('help-modal').style.display = 'block';
    }
    
    hideHelp() {
        document.getElementById('help-modal').style.display = 'none';
    }
    
    showNotification(message) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = 'notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: #007acc;
            color: white;
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 14px;
            z-index: 1001;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            animation: slideIn 0.3s ease;
        `;
        
        // Add animation styles
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(notification);
        
        // Remove notification after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
                if (style.parentNode) {
                    style.parentNode.removeChild(style);
                }
            }, 300);
        }, 3000);
    }
}

// Initialize the editor when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new VimEditor();
});

// Handle window resize
window.addEventListener('resize', () => {
    if (window.vimEditor && window.vimEditor.editor) {
        window.vimEditor.editor.refresh();
    }
});

// Make editor globally accessible for debugging
window.addEventListener('load', () => {
    if (window.vimEditor) {
        window.vimEditor = window.vimEditor;
    }
});
