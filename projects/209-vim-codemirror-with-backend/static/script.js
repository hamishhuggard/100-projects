class CollaborativeVimEditor {
    constructor() {
        this.editor = null;
        this.socket = null;
        this.userId = null;
        this.userInfo = null;
        this.otherUsers = new Map();
        this.isConnected = false;
        this.isUpdatingFromServer = false;
        
        this.init();
    }
    
    init() {
        this.setupSocket();
        this.setupEditor();
        this.setupEventListeners();
    }
    
    setupSocket() {
        // Connect to Flask-SocketIO server
        this.socket = io();
        
        this.socket.on('connect', () => {
            this.isConnected = true;
            this.updateConnectionStatus('Connected');
            console.log('Connected to server');
        });
        
        this.socket.on('disconnect', () => {
            this.isConnected = false;
            this.updateConnectionStatus('Disconnected');
            console.log('Disconnected from server');
        });
        
        this.socket.on('document_state', (data) => {
            this.handleDocumentState(data);
        });
        
        this.socket.on('text_change', (data) => {
            this.handleTextChange(data);
        });
        
        this.socket.on('cursor_change', (data) => {
            this.handleCursorChange(data);
        });
        
        this.socket.on('language_change', (data) => {
            this.handleLanguageChange(data);
        });
        
        this.socket.on('theme_change', (data) => {
            this.handleThemeChange(data);
        });
        
        this.socket.on('vim_mode_change', (data) => {
            this.handleVimModeChange(data);
        });
        
        this.socket.on('user_joined', (data) => {
            this.handleUserJoined(data);
        });
        
        this.socket.on('user_left', (data) => {
            this.handleUserLeft(data);
        });
        
        this.socket.on('user_info_update', (data) => {
            this.handleUserInfoUpdate(data);
        });
    }
    
    setupEditor() {
        // Initialize CodeMirror with vim keybindings
        this.editor = CodeMirror.fromTextArea(document.getElementById('editor'), {
            lineNumbers: true,
            mode: 'javascript',
            theme: 'default',
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
        this.editor.on('change', () => this.handleLocalChange());
        this.editor.on('vim-mode-change', (mode) => this.handleLocalVimModeChange(mode));
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
        document.getElementById('users-btn').addEventListener('click', () => this.showUsers());
        
        // File input
        document.getElementById('file-input').addEventListener('change', (e) => {
            this.handleFileLoad(e.target.files[0]);
        });
        
        // Modals
        document.getElementById('close-help').addEventListener('click', () => this.hideHelp());
        document.getElementById('close-users').addEventListener('click', () => this.hideUsers());
        
        // Close modals on outside click
        document.getElementById('help-modal').addEventListener('click', (e) => {
            if (e.target.id === 'help-modal') {
                this.hideHelp();
            }
        });
        
        document.getElementById('users-modal').addEventListener('click', (e) => {
            if (e.target.id === 'users-modal') {
                this.hideUsers();
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'F1') {
                e.preventDefault();
                this.showHelp();
            } else if (e.key === 'Escape') {
                this.hideHelp();
                this.hideUsers();
            }
        });
    }
    
    // Socket event handlers
    handleDocumentState(data) {
        this.isUpdatingFromServer = true;
        
        // Update editor content
        this.editor.setValue(data.content);
        
        // Update language and theme
        this.changeLanguage(data.language, false);
        this.changeTheme(data.theme, false);
        
        // Update other users' cursors
        this.otherUsers.clear();
        for (const [userId, cursor] of Object.entries(data.cursors)) {
            if (userId !== this.userId) {
                this.otherUsers.set(userId, {
                    cursor: cursor,
                    userInfo: data.users[userId]
                });
            }
        }
        
        this.updateOtherUsersCursors();
        this.updateUsersList(data.users);
        
        this.isUpdatingFromServer = false;
        this.updateStatus();
    }
    
    handleTextChange(data) {
        if (data.user_id !== this.userId) {
            this.isUpdatingFromServer = true;
            this.editor.setValue(data.content);
            this.isUpdatingFromServer = false;
            this.updateStatus();
        }
    }
    
    handleCursorChange(data) {
        if (data.user_id !== this.userId) {
            this.otherUsers.set(data.user_id, {
                cursor: { line: data.line, ch: data.ch },
                userInfo: data.user_info
            });
            this.updateOtherUsersCursors();
        }
    }
    
    handleLanguageChange(data) {
        if (data.user_id !== this.userId) {
            this.changeLanguage(data.language, false);
        }
    }
    
    handleThemeChange(data) {
        if (data.user_id !== this.userId) {
            this.changeTheme(data.theme, false);
        }
    }
    
    handleVimModeChange(data) {
        // Could show other users' vim modes in the future
        console.log(`User ${data.user_id} changed to ${data.mode} mode`);
    }
    
    handleUserJoined(data) {
        this.otherUsers.set(data.id, {
            cursor: { line: 0, ch: 0 },
            userInfo: data
        });
        this.updateOtherUsersCursors();
        this.showNotification(`${data.name} joined the editor`);
    }
    
    handleUserLeft(data) {
        this.otherUsers.delete(data.user_id);
        this.updateOtherUsersCursors();
        this.showNotification('A user left the editor');
    }
    
    handleUserInfoUpdate(data) {
        if (this.otherUsers.has(data.user_id)) {
            const user = this.otherUsers.get(data.user_id);
            user.userInfo = data.user_info;
            this.updateOtherUsersCursors();
        }
    }
    
    // Local event handlers
    handleLocalChange() {
        if (!this.isUpdatingFromServer && this.isConnected) {
            const content = this.editor.getValue();
            this.socket.emit('text_change', { content: content });
        }
        this.updateStatus();
    }
    
    handleLocalVimModeChange(mode) {
        if (this.isConnected) {
            this.socket.emit('vim_mode_change', { mode: mode });
        }
        this.updateVimMode(mode);
    }
    
    // Editor methods
    changeLanguage(language, emit = true) {
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
        
        if (emit && this.isConnected) {
            this.socket.emit('language_change', { language: language });
        }
        
        // Update selector
        document.getElementById('language-select').value = language;
    }
    
    changeTheme(theme, emit = true) {
        this.editor.setOption('theme', theme);
        
        if (emit && this.isConnected) {
            this.socket.emit('theme_change', { theme: theme });
        }
        
        // Update selector
        document.getElementById('theme-select').value = theme;
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
        
        // Send cursor position to server
        if (this.isConnected && !this.isUpdatingFromServer) {
            this.socket.emit('cursor_change', {
                line: cursor.line,
                ch: cursor.ch
            });
        }
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
    
    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connection-status');
        statusElement.textContent = status;
        statusElement.className = status.toLowerCase();
    }
    
    updateOtherUsersCursors() {
        // Remove existing cursor markers
        this.editor.getAllMarks().forEach(mark => {
            if (mark.className && mark.className.includes('other-user-cursor')) {
                mark.clear();
            }
        });
        
        // Add new cursor markers
        this.otherUsers.forEach((user, userId) => {
            const cursor = user.cursor;
            const userInfo = user.userInfo;
            
            if (cursor && userInfo) {
                const marker = this.editor.setBookmark(
                    { line: cursor.line, ch: cursor.ch },
                    {
                        widget: this.createCursorWidget(userInfo),
                        insertLeft: true,
                        className: 'other-user-cursor'
                    }
                );
            }
        });
    }
    
    createCursorWidget(userInfo) {
        const widget = document.createElement('div');
        widget.className = 'other-user-cursor-widget';
        widget.style.cssText = `
            position: relative;
            display: inline-block;
            width: 2px;
            height: 20px;
            background-color: ${userInfo.color};
            border-radius: 1px;
            z-index: 10;
        `;
        
        const label = document.createElement('div');
        label.textContent = userInfo.name;
        label.style.cssText = `
            position: absolute;
            top: -25px;
            left: 0;
            background-color: ${userInfo.color};
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            white-space: nowrap;
            z-index: 11;
        `;
        
        widget.appendChild(label);
        return widget;
    }
    
    updateUsersList(users) {
        const usersList = document.getElementById('users-list');
        usersList.innerHTML = '';
        
        Object.values(users).forEach(user => {
            const userElement = document.createElement('div');
            userElement.className = 'user-item';
            userElement.style.cssText = `
                display: flex;
                align-items: center;
                padding: 8px;
                margin: 4px 0;
                background-color: #3c3c3c;
                border-radius: 4px;
                border-left: 4px solid ${user.color};
            `;
            
            userElement.innerHTML = `
                <div style="width: 12px; height: 12px; background-color: ${user.color}; border-radius: 50%; margin-right: 8px;"></div>
                <div>
                    <div style="font-weight: bold;">${user.name}</div>
                    <div style="font-size: 11px; color: #888;">Connected ${new Date(user.connected_at).toLocaleTimeString()}</div>
                </div>
            `;
            
            usersList.appendChild(userElement);
        });
    }
    
    // File operations
    saveFile() {
        const content = this.editor.getValue();
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = 'file.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
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
            this.showNotification(`File "${file.name}" loaded successfully!`);
        };
        reader.readAsText(file);
    }
    
    newFile() {
        if (this.editor.getValue().trim() && !confirm('Are you sure you want to create a new file? This will clear the current content.')) {
            return;
        }
        
        this.editor.setValue('');
        this.showNotification('New file created!');
    }
    
    // UI methods
    showHelp() {
        document.getElementById('help-modal').style.display = 'block';
    }
    
    hideHelp() {
        document.getElementById('help-modal').style.display = 'none';
    }
    
    showUsers() {
        document.getElementById('users-modal').style.display = 'block';
    }
    
    hideUsers() {
        document.getElementById('users-modal').style.display = 'none';
    }
    
    showNotification(message) {
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
    window.collaborativeEditor = new CollaborativeVimEditor();
});

// Handle window resize
window.addEventListener('resize', () => {
    if (window.collaborativeEditor && window.collaborativeEditor.editor) {
        window.collaborativeEditor.editor.refresh();
    }
});