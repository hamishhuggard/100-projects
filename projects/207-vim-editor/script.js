class VimEditor {
    constructor() {
        this.editor = document.getElementById('editor');
        this.modeIndicator = document.getElementById('mode-indicator');
        this.fileInfo = document.getElementById('file-info');
        this.positionInfo = document.getElementById('position-info');
        this.commandLine = document.getElementById('command-line');
        this.commandInput = document.getElementById('command-input');
        this.commandText = document.getElementById('command-text');
        
        this.mode = 'normal'; // normal, insert, visual
        this.fileName = 'untitled';
        this.isModified = false;
        this.commandBuffer = '';
        this.searchQuery = '';
        this.searchResults = [];
        this.currentSearchIndex = -1;
        this.clipboard = '';
        this.undoStack = [];
        this.redoStack = [];
        this.visualSelectionStart = null;
        this.visualSelectionEnd = null;
        this.visualLineMode = false;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.updateStatusBar();
        this.saveState();
        
        // Add some sample content
        this.editor.textContent = `Welcome to Vim Editor!

This is a web-based vim-like text editor.

Basic commands:
- Press 'i' to enter insert mode
- Press 'Esc' to return to normal mode
- Use h,j,k,l to navigate
- Type ':help' for more commands

Try editing this text!`;
        
        // Apply syntax highlighting
        this.applySyntaxHighlighting();
    }
    
    setupEventListeners() {
        this.editor.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.editor.addEventListener('input', (e) => this.handleInput(e));
        this.commandText.addEventListener('keydown', (e) => this.handleCommandKeyDown(e));
        
        // Prevent default browser shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && ['s', 'o', 'n'].includes(e.key.toLowerCase())) {
                e.preventDefault();
            }
        });
    }
    
    handleKeyDown(e) {
        if (this.mode === 'command') return;
        
        // Handle mode-specific key events
        if (this.mode === 'normal') {
            this.handleNormalMode(e);
        } else if (this.mode === 'insert') {
            this.handleInsertMode(e);
        } else if (this.mode === 'visual') {
            this.handleVisualMode(e);
        }
    }
    
    handleInput(e) {
        if (this.mode === 'normal' || this.mode === 'visual') {
            e.preventDefault();
            return;
        }
        
        this.isModified = true;
        this.updateStatusBar();
        
        // Apply syntax highlighting after input
        setTimeout(() => this.applySyntaxHighlighting(), 0);
    }
    
    handleNormalMode(e) {
        e.preventDefault();
        
        const key = e.key;
        const ctrlKey = e.ctrlKey;
        
        // Mode switching
        if (key === 'i') {
            this.setMode('insert');
            return;
        } else if (key === 'a') {
            this.moveCursorRight();
            this.setMode('insert');
            return;
        } else if (key === 'o') {
            this.insertNewLine();
            this.setMode('insert');
            return;
        } else if (key === 'O') {
            this.insertNewLineAbove();
            this.setMode('insert');
            return;
        } else if (key === 'v') {
            this.startVisualMode();
            return;
        } else if (key === 'V' || (e.shiftKey && key === 'v')) {
            this.startVisualLineMode();
            return;
        }
        
        // Navigation
        if (key === 'h' || key === 'ArrowLeft') {
            this.moveCursorLeft();
        } else if (key === 'j' || key === 'ArrowDown') {
            this.moveCursorDown();
        } else if (key === 'k' || key === 'ArrowUp') {
            this.moveCursorUp();
        } else if (key === 'l' || key === 'ArrowRight') {
            this.moveCursorRight();
        } else if (key === 'w') {
            this.moveToNextWord();
        } else if (key === 'b') {
            this.moveToPreviousWord();
        } else if (key === '0') {
            this.moveToLineStart();
        } else if (key === '^') {
            this.moveToLineStartNonBlank();
        } else if (key === '$') {
            this.moveToLineEnd();
        } else if (key === 'g' && this.commandBuffer === 'g') {
            this.moveToFileStart();
            this.commandBuffer = '';
        } else if (key === 'G') {
            this.moveToFileEnd();
        } else if (key === 'g') {
            this.commandBuffer = 'g';
            return;
        }
        
        // Editing commands
        else if (key === 'd' && this.commandBuffer === 'd') {
            this.deleteLine();
            this.commandBuffer = '';
        } else if (key === 'd') {
            this.commandBuffer = 'd';
            return;
        } else if (key === 'y' && this.commandBuffer === 'y') {
            this.yankLine();
            this.commandBuffer = '';
        } else if (key === 'y') {
            this.commandBuffer = 'y';
            return;
        } else if (key === 'p') {
            this.pasteAfter();
        } else if (key === 'P') {
            this.pasteBefore();
        } else if (key === 'u') {
            this.undo();
        } else if (ctrlKey && key === 'r') {
            this.redo();
        } else if (key === 'x') {
            this.deleteChar();
        } else if (key === 'r') {
            this.commandBuffer = 'r';
            return;
        }
        
        // Search
        else if (key === '/') {
            this.startSearch();
        } else if (key === 'n') {
            this.nextSearch();
        } else if (key === 'N') {
            this.previousSearch();
        }
        
        // Commands
        else if (key === ':') {
            this.startCommand();
        }
        
        // Clear command buffer for other keys
        else {
            this.commandBuffer = '';
        }
        
        this.updateStatusBar();
    }
    
    handleInsertMode(e) {
        if (e.key === 'Escape' || (e.ctrlKey && e.key === 'c')) {
            e.preventDefault();
            this.setMode('normal');
        }
    }
    
    handleVisualMode(e) {
        e.preventDefault();
        
        if (e.key === 'Escape' || (e.ctrlKey && e.key === 'c')) {
            this.endVisualMode();
            return;
        }
        
        // Visual mode commands
        if (e.key === 'y') {
            this.yankVisualSelection();
            this.endVisualMode();
            return;
        } else if (e.key === 'd') {
            this.deleteVisualSelection();
            this.endVisualMode();
            return;
        } else if (e.key === 'c') {
            this.deleteVisualSelection();
            this.setMode('insert');
            return;
        }
        
        // Navigation in visual mode
        if (this.visualLineMode) {
            // In visual line mode, only j/k navigation changes selection
            if (e.key === 'j' || e.key === 'ArrowDown') {
                this.moveCursorDown();
                this.updateVisualLineSelection();
            } else if (e.key === 'k' || e.key === 'ArrowUp') {
                this.moveCursorUp();
                this.updateVisualLineSelection();
            }
        } else {
            // Regular visual mode navigation
            if (e.key === 'h' || e.key === 'ArrowLeft') {
                this.moveCursorLeft();
            } else if (e.key === 'j' || e.key === 'ArrowDown') {
                this.moveCursorDown();
            } else if (e.key === 'k' || e.key === 'ArrowUp') {
                this.moveCursorUp();
            } else if (e.key === 'l' || e.key === 'ArrowRight') {
                this.moveCursorRight();
            }
        }
        
        // Update visual selection
        this.updateVisualSelection();
        this.updateStatusBar();
    }
    
    handleCommandKeyDown(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            this.executeCommand(this.commandText.value);
            this.commandText.value = '';
            this.hideCommandLine();
        } else if (e.key === 'Escape') {
            e.preventDefault();
            this.commandText.value = '';
            this.hideCommandLine();
        }
    }
    
    setMode(mode) {
        this.mode = mode;
        this.updateModeIndicator();
        
        if (mode === 'insert') {
            this.editor.focus();
        } else if (mode === 'command') {
            this.commandText.focus();
        }
    }
    
    startVisualMode() {
        this.setMode('visual');
        this.visualLineMode = false;
        this.visualSelectionStart = this.getCursorPosition();
        this.visualSelectionEnd = this.getCursorPosition();
        this.updateVisualSelection();
    }
    
    startVisualLineMode() {
        this.setMode('visual');
        this.visualLineMode = true;
        const currentPos = this.getCursorPosition();
        this.visualSelectionStart = { line: currentPos.line, column: 1 };
        this.visualSelectionEnd = { line: currentPos.line, column: this.getLineLength(currentPos.line) + 1 };
        this.updateVisualSelection();
    }
    
    endVisualMode() {
        this.clearVisualSelection();
        this.setMode('normal');
        this.visualSelectionStart = null;
        this.visualSelectionEnd = null;
        this.visualLineMode = false;
    }
    
    updateVisualSelection() {
        this.clearVisualSelection();
        
        if (this.mode === 'visual' && this.visualSelectionStart && this.visualSelectionEnd) {
            const currentPos = this.getCursorPosition();
            this.visualSelectionEnd = currentPos;
            
            // Create selection range
            const startPos = this.visualSelectionStart;
            const endPos = this.visualSelectionEnd;
            
            // Ensure start is before end
            const actualStart = this.comparePositions(startPos, endPos) <= 0 ? startPos : endPos;
            const actualEnd = this.comparePositions(startPos, endPos) <= 0 ? endPos : startPos;
            
            this.highlightSelection(actualStart, actualEnd);
        }
    }
    
    clearVisualSelection() {
        // Remove any existing visual selection highlighting
        const selectionElements = this.editor.querySelectorAll('.visual-selection');
        selectionElements.forEach(el => {
            el.classList.remove('visual-selection');
        });
    }
    
    highlightSelection(startPos, endPos) {
        // This is a simplified version - in a real implementation you'd need
        // more sophisticated text range handling
        const text = this.editor.textContent;
        const startIndex = this.positionToIndex(startPos);
        const endIndex = this.positionToIndex(endPos);
        
        if (startIndex < endIndex) {
            const beforeText = text.substring(0, startIndex);
            const selectedText = text.substring(startIndex, endIndex);
            const afterText = text.substring(endIndex);
            
            // Temporarily disable syntax highlighting to avoid conflicts
            const originalContent = this.editor.innerHTML;
            this.editor.innerHTML = this.escapeHtml(beforeText) + 
                                  `<span class="visual-selection">${this.escapeHtml(selectedText)}</span>` + 
                                  this.escapeHtml(afterText);
        }
    }
    
    positionToIndex(pos) {
        const lines = this.editor.textContent.split('\n');
        let index = 0;
        
        for (let i = 0; i < pos.line - 1; i++) {
            index += lines[i].length + 1; // +1 for newline
        }
        index += pos.column - 1;
        
        return Math.min(index, this.editor.textContent.length);
    }
    
    indexToPosition(index) {
        const text = this.editor.textContent;
        const lines = text.substring(0, index).split('\n');
        return {
            line: lines.length,
            column: lines[lines.length - 1].length + 1
        };
    }
    
    comparePositions(pos1, pos2) {
        if (pos1.line !== pos2.line) {
            return pos1.line - pos2.line;
        }
        return pos1.column - pos2.column;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    yankVisualSelection() {
        if (this.visualSelectionStart && this.visualSelectionEnd) {
            const startPos = this.visualSelectionStart;
            const endPos = this.visualSelectionEnd;
            const actualStart = this.comparePositions(startPos, endPos) <= 0 ? startPos : endPos;
            const actualEnd = this.comparePositions(startPos, endPos) <= 0 ? endPos : startPos;
            
            const startIndex = this.positionToIndex(actualStart);
            const endIndex = this.positionToIndex(actualEnd);
            const selectedText = this.editor.textContent.substring(startIndex, endIndex);
            
            this.clipboard = selectedText;
        }
    }
    
    deleteVisualSelection() {
        if (this.visualSelectionStart && this.visualSelectionEnd) {
            const startPos = this.visualSelectionStart;
            const endPos = this.visualSelectionEnd;
            const actualStart = this.comparePositions(startPos, endPos) <= 0 ? startPos : endPos;
            const actualEnd = this.comparePositions(startPos, endPos) <= 0 ? endPos : startPos;
            
            const startIndex = this.positionToIndex(actualStart);
            const endIndex = this.positionToIndex(actualEnd);
            
            const beforeText = this.editor.textContent.substring(0, startIndex);
            const afterText = this.editor.textContent.substring(endIndex);
            
            this.editor.textContent = beforeText + afterText;
            this.setCursorPosition(actualStart.line, actualStart.column);
            this.isModified = true;
            this.updateStatusBar();
        }
    }
    
    updateVisualLineSelection() {
        if (this.visualLineMode) {
            const currentPos = this.getCursorPosition();
            const startLine = this.visualSelectionStart.line;
            const endLine = currentPos.line;
            
            const actualStartLine = Math.min(startLine, endLine);
            const actualEndLine = Math.max(startLine, endLine);
            
            this.visualSelectionStart = { line: actualStartLine, column: 1 };
            this.visualSelectionEnd = { 
                line: actualEndLine, 
                column: this.getLineLength(actualEndLine) + 1 
            };
        }
    }
    
    getLineLength(lineNumber) {
        const lines = this.editor.textContent.split('\n');
        if (lineNumber > 0 && lineNumber <= lines.length) {
            return lines[lineNumber - 1].length;
        }
        return 0;
    }
    
    updateModeIndicator() {
        this.modeIndicator.textContent = this.mode.toUpperCase();
        this.modeIndicator.className = this.mode;
    }
    
    updateStatusBar() {
        const position = this.getCursorPosition();
        this.positionInfo.textContent = `${position.line},${position.column}`;
        
        const modified = this.isModified ? ' [+]' : '';
        this.fileInfo.textContent = this.fileName + modified;
    }
    
    getCursorPosition() {
        const selection = window.getSelection();
        const range = selection.getRangeAt(0);
        const preCaretRange = range.cloneRange();
        preCaretRange.selectNodeContents(this.editor);
        preCaretRange.setEnd(range.endContainer, range.endOffset);
        
        const text = preCaretRange.toString();
        const lines = text.split('\n');
        const line = lines.length;
        const column = lines[lines.length - 1].length + 1;
        
        return { line, column };
    }
    
    // Navigation methods
    moveCursorLeft() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            range.setStart(range.startContainer, Math.max(0, range.startOffset - 1));
            range.collapse(true);
            selection.removeAllRanges();
            selection.addRange(range);
        }
    }
    
    moveCursorRight() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            const textNode = range.startContainer;
            if (textNode.nodeType === Node.TEXT_NODE) {
                if (range.startOffset < textNode.textContent.length) {
                    range.setStart(textNode, range.startOffset + 1);
                    range.collapse(true);
                    selection.removeAllRanges();
                    selection.addRange(range);
                }
            }
        }
    }
    
    moveCursorUp() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            const rect = range.getBoundingClientRect();
            const editorRect = this.editor.getBoundingClientRect();
            
            // Simple line-based movement
            const lines = this.editor.textContent.split('\n');
            const currentPos = this.getCursorPosition();
            if (currentPos.line > 1) {
                const targetLine = currentPos.line - 1;
                const targetColumn = Math.min(currentPos.column, lines[targetLine - 1].length + 1);
                this.setCursorPosition(targetLine, targetColumn);
            }
        }
    }
    
    moveCursorDown() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const lines = this.editor.textContent.split('\n');
            const currentPos = this.getCursorPosition();
            if (currentPos.line < lines.length) {
                const targetLine = currentPos.line + 1;
                const targetColumn = Math.min(currentPos.column, lines[targetLine - 1].length + 1);
                this.setCursorPosition(targetLine, targetColumn);
            }
        }
    }
    
    setCursorPosition(line, column) {
        const lines = this.editor.textContent.split('\n');
        let charIndex = 0;
        
        for (let i = 0; i < line - 1; i++) {
            charIndex += lines[i].length + 1; // +1 for newline
        }
        charIndex += column - 1;
        
        const range = document.createRange();
        const selection = window.getSelection();
        
        if (this.editor.firstChild) {
            range.setStart(this.editor.firstChild, Math.min(charIndex, this.editor.textContent.length));
            range.collapse(true);
            selection.removeAllRanges();
            selection.addRange(range);
        }
    }
    
    moveToNextWord() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            const text = this.editor.textContent;
            const currentPos = range.startOffset;
            
            // Find next word boundary
            let pos = currentPos;
            while (pos < text.length && /\s/.test(text[pos])) {
                pos++;
            }
            while (pos < text.length && !/\s/.test(text[pos])) {
                pos++;
            }
            
            if (pos < text.length) {
                range.setStart(this.editor.firstChild, pos);
                range.collapse(true);
                selection.removeAllRanges();
                selection.addRange(range);
            }
        }
    }
    
    moveToPreviousWord() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            const text = this.editor.textContent;
            const currentPos = range.startOffset;
            
            // Find previous word boundary
            let pos = currentPos - 1;
            while (pos > 0 && /\s/.test(text[pos])) {
                pos--;
            }
            while (pos > 0 && !/\s/.test(text[pos])) {
                pos--;
            }
            
            range.setStart(this.editor.firstChild, pos);
            range.collapse(true);
            selection.removeAllRanges();
            selection.addRange(range);
        }
    }
    
    moveToLineStart() {
        const currentPos = this.getCursorPosition();
        this.setCursorPosition(currentPos.line, 1);
    }
    
    moveToLineStartNonBlank() {
        const currentPos = this.getCursorPosition();
        const lines = this.editor.textContent.split('\n');
        const line = lines[currentPos.line - 1];
        
        // Find first non-whitespace character
        let column = 1;
        for (let i = 0; i < line.length; i++) {
            if (!/\s/.test(line[i])) {
                column = i + 1;
                break;
            }
        }
        
        this.setCursorPosition(currentPos.line, column);
    }
    
    moveToLineEnd() {
        const lines = this.editor.textContent.split('\n');
        const currentPos = this.getCursorPosition();
        const lineLength = lines[currentPos.line - 1].length;
        this.setCursorPosition(currentPos.line, lineLength + 1);
    }
    
    moveToFileStart() {
        this.setCursorPosition(1, 1);
    }
    
    moveToFileEnd() {
        const lines = this.editor.textContent.split('\n');
        const lastLine = lines.length;
        const lastLineLength = lines[lastLine - 1].length;
        this.setCursorPosition(lastLine, lastLineLength + 1);
    }
    
    // Editing methods
    insertNewLine() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            range.insertNode(document.createTextNode('\n'));
            range.collapse(false);
            selection.removeAllRanges();
            selection.addRange(range);
        }
    }
    
    insertNewLineAbove() {
        const currentPos = this.getCursorPosition();
        const lines = this.editor.textContent.split('\n');
        lines.splice(currentPos.line - 1, 0, '');
        this.editor.textContent = lines.join('\n');
        this.setCursorPosition(currentPos.line, 1);
    }
    
    deleteLine() {
        const currentPos = this.getCursorPosition();
        const lines = this.editor.textContent.split('\n');
        if (lines.length > 1) {
            lines.splice(currentPos.line - 1, 1);
            this.editor.textContent = lines.join('\n');
            this.setCursorPosition(Math.min(currentPos.line, lines.length), 1);
        } else {
            this.editor.textContent = '';
        }
        this.isModified = true;
        this.updateStatusBar();
    }
    
    yankLine() {
        const currentPos = this.getCursorPosition();
        const lines = this.editor.textContent.split('\n');
        this.clipboard = lines[currentPos.line - 1] + '\n';
    }
    
    pasteAfter() {
        if (this.clipboard) {
            const selection = window.getSelection();
            if (selection.rangeCount > 0) {
                const range = selection.getRangeAt(0);
                range.insertNode(document.createTextNode(this.clipboard));
                range.collapse(false);
                selection.removeAllRanges();
                selection.addRange(range);
                this.isModified = true;
                this.updateStatusBar();
            }
        }
    }
    
    pasteBefore() {
        if (this.clipboard) {
            const selection = window.getSelection();
            if (selection.rangeCount > 0) {
                const range = selection.getRangeAt(0);
                range.insertNode(document.createTextNode(this.clipboard));
                range.collapse(true);
                selection.removeAllRanges();
                selection.addRange(range);
                this.isModified = true;
                this.updateStatusBar();
            }
        }
    }
    
    deleteChar() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            const textNode = range.startContainer;
            if (textNode.nodeType === Node.TEXT_NODE && range.startOffset < textNode.textContent.length) {
                const char = textNode.textContent[range.startOffset];
                textNode.textContent = textNode.textContent.slice(0, range.startOffset) + 
                                     textNode.textContent.slice(range.startOffset + 1);
                this.isModified = true;
                this.updateStatusBar();
            }
        }
    }
    
    undo() {
        if (this.undoStack.length > 0) {
            const state = this.undoStack.pop();
            this.redoStack.push({
                content: this.editor.textContent,
                cursor: this.getCursorPosition()
            });
            this.editor.textContent = state.content;
            this.setCursorPosition(state.cursor.line, state.cursor.column);
            this.isModified = true;
            this.updateStatusBar();
        }
    }
    
    redo() {
        if (this.redoStack.length > 0) {
            const state = this.redoStack.pop();
            this.undoStack.push({
                content: this.editor.textContent,
                cursor: this.getCursorPosition()
            });
            this.editor.textContent = state.content;
            this.setCursorPosition(state.cursor.line, state.cursor.column);
            this.isModified = true;
            this.updateStatusBar();
        }
    }
    
    saveState() {
        this.undoStack.push({
            content: this.editor.textContent,
            cursor: this.getCursorPosition()
        });
        
        // Limit undo stack size
        if (this.undoStack.length > 50) {
            this.undoStack.shift();
        }
    }
    
    // Command methods
    startCommand() {
        this.setMode('command');
        this.showCommandLine();
    }
    
    showCommandLine() {
        this.commandInput.style.display = 'flex';
        this.commandText.focus();
    }
    
    hideCommandLine() {
        this.commandInput.style.display = 'none';
        this.setMode('normal');
    }
    
    executeCommand(command) {
        const parts = command.trim().split(' ');
        const cmd = parts[0];
        
        switch (cmd) {
            case 'w':
            case 'write':
                this.saveFile();
                break;
            case 'q':
            case 'quit':
                if (this.isModified) {
                    this.commandLine.textContent = 'No write since last change (add ! to override)';
                } else {
                    this.quit();
                }
                break;
            case 'wq':
                this.saveFile();
                this.quit();
                break;
            case 'q!':
                this.quit();
                break;
            case 'help':
                this.showHelp();
                break;
            case 'new':
                this.newFile();
                break;
            case 'set':
                if (parts.length > 1) {
                    this.setFileName(parts[1]);
                } else {
                    this.commandLine.textContent = 'Usage: :set filename.ext';
                }
                break;
            default:
                this.commandLine.textContent = `Unknown command: ${cmd}`;
        }
    }
    
    saveFile() {
        // In a real implementation, this would save to a file
        // For now, we'll just mark as saved
        this.isModified = false;
        this.updateStatusBar();
        this.commandLine.textContent = `"${this.fileName}" written`;
    }
    
    newFile() {
        if (this.isModified) {
            this.commandLine.textContent = 'No write since last change (add ! to override)';
            return;
        }
        this.editor.textContent = '';
        this.fileName = 'untitled';
        this.isModified = false;
        this.updateStatusBar();
    }
    
    quit() {
        // In a real implementation, this would close the editor
        this.commandLine.textContent = 'Quit (use browser close button)';
    }
    
    setFileName(filename) {
        this.fileName = filename;
        this.updateStatusBar();
        this.applySyntaxHighlighting();
        this.commandLine.textContent = `File type set to: ${filename}`;
    }
    
    // Search methods
    startSearch() {
        const query = prompt('Search:');
        if (query) {
            this.searchQuery = query;
            this.performSearch();
        }
    }
    
    performSearch() {
        const text = this.editor.textContent;
        this.searchResults = [];
        
        let index = 0;
        while ((index = text.indexOf(this.searchQuery, index)) !== -1) {
            this.searchResults.push(index);
            index += this.searchQuery.length;
        }
        
        this.currentSearchIndex = 0;
        if (this.searchResults.length > 0) {
            this.jumpToSearchResult(0);
        }
    }
    
    nextSearch() {
        if (this.searchResults.length > 0) {
            this.currentSearchIndex = (this.currentSearchIndex + 1) % this.searchResults.length;
            this.jumpToSearchResult(this.currentSearchIndex);
        }
    }
    
    previousSearch() {
        if (this.searchResults.length > 0) {
            this.currentSearchIndex = (this.currentSearchIndex - 1 + this.searchResults.length) % this.searchResults.length;
            this.jumpToSearchResult(this.currentSearchIndex);
        }
    }
    
    jumpToSearchResult(index) {
        if (index >= 0 && index < this.searchResults.length) {
            const charIndex = this.searchResults[index];
            const lines = this.editor.textContent.substring(0, charIndex).split('\n');
            const line = lines.length;
            const column = lines[lines.length - 1].length + 1;
            this.setCursorPosition(line, column);
        }
    }
    
    showHelp() {
        document.getElementById('help-modal').style.display = 'flex';
    }
    
    // Syntax highlighting methods
    applySyntaxHighlighting() {
        const text = this.editor.textContent;
        const cursorPos = this.getCursorPosition();
        
        // Save cursor position
        const selection = window.getSelection();
        let range = null;
        if (selection.rangeCount > 0) {
            range = selection.getRangeAt(0);
        }
        
        // Apply highlighting based on file extension or content
        let highlightedHTML = this.highlightText(text);
        
        // Update editor content
        this.editor.innerHTML = highlightedHTML;
        
        // Restore cursor position
        if (range) {
            this.setCursorPosition(cursorPos.line, cursorPos.column);
        }
    }
    
    highlightText(text) {
        // Basic syntax highlighting for common languages
        let highlighted = text;
        
        // JavaScript highlighting
        if (this.fileName.endsWith('.js') || this.fileName.endsWith('.jsx')) {
            highlighted = this.highlightJavaScript(highlighted);
        }
        // Python highlighting
        else if (this.fileName.endsWith('.py')) {
            highlighted = this.highlightPython(highlighted);
        }
        // HTML highlighting
        else if (this.fileName.endsWith('.html') || this.fileName.endsWith('.htm')) {
            highlighted = this.highlightHTML(highlighted);
        }
        // CSS highlighting
        else if (this.fileName.endsWith('.css')) {
            highlighted = this.highlightCSS(highlighted);
        }
        // JSON highlighting
        else if (this.fileName.endsWith('.json')) {
            highlighted = this.highlightJSON(highlighted);
        }
        // Generic highlighting for common patterns
        else {
            highlighted = this.highlightGeneric(highlighted);
        }
        
        return highlighted;
    }
    
    highlightJavaScript(text) {
        // Keywords
        const keywords = ['function', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 'return', 'class', 'import', 'export', 'from', 'async', 'await', 'try', 'catch', 'finally', 'throw', 'new', 'this', 'typeof', 'instanceof'];
        keywords.forEach(keyword => {
            const regex = new RegExp(`\\b${keyword}\\b`, 'g');
            text = text.replace(regex, `<span class="keyword">${keyword}</span>`);
        });
        
        // Strings
        text = text.replace(/(["'`])((?:\\.|(?!\1)[^\\])*?)\1/g, '<span class="string">$1$2$1</span>');
        
        // Comments
        text = text.replace(/\/\/.*$/gm, '<span class="comment">$&</span>');
        text = text.replace(/\/\*[\s\S]*?\*\//g, '<span class="comment">$&</span>');
        
        // Numbers
        text = text.replace(/\b\d+\.?\d*\b/g, '<span class="number">$&</span>');
        
        // Functions
        text = text.replace(/\b(\w+)\s*(?=\()/g, '<span class="function">$1</span>');
        
        return text;
    }
    
    highlightPython(text) {
        // Keywords
        const keywords = ['def', 'class', 'if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with', 'import', 'from', 'as', 'return', 'yield', 'lambda', 'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None'];
        keywords.forEach(keyword => {
            const regex = new RegExp(`\\b${keyword}\\b`, 'g');
            text = text.replace(regex, `<span class="keyword">${keyword}</span>`);
        });
        
        // Strings
        text = text.replace(/(["'`])((?:\\.|(?!\1)[^\\])*?)\1/g, '<span class="string">$1$2$1</span>');
        
        // Comments
        text = text.replace(/#.*$/gm, '<span class="comment">$&</span>');
        
        // Numbers
        text = text.replace(/\b\d+\.?\d*\b/g, '<span class="number">$&</span>');
        
        // Functions
        text = text.replace(/\bdef\s+(\w+)/g, '<span class="keyword">def</span> <span class="function">$1</span>');
        
        return text;
    }
    
    highlightHTML(text) {
        // Tags
        text = text.replace(/&lt;(\/?)([^&gt;]+)&gt;/g, '&lt;<span class="keyword">$1$2</span>&gt;');
        text = text.replace(/<(\/?)([^>]+)>/g, '<span class="keyword">&lt;$1$2&gt;</span>');
        
        // Attributes
        text = text.replace(/(\w+)=/g, '<span class="function">$1</span>=');
        
        // Strings
        text = text.replace(/(["'`])((?:\\.|(?!\1)[^\\])*?)\1/g, '<span class="string">$1$2$1</span>');
        
        // Comments
        text = text.replace(/<!--[\s\S]*?-->/g, '<span class="comment">$&</span>');
        
        return text;
    }
    
    highlightCSS(text) {
        // Selectors
        text = text.replace(/([.#]?\w+)\s*{/g, '<span class="function">$1</span> {');
        
        // Properties
        text = text.replace(/(\w+)\s*:/g, '<span class="keyword">$1</span>:');
        
        // Values
        text = text.replace(/:\s*([^;]+);/g, ': <span class="string">$1</span>;');
        
        // Comments
        text = text.replace(/\/\*[\s\S]*?\*\//g, '<span class="comment">$&</span>');
        
        return text;
    }
    
    highlightJSON(text) {
        // Keys
        text = text.replace(/"([^"]+)"\s*:/g, '<span class="keyword">"$1"</span>:');
        
        // Strings
        text = text.replace(/:\s*"([^"]*)"/g, ': <span class="string">"$1"</span>');
        
        // Numbers
        text = text.replace(/:\s*(\d+\.?\d*)/g, ': <span class="number">$1</span>');
        
        // Booleans and null
        text = text.replace(/:\s*(true|false|null)/g, ': <span class="keyword">$1</span>');
        
        return text;
    }
    
    highlightGeneric(text) {
        // Basic string highlighting
        text = text.replace(/(["'`])((?:\\.|(?!\1)[^\\])*?)\1/g, '<span class="string">$1$2$1</span>');
        
        // Basic comment highlighting (// and #)
        text = text.replace(/\/\/.*$/gm, '<span class="comment">$&</span>');
        text = text.replace(/#.*$/gm, '<span class="comment">$&</span>');
        
        // Basic number highlighting
        text = text.replace(/\b\d+\.?\d*\b/g, '<span class="number">$&</span>');
        
        return text;
    }
}

// Global functions for help modal
function hideHelp() {
    document.getElementById('help-modal').style.display = 'none';
}

// Initialize the editor when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new VimEditor();
});
