class VimEditor {
    constructor() {
        this.editor = document.getElementById('editor');
        this.lineNumbers = document.getElementById('line-numbers');
        this.modeDisplay = document.getElementById('mode-display');
        this.cursorPosition = document.getElementById('cursor-position');
        this.vimCommand = document.getElementById('vim-command');
        this.commandLine = document.getElementById('command-line');
        this.commandInput = document.getElementById('command-input');
        
        this.mode = 'normal'; // normal, insert, visual
        this.commandBuffer = '';
        this.lastSearch = '';
        this.marks = {};
        this.registers = {};
        this.history = [];
        this.historyIndex = -1;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.updateLineNumbers();
        this.updateCursorPosition();
        this.setMode('normal');
        
        // Set initial content
        this.editor.textContent = '// Welcome to Vim Code Editor\n// Press i to enter insert mode\n// Press : to enter command mode\n// Press ESC to return to normal mode\n\nfunction hello() {\n    console.log("Hello, Vim!");\n}';
        this.updateLineNumbers();
    }
    
    setupEventListeners() {
        this.editor.addEventListener('keydown', (e) => this.handleKeydown(e));
        this.editor.addEventListener('input', () => this.updateLineNumbers());
        this.editor.addEventListener('click', () => this.updateCursorPosition());
        this.editor.addEventListener('keyup', () => this.updateCursorPosition());
        
        // Command line events
        this.commandInput.addEventListener('keydown', (e) => this.handleCommandKeydown(e));
        
        // Button events
        document.getElementById('new-file').addEventListener('click', () => this.newFile());
        document.getElementById('save-file').addEventListener('click', () => this.saveFile());
        
        // Focus editor on load
        this.editor.focus();
    }
    
    handleKeydown(e) {
        if (this.mode === 'normal') {
            this.handleNormalMode(e);
        } else if (this.mode === 'insert') {
            this.handleInsertMode(e);
        } else if (this.mode === 'visual') {
            this.handleVisualMode(e);
        }
    }
    
    handleNormalMode(e) {
        const key = e.key.toLowerCase();
        const ctrl = e.ctrlKey;
        const shift = e.shiftKey;
        
        e.preventDefault();
        
        // Movement commands
        if (key === 'h' || key === 'arrowleft') {
            this.moveLeft();
        } else if (key === 'j' || key === 'arrowdown') {
            this.moveDown();
        } else if (key === 'k' || key === 'arrowup') {
            this.moveUp();
        } else if (key === 'l' || key === 'arrowright') {
            this.moveRight();
        } else if (key === '0') {
            this.moveToLineStart();
        } else if (key === '$') {
            this.moveToLineEnd();
        } else if (key === 'g' && this.commandBuffer === 'g') {
            this.moveToFileStart();
            this.commandBuffer = '';
        } else if (key === 'g') {
            this.commandBuffer = 'g';
        } else if (key === 'g' && this.commandBuffer === 'g') {
            this.moveToFileEnd();
            this.commandBuffer = '';
        } else if (key === 'w') {
            this.moveToNextWord();
        } else if (key === 'b') {
            this.moveToPrevWord();
        
        // Mode switching
        } else if (key === 'i') {
            this.setMode('insert');
        } else if (key === 'a') {
            this.moveRight();
            this.setMode('insert');
        } else if (key === 'o') {
            this.moveToLineEnd();
            this.insertNewline();
            this.setMode('insert');
        } else if (key === 'v') {
            this.setMode('visual');
        
        // Editing commands
        } else if (key === 'x') {
            this.deleteChar();
        } else if (key === 'd' && this.commandBuffer === 'd') {
            this.deleteLine();
            this.commandBuffer = '';
        } else if (key === 'd') {
            this.commandBuffer = 'd';
        } else if (key === 'y' && this.commandBuffer === 'y') {
            this.yankLine();
            this.commandBuffer = '';
        } else if (key === 'y') {
            this.commandBuffer = 'y';
        } else if (key === 'p') {
            this.paste();
        } else if (key === 'u') {
            this.undo();
        } else if (ctrl && key === 'r') {
            this.redo();
        
        // Search and replace
        } else if (key === '/') {
            this.searchForward();
        } else if (key === '?') {
            this.searchBackward();
        } else if (key === 'n') {
            this.findNext();
        } else if (key === 'n' && shift) {
            this.findPrev();
        
        // Command mode
        } else if (key === ':') {
            this.enterCommandMode();
        
        // Save and quit
        } else if (key === 'z' && this.commandBuffer === 'z') {
            this.saveFile();
            this.commandBuffer = '';
        } else if (key === 'z') {
            this.commandBuffer = 'z';
        } else if (key === 'q' && this.commandBuffer === 'q') {
            this.quit();
            this.commandBuffer = '';
        } else if (key === 'q') {
            this.commandBuffer = 'q';
        
        // Clear command buffer for other keys
        } else {
            this.commandBuffer = '';
        }
        
        this.updateCursorPosition();
    }
    
    handleInsertMode(e) {
        if (e.key === 'Escape') {
            this.setMode('normal');
            e.preventDefault();
        }
        // All other keys are handled by the contenteditable
    }
    
    handleVisualMode(e) {
        const key = e.key.toLowerCase();
        
        if (key === 'escape') {
            this.setMode('normal');
            e.preventDefault();
        } else if (key === 'd') {
            this.deleteSelection();
            this.setMode('normal');
            e.preventDefault();
        } else if (key === 'y') {
            this.yankSelection();
            this.setMode('normal');
            e.preventDefault();
        }
    }
    
    setMode(mode) {
        this.mode = mode;
        this.modeDisplay.textContent = mode.toUpperCase();
        
        if (mode === 'insert') {
            this.editor.focus();
        }
    }
    
    // Movement methods
    moveLeft() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            range.collapse(true);
            range.setStart(range.startContainer, Math.max(0, range.startOffset - 1));
            selection.removeAllRanges();
            selection.addRange(range);
        }
    }
    
    moveRight() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            range.collapse(true);
          range.setStart(range.startContainer, Math.min(range.startContainer.length, range.startOffset + 1));
            selection.removeAllRanges();
            selection.addRange(range);
        }
    }
    
    moveUp() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            const prevNode = this.getPreviousNode(range.startContainer);
            if (prevNode) {
                range.setStart(prevNode, Math.min(prevNode.length, range.startOffset));
                selection.removeAllRanges();
                selection.addRange(range);
            }
        }
    }
    
    moveDown() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            const nextNode = this.getNextNode(range.startContainer);
            if (nextNode) {
                range.setStart(nextNode, Math.min(nextNode.length, range.startOffset));
                selection.removeAllRanges();
                selection.addRange(range);
            }
        }
    }
    
    moveToLineStart() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            const lineStart = this.getLineStart(range.startContainer);
            range.setStart(lineStart, 0);
            selection.removeAllRanges();
            selection.addRange(range);
        }
    }
    
    moveToLineEnd() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            const lineEnd = this.getLineEnd(range.startContainer);
            range.setStart(lineEnd, lineEnd.length);
            selection.removeAllRanges();
            selection.addRange(range);
        }
    }
    
    moveToFileStart() {
        const selection = window.getSelection();
        const range = document.createRange();
        range.setStart(this.editor.firstChild || this.editor, 0);
        selection.removeAllRanges();
        selection.addRange(range);
    }
    
    moveToFileEnd() {
        const selection = window.getSelection();
        const range = document.createRange();
        const lastChild = this.editor.lastChild || this.editor;
        range.setStart(lastChild, lastChild.length);
        selection.removeAllRanges();
        selection.addRange(range);
    }
    
    moveToNextWord() {
        // Simplified word movement
        this.moveRight();
    }
    
    moveToPrevWord() {
        // Simplified word movement
        this.moveLeft();
    }
    
    // Editing methods
    deleteChar() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            if (range.startOffset < range.startContainer.length) {
                range.setEnd(range.startContainer, range.startOffset + 1);
                range.deleteContents();
            }
        }
    }
    
    deleteLine() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            const lineStart = this.getLineStart(range.startContainer);
            const lineEnd = this.getLineEnd(range.startContainer);
            range.setStart(lineStart, 0);
            range.setEnd(lineEnd, lineEnd.length);
            range.deleteContents();
        }
    }
    
    yankLine() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            const lineStart = this.getLineStart(range.startContainer);
            const lineEnd = this.getLineEnd(range.startContainer);
            range.setStart(lineStart, 0);
            range.setEnd(lineEnd, lineEnd.length);
            this.registers['"'] = range.toString();
        }
    }
    
    paste() {
        if (this.registers['"']) {
            const selection = window.getSelection();
            if (selection.rangeCount > 0) {
                const range = selection.getRangeAt(0);
                range.deleteContents();
                range.insertNode(document.createTextNode(this.registers['"']));
            }
        }
    }
    
    undo() {
        document.execCommand('undo');
    }
    
    redo() {
        document.execCommand('redo');
    }
    
    insertNewline() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            range.insertNode(document.createTextNode('\n'));
        }
    }
    
    // Visual mode methods
    deleteSelection() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            this.registers['"'] = range.toString();
            range.deleteContents();
        }
    }
    
    yankSelection() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            this.registers['"'] = range.toString();
        }
    }
    
    // Search methods
    searchForward() {
        this.lastSearch = prompt('Search forward:');
        if (this.lastSearch) {
            this.findNext();
        }
    }
    
    searchBackward() {
        this.lastSearch = prompt('Search backward:');
        if (this.lastSearch) {
            this.findPrev();
        }
    }
    
    findNext() {
        if (this.lastSearch) {
            const text = this.editor.textContent;
            const currentPos = this.getCursorPosition();
            const nextPos = text.indexOf(this.lastSearch, currentPos + 1);
            if (nextPos !== -1) {
                this.setCursorPosition(nextPos);
            }
        }
    }
    
    findPrev() {
        if (this.lastSearch) {
            const text = this.editor.textContent;
            const currentPos = this.getCursorPosition();
            const prevPos = text.lastIndexOf(this.lastSearch, currentPos - 1);
            if (prevPos !== -1) {
                this.setCursorPosition(prevPos);
            }
        }
    }
    
    // Command mode
    enterCommandMode() {
        this.commandLine.style.display = 'flex';
        this.commandInput.focus();
        this.vimCommand.textContent = ':';
    }
    
    handleCommandKeydown(e) {
        if (e.key === 'Enter') {
            this.executeCommand(this.commandInput.value);
            this.commandInput.value = '';
            this.commandLine.style.display = 'none';
            this.editor.focus();
        } else if (e.key === 'Escape') {
            this.commandInput.value = '';
            this.commandLine.style.display = 'none';
            this.editor.focus();
        }
    }
    
    executeCommand(cmd) {
        const [command, ...args] = cmd.trim().split(' ');
        
        switch (command) {
            case 'w':
            case 'write':
                this.saveFile();
                break;
            case 'q':
            case 'quit':
                this.quit();
                break;
            case 'wq':
                this.saveFile();
                this.quit();
                break;
            case 'x':
                this.saveFile();
                this.quit();
                break;
            case 'set':
                if (args[0] === 'number') {
                    this.toggleLineNumbers();
                }
                break;
            default:
                if (command.startsWith('s/')) {
                    this.substitute(command);
                }
        }
    }
    
    substitute(cmd) {
        // Simple substitute command: s/old/new/g
        const match = cmd.match(/^s\/(.+)\/(.+)\/([g]*)$/);
        if (match) {
            const [, old, new_, flags] = match;
            const text = this.editor.textContent;
            const regex = new RegExp(old.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), flags.includes('g') ? 'g' : '');
            this.editor.textContent = text.replace(regex, new_);
        }
    }
    
    // File operations
    newFile() {
        if (confirm('Are you sure you want to create a new file? Unsaved changes will be lost.')) {
            this.editor.textContent = '';
            this.updateLineNumbers();
            this.updateCursorPosition();
        }
    }
    
    saveFile() {
        const content = this.editor.textContent;
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'code.txt';
        a.click();
        URL.revokeObjectURL(url);
        this.vimCommand.textContent = 'File saved';
        setTimeout(() => this.vimCommand.textContent = '', 2000);
    }
    
    quit() {
        if (confirm('Are you sure you want to quit?')) {
            window.close();
        }
    }
    
    // Utility methods
    getCursorPosition() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            return this.getTextOffset(range.startContainer, range.startOffset);
        }
        return 0;
    }
    
    setCursorPosition(offset) {
        const textNode = this.editor.firstChild;
        if (textNode) {
            const range = document.createRange();
            range.setStart(textNode, Math.min(offset, textNode.length));
            const selection = window.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
        }
    }
    
    getTextOffset(node, offset) {
        let totalOffset = 0;
        const walker = document.createTreeWalker(
            this.editor,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );
        
        let currentNode;
        while (currentNode = walker.nextNode()) {
            if (currentNode === node) {
                return totalOffset + offset;
            }
            totalOffset += currentNode.length;
        }
        return totalOffset;
    }
    
    getPreviousNode(node) {
        const walker = document.createTreeWalker(
            this.editor,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );
        
        let prevNode = null;
        let currentNode;
        while (currentNode = walker.nextNode()) {
            if (currentNode === node) {
                return prevNode;
            }
            prevNode = currentNode;
        }
        return null;
    }
    
    getNextNode(node) {
        const walker = document.createTreeWalker(
            this.editor,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );
        
        let found = false;
        let currentNode;
        while (currentNode = walker.nextNode()) {
            if (found) {
                return currentNode;
            }
            if (currentNode === node) {
                found = true;
            }
        }
        return null;
    }
    
    getLineStart(node) {
        // Simplified line start detection
        return node;
    }
    
    getLineEnd(node) {
        // Simplified line end detection
        return node;
    }
    
    updateLineNumbers() {
        const lines = this.editor.textContent.split('\n');
        this.lineNumbers.innerHTML = lines.map((_, i) => `<div>${i + 1}</div>`).join('');
    }
    
    updateCursorPosition() {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            const text = this.editor.textContent;
            const offset = this.getTextOffset(range.startContainer, range.startOffset);
            const lines = text.substring(0, offset).split('\n');
            const line = lines.length;
            const col = lines[lines.length - 1].length + 1;
            this.cursorPosition.textContent = `Ln ${line}, Col ${col}`;
        }
    }
    
    toggleLineNumbers() {
        this.lineNumbers.style.display = 
            this.lineNumbers.style.display === 'none' ? 'block' : 'none';
    }
}

// Initialize the editor when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new VimEditor();
}); 