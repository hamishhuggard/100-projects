# Vim Code Editor

A modern, web-based code editor with full Vim keybindings and a beautiful dark theme.

## Features

- **Vim Keybindings**: Complete Vim modal editing experience
- **Modern UI**: Dark theme with syntax highlighting support
- **Line Numbers**: Automatic line numbering with scroll sync
- **Multiple Languages**: Support for various programming languages
- **File Operations**: New file, save, and download functionality
- **Responsive Design**: Works on desktop and mobile devices

## Getting Started

1. Open `index.html` in your web browser
2. The editor will load with some sample code
3. Start editing using Vim commands!

## Vim Keybindings

### Mode Switching
- `i` - Enter insert mode
- `a` - Enter insert mode after cursor
- `o` - Enter insert mode on new line below
- `v` - Enter visual mode
- `ESC` - Return to normal mode

### Movement (Normal Mode)
- `h`, `j`, `k`, `l` - Move left, down, up, right
- `0` - Move to beginning of line
- `$` - Move to end of line
- `gg` - Move to beginning of file
- `G` - Move to end of file
- `w` - Move to next word
- `b` - Move to previous word

### Editing (Normal Mode)
- `x` - Delete character under cursor
- `dd` - Delete entire line
- `yy` - Yank (copy) entire line
- `p` - Paste after cursor
- `u` - Undo
- `Ctrl+r` - Redo

### Visual Mode
- `v` - Enter visual mode
- `d` - Delete selection
- `y` - Yank selection

### Search and Replace
- `/` - Search forward
- `?` - Search backward
- `n` - Find next match
- `N` - Find previous match

### Command Mode
- `:` - Enter command mode
- `:w` - Save file
- `:q` - Quit
- `:wq` - Save and quit
- `:x` - Save and quit
- `:s/old/new/g` - Substitute text

### File Operations
- `zz` - Save file
- `qq` - Quit

## UI Controls

- **New File**: Clear the editor and start fresh
- **Save**: Download the current content as a text file
- **Language Select**: Choose programming language for syntax highlighting

## Browser Compatibility

This editor works best in modern browsers that support:
- ES6 classes
- ContentEditable
- CSS Grid and Flexbox
- Modern JavaScript APIs

## Customization

You can customize the editor by modifying:
- `styles.css` - Visual appearance and theme
- `script.js` - Editor behavior and additional Vim commands
- `index.html` - Layout and structure

## Keyboard Shortcuts

The editor also supports some standard keyboard shortcuts:
- `Ctrl+Z` - Undo
- `Ctrl+Y` - Redo
- `Ctrl+S` - Save
- `Ctrl+N` - New file

## Tips

1. **Start in Normal Mode**: The editor always starts in normal mode
2. **Use ESC**: Press ESC to return to normal mode from any other mode
3. **Practice Movement**: Get comfortable with hjkl movement before diving into editing
4. **Command Mode**: Use `:` for file operations and advanced commands
5. **Visual Mode**: Use `v` to select text before performing operations

## Contributing

Feel free to add more Vim commands, improve the UI, or add new features!

## License

This project is open source and available under the MIT License.
