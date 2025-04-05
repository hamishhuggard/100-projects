import React, { createContext, useContext, useState } from 'react';

// Create a context
const ThemeContext = createContext();

// Child component that uses the context
function ThemedButton() {
  const { theme, toggleTheme } = useContext(ThemeContext);
  
  return (
    <button 
      onClick={toggleTheme}
      style={{ 
        backgroundColor: theme === 'light' ? '#fff' : '#333',
        color: theme === 'light' ? '#333' : '#fff',
        padding: '8px 16px',
        border: '1px solid #ccc',
        borderRadius: '4px'
      }}
    >
      Toggle Theme (Current: {theme})
    </button>
  );
}

function UseContextDemo() {
  const [theme, setTheme] = useState('light');
  
  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  return (
    <div className="hook-demo">
      <h2>useContext Hook</h2>
      <ThemeContext.Provider value={{ theme, toggleTheme }}>
        <div style={{ 
          padding: '20px', 
          backgroundColor: theme === 'light' ? '#f0f0f0' : '#222',
          color: theme === 'light' ? '#333' : '#fff',
          borderRadius: '8px'
        }}>
          <p>Current theme: {theme}</p>
          <ThemedButton />
        </div>
      </ThemeContext.Provider>
    </div>
  );
}

export default UseContextDemo; 