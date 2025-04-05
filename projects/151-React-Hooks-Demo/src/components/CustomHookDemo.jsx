import React, { useState, useEffect } from 'react';

// Custom hook for window dimensions
function useWindowSize() {
  const [windowSize, setWindowSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight
  });
  
  useEffect(() => {
    const handleResize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight
      });
    };
    
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);
  
  return windowSize;
}

// Custom hook for form input
function useInput(initialValue) {
  const [value, setValue] = useState(initialValue);
  
  const handleChange = (e) => {
    setValue(e.target.value);
  };
  
  const reset = () => {
    setValue(initialValue);
  };
  
  return {
    value,
    onChange: handleChange,
    reset
  };
}

function CustomHookDemo() {
  const { width, height } = useWindowSize();
  const nameInput = useInput('');
  const emailInput = useInput('');
  
  const handleSubmit = (e) => {
    e.preventDefault();
    alert(`Name: ${nameInput.value}, Email: ${emailInput.value}`);
    nameInput.reset();
    emailInput.reset();
  };

  return (
    <div className="hook-demo">
      <h2>Custom Hooks</h2>
      
      <div>
        <h3>useWindowSize</h3>
        <p>Window Width: {width}px</p>
        <p>Window Height: {height}px</p>
      </div>
      
      <div>
        <h3>useInput</h3>
        <form onSubmit={handleSubmit}>
          <div>
            <label>
              Name:
              <input 
                type="text" 
                value={nameInput.value} 
                onChange={nameInput.onChange} 
                placeholder="Enter name"
              />
            </label>
          </div>
          <div>
            <label>
              Email:
              <input 
                type="email" 
                value={emailInput.value} 
                onChange={emailInput.onChange} 
                placeholder="Enter email"
              />
            </label>
          </div>
          <button type="submit">Submit</button>
          <button type="button" onClick={() => {
            nameInput.reset();
            emailInput.reset();
          }}>
            Reset
          </button>
        </form>
      </div>
      
      <p>
        Custom hooks let you extract component logic into reusable functions.
      </p>
    </div>
  );
}

export default CustomHookDemo; 