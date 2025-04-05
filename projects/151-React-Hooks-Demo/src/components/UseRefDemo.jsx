import React, { useState, useRef, useEffect } from 'react';

function UseRefDemo() {
  const [inputValue, setInputValue] = useState('');
  const [renderCount, setRenderCount] = useState(0);
  
  // Create refs
  const inputRef = useRef(null);
  const previousInputValue = useRef('');
  const renderCountRef = useRef(0);
  
  // Focus the input element
  const focusInput = () => {
    inputRef.current.focus();
  };
  
  // Track renders without causing re-renders
  useEffect(() => {
    renderCountRef.current += 1;
  });
  
  // Track previous value
  useEffect(() => {
    previousInputValue.current = inputValue;
  }, [inputValue]);
  
  // This will cause re-renders
  useEffect(() => {
    setRenderCount(prevCount => prevCount + 1);
  }, [inputValue]);

  return (
    <div className="hook-demo">
      <h2>useRef Hook</h2>
      
      <div>
        <input
          ref={inputRef}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Type something..."
        />
        <button onClick={focusInput}>Focus Input</button>
      </div>
      
      <p>Current Value: {inputValue}</p>
      <p>Previous Value: {previousInputValue.current}</p>
      <p>Render Count (state): {renderCount}</p>
      <p>Render Count (ref): {renderCountRef.current}</p>
      
      <p>
        Notice how changing the input updates both render counts,
        but only the state version causes additional re-renders.
      </p>
    </div>
  );
}

export default UseRefDemo; 