import React, { useState, useEffect } from 'react';

function UseEffectDemo() {
  const [count, setCount] = useState(0);
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);

  // Effect that runs on every render
  useEffect(() => {
    document.title = `Count: ${count}`;
  });

  // Effect that runs only on mount and unmount
  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    
    // Cleanup function
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  // Effect that runs when count changes
  useEffect(() => {
    console.log(`Count changed to: ${count}`);
  }, [count]);

  return (
    <div className="hook-demo">
      <h2>useEffect Hook</h2>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
      <p>Window width: {windowWidth}px</p>
      <p>Check the document title and console for effects</p>
    </div>
  );
}

export default UseEffectDemo; 