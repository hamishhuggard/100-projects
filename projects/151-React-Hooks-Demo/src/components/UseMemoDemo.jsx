import React, { useState, useMemo } from 'react';

function UseMemoDemo() {
  const [count, setCount] = useState(0);
  const [wordIndex, setWordIndex] = useState(0);
  
  const words = ['hello', 'world', 'react', 'hooks', 'memo'];
  const word = words[wordIndex];
  
  // Without useMemo - this calculation runs on every render
  const letterCount = word.length;
  
  // With useMemo - this calculation only runs when word changes
  const expensiveCalculation = useMemo(() => {
    console.log('Computing factorial...');
    return factorial(letterCount);
  }, [letterCount]);
  
  // Helper function to calculate factorial
  function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
  }

  return (
    <div className="hook-demo">
      <h2>useMemo Hook</h2>
      <div>
        <p>Counter: {count}</p>
        <button onClick={() => setCount(count + 1)}>Increment</button>
      </div>
      <div>
        <p>Current Word: {word}</p>
        <p>Letter Count: {letterCount}</p>
        <p>Factorial of Letter Count: {expensiveCalculation}</p>
        <button 
          onClick={() => setWordIndex((wordIndex + 1) % words.length)}
        >
          Next Word
        </button>
      </div>
      <p>Check console to see when factorial is recalculated</p>
    </div>
  );
}

export default UseMemoDemo; 