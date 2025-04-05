import React, { useState, useCallback } from 'react';

// Child component that renders when props change
const Button = React.memo(({ onClick, children }) => {
  console.log(`${children} button rendered`);
  return (
    <button onClick={onClick}>
      {children}
    </button>
  );
});

function UseCallbackDemo() {
  const [count1, setCount1] = useState(0);
  const [count2, setCount2] = useState(0);

  // Without useCallback - this function is recreated on every render
  const incrementCount1 = () => {
    setCount1(count1 + 1);
  };

  // With useCallback - this function is memoized and only changes when dependencies change
  const incrementCount2 = useCallback(() => {
    setCount2(count2 + 1);
  }, [count2]);

  return (
    <div className="hook-demo">
      <h2>useCallback Hook</h2>
      <div>
        <p>Count 1: {count1}</p>
        <Button onClick={incrementCount1}>
          Increment Count 1 (without useCallback)
        </Button>
      </div>
      <div>
        <p>Count 2: {count2}</p>
        <Button onClick={incrementCount2}>
          Increment Count 2 (with useCallback)
        </Button>
      </div>
      <p>Check console to see which buttons re-render</p>
    </div>
  );
}

export default UseCallbackDemo; 