import React, { useState } from 'react';

function UseStateDemo() {
  const [count, setCount] = useState(0);
  const [text, setText] = useState('');

  return (
    <div className="hook-demo">
      <h2>useState Hook</h2>
      <div>
        <p>Count: {count}</p>
        <button onClick={() => setCount(count + 1)}>Increment</button>
        <button onClick={() => setCount(count - 1)}>Decrement</button>
      </div>
      <div>
        <input 
          type="text" 
          value={text} 
          onChange={(e) => setText(e.target.value)} 
          placeholder="Type something..."
        />
        <p>You typed: {text}</p>
      </div>
    </div>
  );
}

export default UseStateDemo; 