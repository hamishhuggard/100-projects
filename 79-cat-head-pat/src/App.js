import React, { useState } from 'react';
import './App.css';
import catImage from './cat.png';

function App() {
  const [count, setCount] = useState(0);
  return (
    <div className="App">
      <header className="App-header">
        <img src={catImage} alt="Cat" class="cat" onClick={() => setCount(count + 1)} style={{ cursor: 'pointer', width: '200px' }} />
        <p>Headpats: {count}</p>
      </header>
    </div>
  );
}

export default App;
