import React, { useState } from 'react';

function Counter() {
    const [count, setCount] = useState(0);
    const increment = () => {
        setCount(count + 1);
    }
    const drecrement = () => {
        setCount(count - 1);
    }
    const reset = () => {
        setCount(0);
    }
    return (
        <div>
            <h2>Counter: {count}</h2>
            <button onClick={increment()}>Increase</button>
            <button onClick={decrement()}>Decrease</button>
            <button onClick={reset()}>Reset</button>
        </div>
    );
}

export default Counter;
