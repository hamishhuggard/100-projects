import React, { useState } from 'react';

function UseStateExample() {
    const [count, setCount] = useState(0);
    return (
        <div>
            <div>Count: {count}</div>
            <button onClick={() => setCount(count+1)}>Increment</button>
        </div>
    )
}

export default UseStateExample;
