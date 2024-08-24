import React, { useState, useCallback } from 'react';

function UseCallbackExample() {
    const [count, setCount] = useState(0);
    const increment = useCallback(() => {
        setCount((c) => c + 1);
    }, []);
    return (
        <div>
            <p>Count: {count}</p>
            <button onclick={increment}>increment</button>
        </div>
    )
}

export default UseCallbackExample;
