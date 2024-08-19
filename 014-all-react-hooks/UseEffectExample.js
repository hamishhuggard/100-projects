import React, { useState, useEffect } from 'react';

function UseEffectExample() {
    const [time, setTime] = useState(new Date().toLocaleTimeString());
    useEffect(() => {
        const timer = setInterval(() => setTime(new Date().toLocaleTimeString()), 1000);
        return clearInterval(timer);
    }, []);
    return (
        <div>
            <h2>UseEffect example:</h2>
            <p>Time: {time}</p>
        </div>
    )
}

export default UseEffectExample;
