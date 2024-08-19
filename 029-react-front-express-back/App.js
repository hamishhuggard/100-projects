import React, { useState } from 'react';
import axios from 'axios';

function App() {
    const [number, setNumber] = useState('');
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

    const handleSubmit = async (event) => {
        event.preventDefault();
        setError('');
        try {
            const response = await axios.post('/square', { number: parseFloat(number) });
            setResult(response.data.result);
        } catch (err) {
            setError('Error calculating square. Enter number.');
        }
    };
    
    return (
        <div style={{ padding: '50px' }}>
            <h1>square calculator</h1>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    value={number}
                    onChange={(e) => setNumber(e.target.value)}
                    placeholder="Enter a number"
                />
                <button type="submit">Calculate</button>
            </form>
            {result !== null && <p>Square: {result}</p>}
            {error && <p stye={{ color: 'red' }}>{error}</p>}
        </div>
    )
}
