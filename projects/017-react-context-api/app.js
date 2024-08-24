import React, { useState, createContext, useContext } from 'react';

const SharedStateContext = createContext(null);

function App() {
    const [sharedState, setSharedState] = useState('initial state');
    return (
        <SharedStateContext.Provider value={{ sharedState, setSharedState }}>
            <ComponentA />
            <ComponentB />
        </SharedStateContext.Provider>
    )
}

function ComponentA() {
    const { sharedState, setSharedState } = useContext(SharedStateContext);
    return (
        <p>Component A: {sharedState}</p>
        <button onClick={() => setSharedState('updated by A')}>Update state</button>
    )
}

function ComponentB() {
    const { sharedState } = useContext(SharedStateContext);
    return (
        <p>Component B: {sharedState}</p>
    )
}

export default App;
