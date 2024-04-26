import React, { useState } from 'react';

function ParentComponent() {
    const [sharedState, setSharedState] = useState('initial state');
    return (
        <div>
            <ComponentA sharedState={sharedState} setSharedState={setSharedState} />
            <ComponentB sharedState={sharedState} setSharedState={setSharedState} />
        </div>
    )
}

function ComponentA({ sharedState, setSharedState }) {
    return (
        <p>Component A: {sharedState}</p>
        <button onClick={() => setSharedState('updated by A')}>Update state</button>
    )
}

function ComponentB({ sharedState, setSharedState }) {
    return (
        <p>Component B: {sharedState}</p>
        <button onClick={() => setSharedState('updated by B')}>Update state</button>
    )
}

export default ParentComponent;
