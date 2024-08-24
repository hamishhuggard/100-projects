import React, { useReducer } from 'react';

function reducer(state, action) {
    switch (action.type) {
        case 'increment':
            return { count: state.count+1 }
        case 'decrement':
            return { count: state.count-1 }
        default:
            throw new Error();
    }
}

function UseReducerExample() {
    const [state, dispatch] = useReducer(reducer, {count: 0});
    return (
        <div>
            <p>Count: {state}</p>
            <button onclick={() => dispatch({ type: 'increment' })}>increment</button>
            <button onclick={() => dispatch({ type: 'decrement' })}>decrement</button>
        </div>
    )
}

export default UseReducerExample;
