import React, { useReducer } from 'react';

// Reducer function
const counterReducer = (state, action) => {
  switch (action.type) {
    case 'INCREMENT':
      return { count: state.count + 1 };
    case 'DECREMENT':
      return { count: state.count - 1 };
    case 'RESET':
      return { count: 0 };
    case 'ADD':
      return { count: state.count + action.payload };
    default:
      return state;
  }
};

function UseReducerDemo() {
  // Initial state
  const initialState = { count: 0 };
  
  // useReducer returns current state and dispatch function
  const [state, dispatch] = useReducer(counterReducer, initialState);

  return (
    <div className="hook-demo">
      <h2>useReducer Hook</h2>
      <p>Count: {state.count}</p>
      <div>
        <button onClick={() => dispatch({ type: 'INCREMENT' })}>Increment</button>
        <button onClick={() => dispatch({ type: 'DECREMENT' })}>Decrement</button>
        <button onClick={() => dispatch({ type: 'RESET' })}>Reset</button>
        <button onClick={() => dispatch({ type: 'ADD', payload: 5 })}>Add 5</button>
      </div>
    </div>
  );
}

export default UseReducerDemo; 