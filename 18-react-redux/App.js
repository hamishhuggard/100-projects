import React from 'react';
import { createStore } from 'redux';
import { Provider, useSelector, useDispatch } from 'react-redux';

const initialState = { sharedState: "initialState" };
function reducer(state = initialState, action) {
    switch (action.type) {
        case 'UPDATE_STATE':
            return { ...state, sharedState: action.payload };
        default:
            return state;
    }
}

const store = createStore(reducer);

function ComponentA() {
    const dispatch = useDispatch();
    const sharedState = useSelector(state => state.sharedState);
    return (
        <div>
            <h2>Component A</h2>
            <p>State: {sharedState}</p>
            <button onclick={() => dispatch({ type: 'UPDATE_STATE', payload: "updated by A" })}>update</button>
        </div>
    )
}

function ComponentB() {
    const sharedState = useSelector(state => state.sharedState);
    return (
        <div>
            <h2>Component B</h2>
            <p>State: {sharedState}</p>
        </div>
    )
}

function App() {
    return (
        <Provider store={store}>
            <div>
                <ComponentA/>
                <ComponentB/>
            </div>
        </Provider>
    )
}
