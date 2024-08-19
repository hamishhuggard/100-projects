import React, { useContext, createContext } from 'react';

const MyContext = createContext();

export function MyContextProvider({ children }) {
    return <MyContext.Provider value="Hello from Context">{children}</MyContext.Provider>
}

function UseContextExample() {
    const message = useContext(MyContext);
    return (
        <div>{message}</div>
    )
}

export default UseContextExample;
