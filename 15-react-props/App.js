import React from 'react';
import MessageDisplay from './MessageDisplay';

function App() {
    const message = 'hello';
    return (
        <div>
            <MessageDisplay message={message} />
        </div>
    )
};

export default App;
