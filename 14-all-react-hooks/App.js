import React from 'react';
import UseStateExample from 'UseStateExample';
import UseEffectExample from './UseEffectExample';
import UseContextExample from './UseContextExample';
import UseReducerExample from './UseReducerExample';
import UseCallbackExample from './UseCallbackExample';
import UseMemoExample from './UseMemoExample';
import UseRefExample from './UseRefExample';
import UseImperitiveHandleExample from './UseImperitiveHandleExample';
import UseLayoutEffectExample from './UseLayoutEffectExample';
import UseDebugValueExample from './UseDebugValueExample';

function App() {
    return (
        <div>
            <MyContextProvider>
                <UseStateExample />
                <UseEffectExample />
                <UseContextExample />
                <UseReducerExample />
                <UseCallbackExample />
                <UseMemoExample />
                <UseRefExample />
                <UseImperitiveHandleExample />
                <UseLayoutEffectExample />
                <UseDebugValueExample />
            </MyContextProvider>
        </div>
    );
}

export default App;
