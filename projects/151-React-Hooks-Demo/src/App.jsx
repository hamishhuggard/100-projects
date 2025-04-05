import React from 'react';
import './App.css';
import UseStateDemo from './components/UseStateDemo';
import UseEffectDemo from './components/UseEffectDemo';
import UseContextDemo from './components/UseContextDemo';
import UseReducerDemo from './components/UseReducerDemo';
import UseCallbackDemo from './components/UseCallbackDemo';
import UseMemoDemo from './components/UseMemoDemo';
import UseRefDemo from './components/UseRefDemo';
import UseLayoutEffectDemo from './components/UseLayoutEffectDemo';
import UseImperativeHandleDemo from './components/UseImperativeHandleDemo';
import UseDebugValueDemo from './components/UseDebugValueDemo';
import CustomHookDemo from './components/CustomHookDemo';

function App() {
  return (
    <div className="App">
      <h1>React Hooks Demo</h1>
      <div className="hook-container">
        <UseStateDemo />
        <UseEffectDemo />
        <UseContextDemo />
        <UseReducerDemo />
        <UseCallbackDemo />
        <UseMemoDemo />
        <UseRefDemo />
        <UseLayoutEffectDemo />
        <UseImperativeHandleDemo />
        <UseDebugValueDemo />
        <CustomHookDemo />
      </div>
    </div>
  );
}

export default App; 