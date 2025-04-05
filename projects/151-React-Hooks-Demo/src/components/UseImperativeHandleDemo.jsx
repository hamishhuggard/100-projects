import React, { useRef, useImperativeHandle, forwardRef } from 'react';

// Child component with forwarded ref
const FancyInput = forwardRef((props, ref) => {
  const inputRef = useRef();
  
  // Expose only certain functions to parent
  useImperativeHandle(ref, () => ({
    focus: () => {
      inputRef.current.focus();
    },
    blur: () => {
      inputRef.current.blur();
    },
    setValue: (value) => {
      inputRef.current.value = value;
    }
  }));
  
  return <input ref={inputRef} {...props} />;
});

function UseImperativeHandleDemo() {
  const fancyInputRef = useRef();
  
  const focusInput = () => {
    fancyInputRef.current.focus();
  };
  
  const blurInput = () => {
    fancyInputRef.current.blur();
  };
  
  const setInputValue = () => {
    fancyInputRef.current.setValue('Hello from parent!');
  };

  return (
    <div className="hook-demo">
      <h2>useImperativeHandle Hook</h2>
      <FancyInput 
        ref={fancyInputRef} 
        placeholder="Type here..."
      />
      <div>
        <button onClick={focusInput}>Focus Input</button>
        <button onClick={blurInput}>Blur Input</button>
        <button onClick={setInputValue}>Set Value</button>
      </div>
      <p>
        useImperativeHandle customizes the instance value exposed when using refs,
        allowing parent components to call specific methods on children.
      </p>
    </div>
  );
}

export default UseImperativeHandleDemo; 