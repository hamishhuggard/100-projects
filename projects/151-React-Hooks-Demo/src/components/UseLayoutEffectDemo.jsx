import React, { useState, useEffect, useLayoutEffect, useRef } from 'react';

function UseLayoutEffectDemo() {
  const [width, setWidth] = useState(0);
  const [height, setHeight] = useState(0);
  const [useLayoutEffectWidth, setUseLayoutEffectWidth] = useState(0);
  const elementRef = useRef(null);
  
  // useEffect runs after the browser has painted
  useEffect(() => {
    console.log('useEffect ran');
    if (elementRef.current) {
      setWidth(elementRef.current.getBoundingClientRect().width);
    }
  }, []);
  
  // useLayoutEffect runs synchronously after all DOM mutations but before browser paint
  useLayoutEffect(() => {
    console.log('useLayoutEffect ran');
    if (elementRef.current) {
      setUseLayoutEffectWidth(elementRef.current.getBoundingClientRect().width);
      setHeight(elementRef.current.getBoundingClientRect().height);
    }
  }, []);

  return (
    <div className="hook-demo">
      <h2>useLayoutEffect Hook</h2>
      <div
        ref={elementRef}
        style={{
          padding: '20px',
          backgroundColor: '#f0f0f0',
          border: '1px solid #ccc',
          marginBottom: '10px'
        }}
      >
        This element's dimensions are measured
      </div>
      
      <p>Width from useEffect: {width}px</p>
      <p>Width from useLayoutEffect: {useLayoutEffectWidth}px</p>
      <p>Height from useLayoutEffect: {height}px</p>
      
      <p>
        useLayoutEffect runs synchronously before browser paint,
        while useEffect runs after paint. Check the console to see
        the order they run in.
      </p>
    </div>
  );
}

export default UseLayoutEffectDemo; 