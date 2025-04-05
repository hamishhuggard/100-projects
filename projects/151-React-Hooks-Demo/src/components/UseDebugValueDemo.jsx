import React, { useState, useEffect, useDebugValue } from 'react';

// Custom hook that uses useDebugValue
function useUserStatus(userId) {
  const [isOnline, setIsOnline] = useState(false);
  
  // In a real app, this would connect to a service
  useEffect(() => {
    // Simulate fetching user status
    const timer = setTimeout(() => {
      setIsOnline(Math.random() > 0.5);
    }, 2000);
    
    return () => clearTimeout(timer);
  }, [userId]);
  
  // This value shows up in React DevTools
  useDebugValue(isOnline ? 'Online' : 'Offline');
  
  return isOnline;
}

function UseDebugValueDemo() {
  const [userId, setUserId] = useState('user1');
  const isUserOnline = useUserStatus(userId);
  
  return (
    <div className="hook-demo">
      <h2>useDebugValue Hook</h2>
      <p>
        Current User: {userId}
      </p>
      <p>
        Status: {isUserOnline ? '🟢 Online' : '🔴 Offline'}
      </p>
      <button onClick={() => setUserId('user' + Math.floor(Math.random() * 10))}>
        Change User
      </button>
      <p>
        useDebugValue adds a label to custom hooks in React DevTools.
        Open React DevTools to see the "Online"/"Offline" label.
      </p>
    </div>
  );
}

export default UseDebugValueDemo; 