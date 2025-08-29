import React from 'react';
import './ChatHeader.css';

const ChatHeader = ({ onReset }) => {
  const currentDate = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  return (
    <header className="chat-header">
      <div className="header-content">
        <div className="header-left">
          <h1>Perfectly Normal Chatbot</h1>
        </div>
        <button className="reset-button" onClick={onReset}>
          Reset
        </button>
      </div>
    </header>
  );
};

export default ChatHeader; 