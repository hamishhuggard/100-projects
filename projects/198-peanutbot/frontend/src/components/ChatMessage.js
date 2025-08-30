import React from 'react';
import './ChatMessage.css';

const ChatMessage = ({ message }) => {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  return (
    <div className={`message ${isUser ? 'user-message' : isSystem ? 'system-message' : 'bot-message'}`}>
      <div className="message-content">
        <div className="message-header">
          <span className="message-role">
            {isUser ? '🙂 You' : isSystem ? '⚠️ System' : '🤖 AI Assistant'}
          </span>
          <span className="message-time">{message.timestamp}</span>
        </div>
        <div className="message-text">
          {message.content}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage; 