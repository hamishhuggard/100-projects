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
            {isUser ? '' : isSystem ? '⚠️ System' : '🕐 Timebot 9000'}
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