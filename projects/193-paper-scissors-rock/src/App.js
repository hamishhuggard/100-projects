import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import ChatHeader from './components/ChatHeader';
import GameMenu from './components/GameMenu';

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (message) => {
    if (!message.trim()) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: message,
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Use environment variable for backend URL, fallback to proxy
      const backendUrl = process.env.REACT_APP_BACKEND_URL || '';
      const apiUrl = backendUrl ? `${backendUrl}/api/chat` : '/api/chat';
      
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      
      const botMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.response,
        timestamp: new Date().toLocaleTimeString()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        id: Date.now() + 1,
        role: 'system',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGameResult = (gameResult) => {
    // Create a system message to display the game result in the chat
    const gameResultMessage = {
      id: Date.now(),
      role: 'system',
      content: `🎮 **Game Result:** ${gameResult.result}`,
      timestamp: new Date().toLocaleTimeString()
    };
    
    setMessages(prev => [...prev, gameResultMessage]);
  };

  const handleReset = () => {
    setMessages([]);
  };

  return (
    <div className="App">
      <ChatHeader onReset={handleReset} />
      <div className="main-container">
        <div className="chat-section">
          <div className="messages-container">
            {messages.length === 0 && (
              <div className="welcome-message">
                <h2>Welcome to the Game Chatbot! 🤖🎮</h2>
                <p>Start a conversation or choose a game to play from the menu on the right!</p>
                <p>Try saying: "Let's play rock paper scissors" or "I want to play a game"</p>
              </div>
            )}
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            {isLoading && (
              <div className="loading-message">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
        </div>
        
        <div className="game-section">
          <GameMenu onGameResult={handleGameResult} />
        </div>
      </div>
    </div>
  );
}

export default App; 