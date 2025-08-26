import React, { useState, useEffect } from 'react';
import './GameMenu.css';

const GameMenu = ({ onGameResult }) => {
  const [currentGame, setCurrentGame] = useState(null);
  const [gameMenu, setGameMenu] = useState(null);
  const [userChoice, setUserChoice] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);

  const availableGames = [
    {
      id: 'rock_paper_scissors',
      name: 'Rock Paper Scissors',
      description: 'Classic hand game',
      emoji: '✂️'
    },
    {
      id: 'ultimatum_game',
      name: 'Ultimatum Game',
      description: 'Propose resource splits',
      emoji: '💰'
    },
    {
      id: 'prisoners_dilemma',
      name: 'Prisoner\'s Dilemma',
      description: 'Game theory challenge',
      emoji: '🤝'
    },
    {
      id: 'number_guessing',
      name: 'Number Guessing',
      description: 'I think of a number, you guess',
      emoji: '🔢'
    }
  ];

  const startGame = async (gameId) => {
    try {
      const response = await fetch('/api/setup_game_menu', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          game_type: gameId,
          session_id: 'default'
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setGameMenu(data.game_menu);
        setCurrentGame(gameId);
        setIsPlaying(true);
      }
    } catch (error) {
      console.error('Error starting game:', error);
    }
  };

  const makeChoice = async () => {
    if (!userChoice.trim() || !currentGame) return;

    try {
      const response = await fetch('/api/make_game_decision', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          user_choice: userChoice,
          session_id: 'default'
        }),
      });

      if (response.ok) {
        const data = await response.json();
        
        // Send game result to parent component to display in chat
        if (onGameResult) {
          onGameResult(data.game_result);
        }
        
        // Reset game state
        setGameMenu(null);
        setCurrentGame(null);
        setIsPlaying(false);
        setUserChoice('');
      }
    } catch (error) {
      console.error('Error making choice:', error);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      makeChoice();
    }
  };

  const cancelGame = () => {
    setGameMenu(null);
    setCurrentGame(null);
    setIsPlaying(false);
    setUserChoice('');
  };

  if (isPlaying && gameMenu) {
    return (
      <div className="game-menu">
        <div className="game-header">
          <h3>{gameMenu.title}</h3>
          <button className="cancel-btn" onClick={cancelGame}>✕</button>
        </div>
        
        <div className="game-content">
          <p className="game-description">{gameMenu.description}</p>
          
          <div className="game-options">
            {gameMenu.options.map((option, index) => (
              <div key={index} className="game-option">
                {option}
              </div>
            ))}
          </div>
          
          <div className="game-input">
            <input
              type="text"
              placeholder="Type your choice..."
              value={userChoice}
              onChange={(e) => setUserChoice(e.target.value)}
              onKeyPress={handleKeyPress}
              autoFocus
            />
            <button onClick={makeChoice} disabled={!userChoice.trim()}>
              Play!
            </button>
          </div>
          
          <p className="game-instructions">{gameMenu.instructions}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="game-menu">
      <div className="game-header">
        <h3>🎮 Game Menu</h3>
      </div>
      
      <div className="game-content">
        <p className="game-intro">
          Choose a game to play with the AI chatbot!
        </p>
        
        <div className="game-list">
          {availableGames.map((game) => (
            <div
              key={game.id}
              className="game-item"
              onClick={() => startGame(game.id)}
            >
              <div className="game-emoji">{game.emoji}</div>
              <div className="game-info">
                <h4>{game.name}</h4>
                <p>{game.description}</p>
              </div>
            </div>
          ))}
        </div>
        
        <div className="game-tip">
          <p>💡 <strong>Tip:</strong> Ask the chatbot to play a game, or click on any game above to start!</p>
        </div>
      </div>
    </div>
  );
};

export default GameMenu;
