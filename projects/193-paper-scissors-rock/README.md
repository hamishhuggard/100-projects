# 🎮 Game Chatbot

A Flask + React chatbot that can play various games with users! The chatbot has a split-screen interface with chat on the left and game menu on the right.

## 🎯 Features

- **Split Layout**: Chat interface on the left, game menu on the right
- **Multiple Games**: 
  - Rock Paper Scissors ✂️
  - Ultimatum Game 💰
  - Prisoner's Dilemma 🤝
  - Number Guessing 🔢
- **AI Integration**: Powered by OpenAI GPT-4 for natural conversation
- **Game Tools**: The AI has special tools to set up games and make decisions
- **Responsive Design**: Works on both desktop and mobile devices

## 🚀 How to Use

### Starting the Backend
```bash
cd projects/193-paper-scissors-rock
python app.py
```

### Starting the Frontend
```bash
cd projects/193-paper-scissors-rock
npm start
```

### Playing Games

1. **Ask the chatbot to play**: Say "Let's play rock paper scissors" or "I want to play a game"
2. **Use the game menu**: Click on any game in the right sidebar to start
3. **Make your choice**: Type your selection and press Enter
4. **See the result**: The AI will make its decision and show the game outcome

## 🎮 Available Games

### Rock Paper Scissors
- Classic hand game
- Choose: rock, paper, or scissors
- AI makes random choices

### Ultimatum Game
- Propose resource splits
- Options: fair (50-50), generous (40-60), selfish (80-20)
- AI accepts fair/generous offers, rejects selfish ones

### Prisoner's Dilemma
- Game theory scenario
- Choose: cooperate or defect
- Payoff matrix determines results

### Number Guessing
- AI thinks of a number 1-10
- You guess and get feedback
- Perfect for testing the system

## 🛠️ Technical Details

- **Backend**: Flask with OpenAI API integration
- **Frontend**: React with modern CSS styling
- **Game Logic**: Python functions for each game type
- **State Management**: Session-based conversation and game state tracking
- **API Endpoints**: 
  - `/api/chat` - Regular chat
  - `/api/setup_game_menu` - Initialize games
  - `/api/make_game_decision` - Process game choices

## 🔧 Environment Setup

Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

## 📱 Responsive Design

- Desktop: Side-by-side chat and game layout
- Mobile: Stacked layout with chat on top, games below
- Touch-friendly game controls

## 🎨 UI Features

- Gradient backgrounds and modern styling
- Smooth animations and hover effects
- Clear game instructions and feedback
- Intuitive game selection interface

## 🔄 Reset Functionality

- Reset button clears both chat history and active games
- Fresh start for new conversations and game sessions

Enjoy playing games with your AI chatbot! 🎉