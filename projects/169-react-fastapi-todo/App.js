import React, { useState, useEffect } from 'react';

const App () => {
    const [todos, setTodos] = useState([]);
    const [newTodoText, setNewTodoText] = useState('');
    const API_BASE_URL = 'http://localhost:8000/api/todos';

    useEffect(() => {
        fetchTodos();
    }, [])

    const fetchTodos = async () => {
        try {
            const response = await fetch(API_BASE_URL)
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json()
            setTodos(data);
        } catch (error) {
            console.error('Error fetching To-Do item', error);
        }
    }

    const addTodo = async (e) => {
        e.preventDefault();
        if (!newTodoText.trim()) return;

        try {
            const response = await fetch(API_BASE_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
                body: JSON.stringify({ text: newTodoText }),
            });
        } catch (error) {
            console.error('Error adding To-Do item:', error);
        }
    }
}
