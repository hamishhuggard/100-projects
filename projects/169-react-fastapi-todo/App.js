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
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const newTodo = await response.json()
            setTodos([...todos, newTodo]);
            setNewTodoText(''); // clear the input field
        } catch (error) {
            console.error('Error adding To-Do item:', error);
        }
    }

    const toggleTodoCompletion = async (id) => {
        const todoToUpdate = todos.find(todo => todo.id === id)
        if (!todoToUpdate) return;
        try {
            const response = await fetch(`${API_BASE_URL}/${id}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                }
                body: JSON.stringify({ completed: !todoToUpdate.completed }),
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const updatedTodo = await response.json()
            setTodos(todos.map(todo => todo.id === id ? updatedTodo : todo));
        } catch (error) {
            console.error('Error toggling To-Do item completion:', error);
        }
    }

    const deleteTodo = async (id) => {
        const todoToUpdate = todos.find(todo => todo.id === id)
        if (!todoToUpdate) return;
        try {
            const response = await fetch(`${API_BASE_URL}/${id}`, {
                method: 'DELETE',
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const updatedTodo = await response.json()
            setTodos(todos.filter(todo => todo.id !== id));
        } catch (error) {
            console.error('Error deleting To-Do item completion:', error);
        }
    }

    return (
        <div className="min-h-screen bg-gradient-to-r from-purple-400 via-pink-500 to-red-500 flex items-center justify-center p-4 font-sans">
            <div className="bg-white p-8 rounded-xl shadow-2xl w-full max-w-md">
                <h1 className="text-4xl font-extrabold text-center text-gray-800 mb-8">
                    My To-Do List
                </h1>
            </div>
        </div>
    )
}
