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
                <form onSubmit={addTodo} className="flex gap-3 mb-8">
                    <input 
                        type="text"
                        value={newTodoText}
                        onChange={e => setNewTodoText(e.target.value)}
                        placeholder="Add a new task..."
                        className="flex-grow p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-lg"
                    />
                    <button
                        type="submit"
                        className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 transition duration-300 ease-in-out transform hover:scale-105"
                    />
                        Add
                    </button>
                </form>
                {todos.length === 0 ? (
                    <p className="text-center text-gray-600 text-lg">No tasks yet! Add some above.</p>
                ) : (
                    <ul className="space-y-4">
                    {todos.map((todo) => (
                        <li
                            key={todo.id}
                            className="flex items-center justify-between bg-gray-50 p-4 rounded-lg shadow-sm border border-gray-200 transition duration-200 ease-in-out hover:shadow-md"
                        >
                            <span
                                className={`flex-grow text-lg cursor-pointer ${todo.completed ? 'line-through text-gray-500' : 'text-gray-800'}`}
                                onClick={() => toggleTodoCompleted(todo.id)}
                            >
                                {todo.text}
                            </span>
                            <div className="flex gap-2">
                                <button
                                    onClick={() => toggleTodoCompletion(todo.id)}
                                    className={`p-2 rounded-full text-white transition duration-300 ease-in-out transform hover:scale-110 ${todo.completed ? 'bg-green-500 hover:bg-green-600' : 'bg-yellow-500 hover:bg-yellow-600'}`}
                                    title={todo.completed ? 'Mark as incomplete' : 'Mark as complete'}
                                >

                                    {todo.completed ? (
                                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                            </svg>
                                    ) : (
                                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                                            </svg>
                                    )}

                                </button>
                                <button
                                    onClick={() => deleteTodo(todo.id)}
                                    className="p-2 rounded-full bg-red-500 text-white hover:bg-red-600 transition duration-300 ease-in-out transform hover:scale-110"
                                    title=
                                >
                                </button>
            </div>
        </div>
    )
}
