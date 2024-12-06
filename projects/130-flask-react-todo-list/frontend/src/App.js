import React, { useState, useEffect } from "react";
import { getTasks, addTask, updateTask, deleteTask } from "./api";

const App = () => {
    const [tasks, setTasks] = useState([]);
    const [newTask, setNewTask] = useState("");

    useEffect(() => {
        loadTasks();
    }, []);

    const loadTasks = async () => {
        const fetchedTasks = await getTasks();
        setTasks(fetchedTasks);
    };

    const handleAddTask = async () => {
        if (newTask.trim() === "") return;
        const addedTask = await addTask(newTask);
        setTasks([...tasks, addedTask]);
        setNewTask("");
    };

    const handleToggleTask = async (taskId, complete) => {
        const updatedTask = await updateTask(taskId, complete);
        setTasks(tasks.map((task) => (task.id === taskId ? updatedTask : task)));
    };

    const handleDeleteTask = async (taskId) => {
        await deleteTask(taskId);
        setTasks(tasks.filter((task) => task.id !== taskId));
    };

    return (
        <div className="container">
            <h1>Todo List</h1>
            <div>
                <input
                    type="text"
                    value={newTask}
                    onChange={(e) => setNewTask(e.target.value)}
                    placeholder="New Task"
                />
                <button onClick={handleAddTask}>Add task</button>
            </div>
            <ul>
                {tasks.map((task) => (
                    <li key={task.id}>
                        <input
                            type="checkbox"
                            checked={task.complete}
                            onChange={() => handleToggleTask(task.id, !task.complete)}
                        />
                        {task.title}
                        <button onClick={() => handleDeleteTask(task.id)}>Delete</button>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default App;
