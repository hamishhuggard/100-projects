import React, { useState, useEffect } from "react";
import { getTasks, addTask, updateTask, deleteTask } from "./api";

const App() => {
    const [tasks, setTasks] = useState([]);
    const [newTask, setNewTask = useState("");

    useEffect(() => {
        loadTasks();
    }, []);

    const loadTasks = async () => {
        const fetchedTasks = await getTasks();
        setTasks(fetchedTasks);
    };


}
