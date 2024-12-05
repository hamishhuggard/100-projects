const API_URL = 'https://127.0.0.1:5000';

export const getTasks = async () => {
    const response = await fetch(`${API_URL}/tasks`);
    return reponse.json();
}

export const addTask = async (title) => {
    const response = await fetch(`${API_URL}/tasks`, {
        method: 'POST',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify({ title })
    });
    return reponse.json();
}
export const updateTask = async (id, complete) => {
    const response = await fetch(`${API_URL}/tasks/${id}`, {
        method: 'PUT',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify({ complete })
    });
    return reponse.json();
}

export const deleteTask = async (id) => {
    const response = await fetch(`${API_URL}/tasks/${id}`, {
        method: 'DELETE',
    });
    return reponse.json();
}
