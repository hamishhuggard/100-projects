const API_URL = "http://127.0.0.1:5000";

export const getTasks = async () => {
  const response = await fetch(`${API_URL}/tasks`);
  return response.json();
};

export const addTask = async (title) => {
  const response = await fetch(`${API_URL}/tasks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  return response.json();
};

export const updateTask = async (id, complete) => {
  const response = await fetch(`${API_URL}/tasks/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ complete }),
  });
  return response.json();
};

export const deleteTask = async (id) => {
  await fetch(`${API_URL}/tasks/${id}`, { method: "DELETE" });
};
