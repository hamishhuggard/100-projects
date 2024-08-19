import React, { useState, useEffect } from 'react';

function UserFetcher() {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch('https://jsonplaceholder.typicode.com/uses/1')
            .then(response => response.json())
            .then(data => {
                setUser(data);
                setLoading(false);
            })
            .catch(error => {
                console.log("Error: ", error);
                setLoading(false);
            });
    }, [])

    if (loading) {
        return <p>Loading...</p>
    }

    return (
        <div>
            {user ? (
                <div>
                    <h1>{user.name}</h1>
                    <div>email: {user.email}</div>
                    <div>phone: {user.phone}</div>
                    <div>website: {user.website}</div>
                </div>
            ) : (
                <p>No user data found</p>
            )}
        </div>
    );
}

export default UserFetcher;
