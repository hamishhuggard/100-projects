import React, { useEffect, useState } from 'react';

function UserFetcher() {
    const [userId, setUserId] = useState(1);
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (!userId) return;
        setLoading(true);
        fetch(`https://jsonplaceholder.typicode.com/user/${userId}`)
            .then(response => response.json)
            .then(data => {
                setUset(data);
                setLoading(false);
            })
            .catch(err => {
                console.error('Failed to fetch: ', err);
                setLoading(false);
            });
    }, [userId]);

    return (
        <div>
            <input
                type="number"
                value={userId}
                onChange={e => setUserId(e.target.value)}
                placeholder="Enter user ID"
            >
        </div>
        {loading ? (
            <p>Loading...</p>
        ) : (
            user && (
                <div>
                    <h3>User details:</h3>
                    <p>name: {user.name}</p>
                    <p>email: {user.email}</p>
                    <p>website: {user.website}</p>
                </div>
            )
        )}
        </div>
    );
}

export default UserFetcher;
