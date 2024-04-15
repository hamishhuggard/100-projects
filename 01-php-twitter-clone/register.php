<?php
$server_name = "localhost";
$username = "your_username";
$password = "your_password";
$db_name = "twitter_clone";

$conn = new mysqli($server_name, $username, $password, $dbname);

if ($conn->connect_error) {
    die("connection failed: " . $conn->connect_error);
}

if ($_SERVER["REQUEST_METHOD"]=="POST") {
    $username = trim($_POST["username"]);
    $email = trim($_POST["email"]);
    $password = $_POST["password"];
    $hashed_password = password_hash($password, PASSWORD_DEFAULT);

    $stmt = $conn->prepare("INSERT INTO Users (username, email, password) VALUES (?,?,?)");
    $stmt->bind_param("sss", $username, $email, $hashed_password);

    if ($stmt->execute()) {
        echo "User registered successfully.";
    } else {
        echo "Error: " . $stmt->error;
    }

    $stmt->close();
};

$conn->close();
?>

<form action="register.php" method="post">
    Username: <input type="text" name="username" required><br>
    Email: <input type="text" name="email" required><br>
    Password <input type="password" name="password" required><br>
    <input type="submit" value="Register">
    
</form>
