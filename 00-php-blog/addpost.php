<?php
include 'db.php';

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $title = $conn->real_escape_string($_POST["title"]);
    $content = $conn->real_escape_string($_POST["content"]);
    
    $sql = "INSERT INTO posts (title, content) VALUE ('$title', '$content')";

    if ($conn->query($sql) == TRUE) {
        header("Location: index.php");
    } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
    }
    $conn->close();
}
?>

<!DOCTYPE html>
<html>
<head>
    <title>New Post</title>
</head>
<body>
<h1>New Post</h1>
<form method="post" action="addpost.php">
    Title: <input type="text" name="title" required><br>
    Content: <br><textarea name="content" rows="5" columns="40" required></textarea><br>
    <input type="submit" value="Submit">
</form>
</body>
</html>
