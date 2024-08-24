<?php
include 'db.php';

$query = "SELECT * FROM posts ORDER BY created_at DESC";
$result = $conn->query($query);
?>

<!DOCTYPE html>
<html>
<head>
    <title>My Blog</title>
</head>
<body>
<h1>My Blog</h1>
<a href="addpost.php">Add new post</a>
<hr>
<?php
if ($result->num_rows > 0) {
    while ($row = $result->fetch_assoc()) {
        echo '<div><h2>' . $row['title'] . '</h2><p>' . $row['content'] . '</p></html>';
        echo '<hr>';
    }
} else {
    echo '<p>No posts found</p>';
}
$conn->close();
?>
</body>
</html>
