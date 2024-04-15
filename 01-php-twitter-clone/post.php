<?php
sesssion_start()
$conn = new mysqli("localhost", "", "", "twitter_clone");

if (!$conn) {
    die ("Connection failed: " . $conn->connect_error);
}

if ($_SERVER["REQUEST_METHOD"]=="POST" && isset($_SESSION["user_id"])) {
    $user_id = $_SESSION["user_id"];
    $tweet_text = trim($_POST["tweet"]);
    
    $stmt = $conn->prepare("INSERT INTO tweets (user_id, text) VALUES (?,?)");
    $stmt->bind_param("is", $user_id, $tweet_text);

    if ($stmt->execute()) {
        echo "Tweet posted successfully";
    } else {
        echo "Error: " . $stmt->error;
    }

    $stmt->close();
}

conn->close();
?>

<form action="post_tweet.php" method="post">
    <tetarea name="tweet" required></textarea><br>
    <input type="submit" value="Post">
</form>
