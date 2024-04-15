<?php
if ($_SERVER["REQUEST_METHOD"]=="POST") {
    $name = htmlspecialchars($_POST["name"]);
    $email = htmlspecialchars($_POST["email"]);
    $message = htmlspecialchars($_POST["message"]);
    
    $to = "hamishhuggard@gmail.com";
    $subject = "new phpmail";
    $body = "You have received a new message from: $name\nEmail: $email\nMessage: $message";
    $headers = "From: webmaster@example.com";

    if (mail($to, $subject, $body, $headers)) {
        echo "Message sent successfully";
    } else {
        echo "Failed to send message";
    }
}
?>
