$(document).ready(function() {
    function getPostFileFromURL() {
        const params = new URLSearchParams(window.location.search);
        return params.get('file');
    }

    function loadPost() {
        const postFile = getPostFileFromURL();
        if (postFile) {
            $.get(`/posts/${postFile}`, function(data) {
                $('#post-container').html(data);
            });
        } else {
            $('#post-container').html('<p>Post not found.</p>');
        }
    }

    loadPost();
});
