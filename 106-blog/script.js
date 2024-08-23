$(document).ready(function() {
    const postsPerPage = 5;
    let posts = [];
    let currentPage = 1;
    let totalPages = 0;

    function loadPosts(page) {
        const start = (page - 1) * postsPerPage;
        const end = start + postsPerPage;
        const postsToLoad = posts.slice(start, end);

        $('#posts-container').empty();
        $.each(postsToLoad, function(index, post) {
            $.get(`/posts/${post}`, function(data) {
                $('#posts-container').append(`<div class="post">${data}</div>`);
            });
        });
        updatePagination();
    }

    function updatePagination() {
        $('#page-info').text(`Page ${currentPage} / ${totalPages}`);
        $('#first-page').prop('disabled', currentPage === 1);
        $('#prev-page').prop('disabled', currentPage === 1);
        $('#next-page').prop('disabled', currentPage === totalPages);
        $('#last-page').prop('disabled', currentPage === totalPages);
    }

    function init() {
        $.get('posts/index.txt', function(data) {
            posts = data.trim().split('\n').filter(post => post !== 'index.txt');
            totalPages = Math.ceil(posts.length / postsPerPage);
            loadPosts(currentPage);
        });

        $('#first-page').click(function() {
            if (currentPage > 1) {
                currentPage = 1;
                loadPosts(currentPage);
            }
        });

        $('#prev-page').click(function() {
            if (currentPage > 1) {
                currentPage--;
                loadPosts(currentPage);
            }
        });

        $('#next-page').click(function() {
            if (currentPage < totalPages) {
                currentPage++;
                loadPosts(currentPage);
            }
        });

        $('#last-page').click(function() {
            if (currentPage < totalPages) {
                currentPage = totalPages;
                loadPosts(currentPage);
            }
        });
    }

    init();
});
