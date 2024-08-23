$(document).ready(function() {
    const postsPerPage = 5;
    let posts = [];

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
    }

    function createPagination(totalPages) {
        $('#pagination').empty();
        for (let i = 1; i <= totalPages; i++) {
            $('#pagination').append(`<div class="page-button" data-page="${i}">${i}</div>`);
        }
        $('.page-button').click(function() {
            const page = $(this).data('page');
            $('.page-button').removeClass('active');
            $(this).addClass('active');
            loadPosts(page);
        });
    }

    function init() {
        $.get('posts/index.txt', function(data) {
            posts = data.trim().split('\n').filter(post => post !== 'index.txt');
            const totalPages = Math.ceil(posts.length / postsPerPage);
            createPagination(totalPages);
            loadPosts(1);
            $('.page-button').first().addClass('active');
        });
    }

    init();
});
