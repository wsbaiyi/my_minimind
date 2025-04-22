// 等待HTML文档完全加载和解析
document.addEventListener('DOMContentLoaded', function() {

    // --- 搜索功能 ---
    const searchButton = document.querySelector('.search-button');
    const searchInput = document.querySelector('.search-input');

    // 检查元素是否存在
    if (searchButton && searchInput) {
        // 给搜索按钮添加点击事件监听器
        searchButton.addEventListener('click', function() {
            const searchTerm = searchInput.value.trim(); // 获取输入框的值并去除首尾空格

            if (searchTerm) {
                // 这里只是一个示例提示，实际应用中会进行页面跳转或API请求
                alert('您搜索了: ' + searchTerm);
                // 例如: window.location.href = '/results?search_query=' + encodeURIComponent(searchTerm);
            } else {
                alert('请输入搜索内容！');
            }
        });

        // 可选：给搜索输入框添加回车键监听
        searchInput.addEventListener('keypress', function(event) {
            // 检查是否按下了回车键 (keyCode 13)
            if (event.key === 'Enter' || event.keyCode === 13) {
                searchButton.click(); // 触发搜索按钮的点击事件
            }
        });
    } else {
        console.error("未能找到搜索按钮或输入框元素。");
    }


    // --- 侧边栏切换 (简单示例，可以扩展) ---
    const menuButton = document.querySelector('.menu-icon');
    const sidebar = document.querySelector('.sidebar');
    const contentGrid = document.querySelector('.content-grid'); // 获取内容区以便调整

     if(menuButton && sidebar && contentGrid) {
         menuButton.addEventListener('click', function() {
             // 切换一个CSS类来控制侧边栏的显示/隐藏或样式
             sidebar.classList.toggle('collapsed'); // 假设你有一个 .collapsed 类来隐藏或缩小侧边栏
             contentGrid.classList.toggle('shifted'); // 假设你有一个 .shifted 类来调整内容区的边距

             // 你需要在 CSS 中定义 .sidebar.collapsed 和 .content-grid.shifted 的样式
             // 例如:
             // .sidebar.collapsed { width: 72px; /* 或者 display: none; */ }
             // .content-grid.shifted { margin-left: 72px; /* 或者 margin-left: 0; */ }
             console.log("菜单按钮被点击，切换侧边栏状态");
         });
     } else {
         console.error("未能找到菜单按钮、侧边栏或内容网格元素。");
     }


    // --- 视频卡片点击 (示例) ---
    const videoCards = document.querySelectorAll('.video-card');

    videoCards.forEach(card => {
        card.addEventListener('click', function(event) {
            // 阻止点击事件冒泡到链接上（如果需要区分点击卡片和点击具体链接）
            // event.stopPropagation();

            // 获取视频标题或其他信息作为示例
            const titleElement = card.querySelector('.video-title');
            const title = titleElement ? titleElement.textContent : '未知视频';

            // 实际应用中会跳转到视频播放页
            alert('即将播放: ' + title);
            // 例如: const videoLink = card.querySelector('a').href; window.location.href = videoLink;
        });
    });

}); // DOMContentLoaded结束