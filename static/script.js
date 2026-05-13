// Состояние приложения
let currentSession = null;

// DOM элементы
const userPanel = document.getElementById('userPanel');
const sessionPanel = document.getElementById('sessionPanel');
const statsPanel = document.getElementById('statsPanel');
const userSelect = document.getElementById('userSelect');
const startBtn = document.getElementById('startBtn');
const endBtn = document.getElementById('endBtn');
const skipBtn = document.getElementById('skipBtn');
const closeStatsBtn = document.getElementById('closeStatsBtn');
const currentUserSpan = document.getElementById('currentUser');
const clickedCountSpan = document.getElementById('clickedCount');
const roundCountSpan = document.getElementById('roundCount');
const availableCountSpan = document.getElementById('availableCount');
const recommendationsContainer = document.getElementById('recommendationsContainer');
const statsContent = document.getElementById('statsContent');

// Загрузка списка пользователей
async function loadUsers() {
    try {
        const response = await fetch('/api/users');
        const data = await response.json();

        userSelect.innerHTML = '<option value="">-- Выберите пользователя --</option>';

        if (data.users && data.users.length > 0) {
            data.users.forEach(user => {
                const option = document.createElement('option');
                option.value = user;
                option.textContent = user;
                userSelect.appendChild(option);
            });
            startBtn.disabled = false;
        } else {
            userSelect.innerHTML = '<option value="">-- Пользователи не найдены --</option>';
        }
    } catch (error) {
        console.error('Ошибка загрузки пользователей:', error);
        userSelect.innerHTML = '<option value="">-- Ошибка загрузки --</option>';
    }
}

// Начало сессии
async function startSession() {
    const userId = userSelect.value;

    if (!userId) {
        alert('Пожалуйста, выберите пользователя');
        return;
    }

    try {
        const response = await fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: userId })
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        currentUserSpan.textContent = userId;
        userPanel.style.display = 'none';
        sessionPanel.style.display = 'block';

        if (data.recommendations) {
            displayRecommendations(data.recommendations);
            updateStats(data.stats);
        } else if (data.error) {
            showError(data.error);
        }

    } catch (error) {
        console.error('Ошибка начала сессии:', error);
        alert('Ошибка начала сессии');
    }
}

// Отображение рекомендаций
function displayRecommendations(recommendations) {
    if (!recommendations || recommendations.length === 0) {
        recommendationsContainer.innerHTML = '<div class="error">Нет рекомендаций</div>';
        return;
    }

    recommendationsContainer.innerHTML = '';

    recommendations.forEach(rec => {
        const card = document.createElement('div');
        card.className = 'recommendation-card';
        card.onclick = () => registerClick(rec.item_id);

        // Используем правильное поле - probability (не mf_score)
        let relevance = rec.probability;
        if (typeof relevance === 'undefined' || relevance === null || isNaN(relevance)) {
            relevance = 0;
        }
        const relevancePercent = Math.round(relevance * 100);

        card.innerHTML = `
            <div class="rank">${rec.rank}</div>
            <div class="title">${escapeHtml(rec.title || 'Без названия')}</div>
            <div class="genre">${escapeHtml(rec.category || 'unknown')}</div>
            <div class="relevance">Релевантность: ${relevancePercent}%</div>
        `;

        recommendationsContainer.appendChild(card);
    });
}

// Обновление статистики
function updateStats(stats) {
    if (stats) {
        clickedCountSpan.textContent = stats.clicked || 0;
        roundCountSpan.textContent = stats.round || 0;
        availableCountSpan.textContent = stats.available || 0;
    }
}

async function registerClick(articleId) {
    try {
        const response = await fetch('/api/click', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ article_id: articleId })
        });

        const data = await response.json();

        if (data.error) {
            if (data.session_complete) {
                alert('Сессия завершена! Все статьи просмотрены.');
                endSession();
            } else {
                alert(data.error);
            }
            return;
        }

        if (data.recommendations) {
            displayRecommendations(data.recommendations);
            updateStats(data.stats);
        }

    } catch (error) {
        console.error('Ошибка:', error);
        alert('Ошибка при регистрации клика');
    }
}

// Регистрация пропуска
async function registerSkip() {
    // Показываем индикатор загрузки
    recommendationsContainer.innerHTML = '<div class="loading">Загрузка новых рекомендаций...</div>';

    try {
        const response = await fetch('/api/skip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        console.log('Ответ при пропуске:', data);

        if (data.error) {
            if (data.session_complete) {
                alert('Сессия завершена! Все статьи просмотрены.');
                endSession();
            } else {
                alert(data.error);
                // Восстанавливаем предыдущие рекомендации
                if (data.recommendations) {
                    displayRecommendations(data.recommendations);
                }
            }
            return;
        }

        if (data.recommendations) {
            displayRecommendations(data.recommendations);
            updateStats(data.stats);
        } else {
            // Если нет рекомендаций, пробуем получить их отдельно
            await refreshRecommendations();
        }

    } catch (error) {
        console.error('Ошибка при пропуске:', error);
        alert('Ошибка при регистрации пропуска');
        // Пробуем обновить рекомендации
        await refreshRecommendations();
    }
}

// Функция для принудительного обновления рекомендаций
async function refreshRecommendations() {
    try {
        const response = await fetch('/api/recommend');
        const data = await response.json();

        if (data.recommendations) {
            displayRecommendations(data.recommendations);
            updateStats(data.stats);
        } else if (data.error) {
            recommendationsContainer.innerHTML = `<div class="error">${data.error}</div>`;
        }
    } catch (error) {
        console.error('Ошибка обновления:', error);
    }
}


// Завершение сессии
async function endSession() {
    try {
        await fetch('/api/end', { method: 'POST' });

        // Просто закрываем сессию без показа статистики
        sessionPanel.style.display = 'none';
        userPanel.style.display = 'block';
        userSelect.value = '';

        console.log('Сессия завершена');

    } catch (error) {
        console.error('Ошибка:', error);
        sessionPanel.style.display = 'none';
        userPanel.style.display = 'block';
    }
}

// Показать статистику
async function showStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();

        if (data.error) {
            statsContent.innerHTML = `<div class="error">${data.error}</div>`;
        } else {
            let genreHtml = '';
            if (data.genre_distribution && Object.keys(data.genre_distribution).length > 0) {
                genreHtml = '<h4>Клики по жанрам:</h4><ul>';
                for (const [genre, count] of Object.entries(data.genre_distribution)) {
                    genreHtml += `<li>${escapeHtml(genre)}: ${count} кликов</li>`;
                }
                genreHtml += '</ul>';
            } else {
                genreHtml = '<p>Нет кликов в этой сессии</p>';
            }

            statsContent.innerHTML = `
                <p><strong>Пользователь:</strong> ${escapeHtml(data.user || '—')}</p>
                <p><strong>Кликнуто статей:</strong> ${data.articles_clicked || 0}</p>
                <p><strong>Показано статей:</strong> ${data.articles_shown || 0}</p>
                <p><strong>Уникальных жанров:</strong> ${data.unique_genres || 0}</p>
                <p><strong>Раундов рекомендаций:</strong> ${data.recommendation_rounds || 0}</p>
                ${genreHtml}
            `;
        }

        statsPanel.style.display = 'block';

    } catch (error) {
        console.error('Ошибка:', error);
        statsContent.innerHTML = '<div class="error">Ошибка загрузки статистики</div>';
        statsPanel.style.display = 'block';
    }
}


// Закрыть статистику
function closeStats() {
    statsPanel.style.display = 'none';
}

// Вспомогательная функция для экранирования HTML
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showError(message) {
    recommendationsContainer.innerHTML = `<div class="error">${escapeHtml(message)}</div>`;
}

// Обработчики событий
startBtn.addEventListener('click', startSession);
endBtn.addEventListener('click', endSession);
skipBtn.addEventListener('click', registerSkip);
closeStatsBtn.addEventListener('click', closeStats);

// Загрузка пользователей при старте
loadUsers();