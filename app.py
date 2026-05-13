"""
Flask веб-сервер для гибридной рекомендательной системы
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from recommender_web import recommender_api

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/users', methods=['GET'])
def get_users():
    users = recommender_api.get_users()
    return jsonify({'users': users})


@app.route('/api/start', methods=['POST'])
def start_session():
    data = request.get_json()
    user_id = data.get('user_id')

    if not user_id:
        return jsonify({'error': 'Не указан ID пользователя'}), 400

    result = recommender_api.start_session(user_id)
    return jsonify(result)


@app.route('/api/recommend', methods=['GET'])
def get_recommendations():
    result = recommender_api.get_recommendations()
    return jsonify(result)


@app.route('/api/click', methods=['POST'])
def register_click():
    data = request.get_json()
    article_id = data.get('article_id')

    if not article_id:
        return jsonify({'error': 'Не указан ID статьи'}), 400

    result = recommender_api.register_click(article_id)
    return jsonify(result)


@app.route('/api/skip', methods=['POST'])
def register_skip():
    result = recommender_api.register_skip()
    return jsonify(result)


@app.route('/api/stats', methods=['GET'])
def get_stats():
    result = recommender_api.get_session_stats()
    return jsonify(result)


@app.route('/api/end', methods=['POST'])
def end_session():
    result = recommender_api.end_session()
    return jsonify(result)


if __name__ == '__main__':
    print("\n" + "="*50)
    print("ЗАПУСК ГИБРИДНОГО РЕКОМЕНДАТЕЛЯ")
    print("MF + Contextual Bandit с учётом жанров")
    print("="*50)
    print("Откройте в браузере: http://localhost:5000")
    print("="*50 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)