from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_video():
    try:
        video_data = request.data
        # Здесь обработайте видео в вашей нейронной сети
        # и получите результаты в формате JSON
        # В данном примере просто возвращаем случайные результаты
        result = {'prediction': 'Данные от нейронки'}
        return jsonify(result)
    except Exception as e:
        print(str(e))
        return jsonify({'error': 'Internal Server Error'}), 500

# Запуск сервера Flask
if __name__ == '__main__':
    app.run(port=5000)
