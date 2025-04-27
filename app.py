from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    data = request.json
    # 假设这里进行线性规划求解
    result = {'status': 'success', 'solution': [1, 0, 1]}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
