from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder='', template_folder='')

# Serve index.html
@app.route('/')
def home():
    return send_from_directory('', 'index.html')

# Serve CSS
@app.route('/style.css')
def serve_css():
    return send_from_directory('', 'style.css')

# Prediction endpoint
@app.route('/predict', methods=['GET'])
def predict():
    stock_symbol = request.args.get('stock_symbol', '')
    if not stock_symbol:
        return jsonify({'error': 'No stock symbol provided'}), 400

    # Mock prediction data
    prediction_data = {
        "stock_symbol": stock_symbol,
        "final_prediction": 1234.56,
        "suggestion": "Buy"
    }
    return jsonify(prediction_data)

if __name__ == '__main__':
    app.run(debug=True)
