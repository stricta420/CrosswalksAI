from flask import Flask, request, jsonify
import numpy as np
import torch
import test

app = Flask(__name__)

@app.route('/process_tensor', methods=['POST'])
def process_tensor():
    data = request.get_json()
    
    if "tensor_data" not in data:
        return jsonify({"error": "Brak danych tensorowych"}), 400

    # Odtworzenie tensora z listy
    tensor_list = data["tensor_data"]
    tensor_pt = torch.tensor(tensor_list)

    shape,prob = test.allInOne(tensor_pt)

    # Odpowiedź serwera
    return jsonify({"shape": shape, "probability": prob}), 200

if __name__ == '__main__':
    app.run(debug=True)