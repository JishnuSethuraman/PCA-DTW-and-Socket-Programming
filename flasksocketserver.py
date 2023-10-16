from flask import Flask, request, jsonify
import csv
from datetime import datetime
import socket
app = Flask(__name__)

# Global buffer to store data
data_buffer = []
BATCH_SIZE = 10

@app.route('/data', methods=['POST'])
def receive_data():
    global data_buffer

    data = request.json
    data_buffer.append(data)

    if len(data_buffer) >= BATCH_SIZE: #buffer
        save_to_csv(data_buffer)
        data_buffer.clear()

    return jsonify(success=True), 200

def save_to_csv(buffered_data):
    filename = datetime.now().strftime('%d%H%M%S') + ".csv"
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = buffered_data[0].keys()  # Assuming all JSON objects have the same keys
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in buffered_data:
            writer.writerow(data)

if __name__ == '__main__':
    app.run(host= socket.gethostname(), port=8000, debug=True)