import os
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"status": "PhotoCoach AI backend running"})

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # Render expects the service to bind to the PORT environment variable
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
