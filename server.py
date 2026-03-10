import os
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return jsonify({"status": "PhotoCoach AI backend running"})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}

    image = data.get("image")
    question = data.get("question")

    if not image:
        return jsonify({
            "success": False,
            "error": "Kein Bild übergeben"
        }), 400

    # Übergangslösung: feste Antwort, damit dein Frontend funktioniert
    return jsonify({
        "success": True,
        "sceneType": "Allgemein",
        "confidence": 82,
        "fullAnalysis": """## Sofort-Tipps
- Achte auf ein klares Hauptmotiv.
- Prüfe, ob störende Elemente am Bildrand entfernt werden können.
- Verändere leicht deine Perspektive für mehr Tiefe.

## Workshop-Mission
Fotografiere dieselbe Szene noch einmal aus einer etwas tieferen Perspektive und achte auf eine klarere Bildführung.
""",
        "settings": {
            "Modus": "A (Blende)",
            "ISO": "200-800",
            "Blende": "f/4.0-8.0",
            "Verschlusszeit": "1/125 - 1/500",
            "AF-Modus": "AF-S",
            "Fokusfeld": "Flexible Spot",
            "Weißabgleich": "Auto"
        },
        "bookSources": []
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)