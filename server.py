#!/usr/bin/env python3.11
"""
PhotoCoach AI - Backend Server
Flask API mit OpenAI Vision + RAG-System (Buchinhalt)
"""

import os
import json
import base64
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.')
CORS(app)

# OpenAI Client
openai_client = OpenAI()

# Embedding-Modell (einmalig laden)
logger.info("Lade Embedding-Modell...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
logger.info("Embedding-Modell geladen.")

# ChromaDB Client
chroma_client = chromadb.PersistentClient(path="/home/ubuntu/photocoach-ai/data/chroma_db")

def get_book_context(query, n_results=5):
    """Sucht relevante Buchpassagen für eine Anfrage."""
    try:
        collection = chroma_client.get_collection("photocoach_books")
        
        # Query-Embedding erstellen
        query_embedding = embedding_model.encode([query]).tolist()
        
        # Suche in ChromaDB
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        context_parts = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            similarity = 1 - dist
            if similarity > 0.2:  # Nur relevante Ergebnisse
                context_parts.append({
                    'text': doc,
                    'book': meta.get('book', 'Unbekannt'),
                    'chapter': meta.get('chapter', 'Unbekannt'),
                    'category': meta.get('category', 'allgemein'),
                    'similarity': round(similarity, 3)
                })
        
        return context_parts
    except Exception as e:
        logger.error(f"Fehler bei Buchsuche: {e}")
        return []


def format_book_context(context_parts):
    """Formatiert den Buchkontext für den KI-Prompt."""
    if not context_parts:
        return ""
    
    formatted = "\n\n=== WISSEN AUS DEN BÜCHERN ===\n"
    for i, part in enumerate(context_parts, 1):
        book_name = part['book'].replace('_', ' ').title()
        formatted += f"\n[Quelle {i}: {book_name} – {part['chapter']}]\n"
        formatted += part['text'][:600] + "...\n"
    
    return formatted


def analyze_photo_with_rag(image_base64, user_question=None):
    """
    Analysiert ein Foto mit OpenAI Vision und reichert die Antwort
    mit Wissen aus den Büchern (RAG) an.
    """
    
    # Schritt 1: Erste Bildanalyse für Kontext-Suche
    logger.info("Schritt 1: Bildanalyse für RAG-Kontext...")
    
    initial_response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "low"
                        }
                    },
                    {
                        "type": "text",
                        "text": """Analysiere dieses Foto kurz und beschreibe:
1. Motiv/Szenentyp (Portrait, Landschaft, Street, Architektur, Makro, etc.)
2. Lichtsituation (Tageslicht, Kunstlicht, Gegenlicht, etc.)
3. Hauptprobleme die du siehst (Belichtung, Schärfe, Komposition, etc.)
Antworte in 3-4 Sätzen auf Deutsch."""
                    }
                ]
            }
        ],
        max_tokens=300
    )
    
    initial_analysis = initial_response.choices[0].message.content
    logger.info(f"Initiale Analyse: {initial_analysis[:100]}...")
    
    # Schritt 2: Relevante Buchpassagen suchen
    logger.info("Schritt 2: Buchpassagen suchen...")
    
    # Suchanfragen basierend auf der Analyse
    search_queries = [initial_analysis]
    if user_question:
        search_queries.append(user_question)
    
    all_context = []
    seen_texts = set()
    
    for query in search_queries:
        context = get_book_context(query, n_results=4)
        for item in context:
            if item['text'][:100] not in seen_texts:
                seen_texts.add(item['text'][:100])
                all_context.append(item)
    
    # Top 5 relevanteste Ergebnisse
    all_context.sort(key=lambda x: x['similarity'], reverse=True)
    top_context = all_context[:5]
    
    book_context = format_book_context(top_context)
    logger.info(f"Buchkontext gefunden: {len(top_context)} relevante Passagen")
    
    # Schritt 3: Detaillierte Analyse mit Buchkontext
    logger.info("Schritt 3: Detaillierte KI-Analyse mit Buchkontext...")
    
    system_prompt = """Du bist ein professioneller Fotografie-Coach mit tiefem Fachwissen.
Du analysierst Fotos präzise und gibst konkrete, umsetzbare Verbesserungstipps.
Deine Antworten basieren IMMER auf dem Wissen aus den bereitgestellten Fachbüchern.
Du beziehst dich auf spezifische Kameraeinstellungen der Sony Alpha 6000.
Deine Sprache ist klar, professionell und ermutigend."""

    user_prompt = f"""Analysiere dieses Foto als professioneller Fotografie-Coach.

{book_context}

Basierend auf dem Buchwissen oben, erstelle eine strukturierte Analyse:

## Szenentyp & Ersteinschätzung
[Beschreibe Motiv, Lichtsituation und Gesamteindruck]

## Fehleranalyse
[Benenne konkret 2-4 Schwachpunkte: Belichtung, Schärfe, Komposition, Farbe, etc.]

## Sony Alpha 6000 Einstellungen
[Empfehle konkrete Kameraeinstellungen basierend auf dem Handbuch:
- Aufnahmemodus (A/S/M/P)
- ISO
- Blende
- Verschlusszeit
- Autofokus-Modus
- Weißabgleich]

## Verbesserungstipps
[Gib 3-5 konkrete, sofort umsetzbare Tipps zur Bildverbesserung]

## Workshop-Mission
[Eine spezifische Übungsaufgabe für das nächste Foto]

{f'Besondere Frage des Nutzers: {user_question}' if user_question else ''}

Antworte vollständig auf Deutsch. Beziehe dich auf das Buchwissen."""

    final_response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ],
        max_tokens=1500
    )
    
    analysis_text = final_response.choices[0].message.content
    
    # Schritt 4: Strukturierte Daten extrahieren
    scene_type = extract_scene_type(initial_analysis)
    settings = extract_settings(analysis_text)
    
    return {
        "success": True,
        "sceneType": scene_type,
        "initialAnalysis": initial_analysis,
        "fullAnalysis": analysis_text,
        "settings": settings,
        "bookSources": [
            {
                "book": ctx['book'].replace('_', ' ').title(),
                "chapter": ctx['chapter'],
                "similarity": ctx['similarity']
            }
            for ctx in top_context
        ],
        "confidence": calculate_confidence(top_context)
    }


def extract_scene_type(analysis_text):
    """Extrahiert den Szenentyp aus der Analyse."""
    text_lower = analysis_text.lower()
    
    if any(w in text_lower for w in ['portrait', 'porträt', 'person', 'gesicht', 'mensch']):
        return 'Portrait'
    elif any(w in text_lower for w in ['landschaft', 'natur', 'himmel', 'feld', 'wald', 'berg']):
        return 'Landschaft'
    elif any(w in text_lower for w in ['architektur', 'gebäude', 'haus', 'kirche', 'brücke']):
        return 'Architektur'
    elif any(w in text_lower for w in ['street', 'straße', 'stadt', 'urban']):
        return 'Street'
    elif any(w in text_lower for w in ['makro', 'nahaufnahme', 'detail', 'blume', 'insekt']):
        return 'Makro'
    elif any(w in text_lower for w in ['sport', 'bewegung', 'action']):
        return 'Sport'
    else:
        return 'Allgemein'


def extract_settings(analysis_text):
    """Extrahiert Kameraeinstellungen aus der Analyse."""
    settings = {}
    
    # Einfache Extraktion basierend auf Schlüsselwörtern
    lines = analysis_text.split('\n')
    in_settings = False
    
    for line in lines:
        line = line.strip()
        if 'einstellung' in line.lower() or 'sony' in line.lower():
            in_settings = True
        
        if in_settings:
            if 'iso' in line.lower():
                settings['ISO'] = extract_value(line)
            elif 'blende' in line.lower() or 'f/' in line.lower():
                settings['Blende'] = extract_value(line)
            elif 'verschluss' in line.lower() or '1/' in line:
                settings['Verschlusszeit'] = extract_value(line)
            elif 'modus' in line.lower() and any(m in line for m in ['A', 'S', 'M', 'P']):
                settings['Modus'] = extract_value(line)
            elif 'autofokus' in line.lower() or 'af-' in line.lower():
                settings['AF-Modus'] = extract_value(line)
            elif 'weißabgleich' in line.lower():
                settings['Weißabgleich'] = extract_value(line)
    
    # Fallback-Einstellungen
    if not settings:
        settings = {
            'Modus': 'A (Blende)',
            'ISO': '100-400',
            'Blende': 'f/2.8-8.0',
            'Verschlusszeit': '1/125-1/500',
            'AF-Modus': 'AF-S',
            'Weißabgleich': 'Auto'
        }
    
    return settings


def extract_value(line):
    """Extrahiert den Wert aus einer Einstellungszeile."""
    # Entferne Markdown-Formatierung und Präfixe
    line = line.replace('**', '').replace('*', '')
    
    # Suche nach Doppelpunkt
    if ':' in line:
        return line.split(':', 1)[1].strip()
    elif '-' in line:
        parts = line.split('-', 1)
        if len(parts) > 1:
            return parts[1].strip()
    
    return line.strip()


def calculate_confidence(context_parts):
    """Berechnet einen Konfidenzwert basierend auf der Buchübereinstimmung."""
    if not context_parts:
        return 65
    
    avg_similarity = sum(c['similarity'] for c in context_parts) / len(context_parts)
    confidence = int(60 + avg_similarity * 40)
    return min(99, max(60, confidence))


# ============================================================
# API-Endpunkte
# ============================================================

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Hauptendpunkt für die Bildanalyse."""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'Kein Bild übermittelt'}), 400
        
        image_data = data['image']
        user_question = data.get('question', None)
        
        # Base64-Präfix entfernen falls vorhanden
        if ',' in image_data:
            image_base64 = image_data.split(',')[1]
        else:
            image_base64 = image_data
        
        logger.info(f"Analyse gestartet. Frage: {user_question}")
        
        result = analyze_photo_with_rag(image_base64, user_question)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Fehler bei Analyse: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/add-book', methods=['POST'])
def add_book():
    """Endpunkt zum Hinzufügen eines neuen Buchs."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Keine Datei übermittelt'}), 400
        
        file = request.files['file']
        book_name = request.form.get('name', 'neues_buch')
        book_category = request.form.get('category', 'fotografie')
        
        # Datei speichern
        upload_dir = '/home/ubuntu/photocoach-ai/data/uploads'
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)
        
        # Buch verarbeiten
        import sys
        sys.path.insert(0, '/home/ubuntu/photocoach-ai')
        from extract_epub import process_book
        
        chunks = process_book(file_path, book_name, book_category)
        
        # Chunks speichern
        chunks_file = f'/home/ubuntu/photocoach-ai/data/{book_name}_chunks.json'
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        # RAG-System neu aufbauen
        from build_rag import build_collection
        
        chunks_files = []
        for f in os.listdir('/home/ubuntu/photocoach-ai/data'):
            if f.endswith('_chunks.json'):
                chunks_files.append(f'/home/ubuntu/photocoach-ai/data/{f}')
        
        build_collection("photocoach_books", chunks_files)
        
        return jsonify({
            'success': True,
            'message': f'Buch "{book_name}" erfolgreich hinzugefügt',
            'chunks': len(chunks)
        })
    
    except Exception as e:
        logger.error(f"Fehler beim Hinzufügen des Buchs: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/books', methods=['GET'])
def get_books():
    """Gibt eine Liste aller verfügbaren Bücher zurück."""
    try:
        collection = chroma_client.get_collection("photocoach_books")
        
        # Alle einzigartigen Bücher ermitteln
        results = collection.get(include=['metadatas'])
        
        books = {}
        for meta in results['metadatas']:
            book = meta.get('book', 'Unbekannt')
            if book not in books:
                books[book] = {
                    'name': book,
                    'displayName': book.replace('_', ' ').title(),
                    'category': meta.get('category', 'allgemein'),
                    'chunks': 0
                }
            books[book]['chunks'] += 1
        
        return jsonify({
            'success': True,
            'books': list(books.values()),
            'totalChunks': collection.count()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health-Check Endpunkt."""
    try:
        collection = chroma_client.get_collection("photocoach_books")
        return jsonify({
            'status': 'ok',
            'rag_chunks': collection.count(),
            'model': 'gpt-4.1-mini'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_static(path):
    """Statische Dateien servieren."""
    return send_from_directory('/home/ubuntu/photocoach-ai', path)


if __name__ == '__main__':
    logger.info("PhotoCoach AI Server startet...")
    app.run(host='0.0.0.0', port=5001, debug=False)
