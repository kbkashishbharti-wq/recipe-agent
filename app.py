"""
Flask API Server — Recipe Preparation Agent
Provides REST endpoints consumed by the frontend.
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from recipe_agent import RecipeAgent

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# Initialize the agent once at startup
agent = RecipeAgent()


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ---------------------------------------------------------------------------
# Health / Status
# ---------------------------------------------------------------------------

@app.route("/api/status", methods=["GET"])
def status():
    """Return agent health and configuration."""
    return jsonify({"ok": True, **agent.get_status()})


# ---------------------------------------------------------------------------
# Core recipe finding endpoint
# ---------------------------------------------------------------------------

@app.route("/api/recipes", methods=["POST"])
def find_recipes():
    """
    POST /api/recipes
    Body: {
        "ingredients": ["egg", "flour", "milk"],
        "dietary_preferences": ["vegetarian"],   // optional
        "query": "quick breakfast"               // optional free-text
    }
    """
    data = request.get_json(silent=True) or {}

    ingredients = data.get("ingredients", [])
    if isinstance(ingredients, str):
        # Accept comma-separated string as well
        ingredients = [i.strip() for i in ingredients.split(",") if i.strip()]

    dietary = data.get("dietary_preferences", [])
    query = data.get("query", "").strip()

    if not ingredients:
        return jsonify({"success": False, "error": "Please provide at least one ingredient."}), 400

    result = agent.find_recipes(
        ingredients=ingredients,
        dietary_preferences=dietary if dietary else None,
        user_query=query,
    )

    return jsonify(result)


# ---------------------------------------------------------------------------
# Substitution advisor
# ---------------------------------------------------------------------------

@app.route("/api/substitution", methods=["POST"])
def substitution():
    """
    POST /api/substitution
    Body: {
        "recipe_id": "R001",
        "missing_ingredient": "parsley",
        "available_ingredients": ["basil", "chives"]
    }
    """
    data = request.get_json(silent=True) or {}
    recipe_id = data.get("recipe_id", "")
    missing = data.get("missing_ingredient", "")
    available = data.get("available_ingredients", [])

    if not recipe_id or not missing:
        return jsonify({"success": False, "error": "recipe_id and missing_ingredient required."}), 400

    result = agent.get_substitution(recipe_id, missing, available)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Cooking tips
# ---------------------------------------------------------------------------

@app.route("/api/tips", methods=["POST"])
def tips():
    """
    POST /api/tips
    Body: { "recipe_id": "R001" }
    """
    data = request.get_json(silent=True) or {}
    recipe_id = data.get("recipe_id", "")

    if not recipe_id:
        return jsonify({"success": False, "error": "recipe_id required."}), 400

    result = agent.get_cooking_tips(recipe_id)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Browse all recipes
# ---------------------------------------------------------------------------

@app.route("/api/all-recipes", methods=["GET"])
def all_recipes():
    """Return all recipes in the knowledge base."""
    return jsonify({"recipes": agent.rag.get_all_recipes()})


# ---------------------------------------------------------------------------
# Ingredient autocomplete
# ---------------------------------------------------------------------------

@app.route("/api/ingredients", methods=["GET"])
def ingredients():
    """Return all unique ingredient names for autocomplete."""
    return jsonify({"ingredients": agent.rag.get_unique_ingredients()})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    print(f"\n{'='*60}")
    print(f"  Recipe Preparation Agent")
    print(f"  IBM Granite + RAG powered cooking assistant")
    print(f"  Server: http://{host}:{port}")
    print(f"{'='*60}\n")

    app.run(host=host, port=port, debug=debug)
