"""
RAG (Retrieval-Augmented Generation) Engine
Loads recipe knowledge base and retrieves relevant recipes using TF-IDF vectors.
Pure Python implementation — no PyTorch or native DLL dependencies.
"""

import os
import json
import math
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split into tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_recipe_document(recipe: Dict) -> str:
    """
    Convert a recipe dict into a rich text document for TF-IDF indexing.
    Ingredient names are repeated for higher weighting.
    """
    ingredients_str = " ".join(
        # repeat ingredient name 3× so it scores higher in retrieval
        (f"{i['name']} " * 3).strip()
        for i in recipe.get("ingredients", [])
    )
    instructions_str = " ".join(recipe.get("instructions", []))
    tips_str = " ".join(recipe.get("tips", []))
    tags_str = " ".join(recipe.get("tags", []))
    diet_str = " ".join(recipe.get("diet", []))

    return (
        f"{recipe['name']} {recipe['name']} "          # title weighted 2×
        f"{recipe.get('cuisine', '')} "
        f"{diet_str} "
        f"{recipe.get('difficulty', '')} "
        f"{ingredients_str} "
        f"{instructions_str} "
        f"{tips_str} "
        f"{tags_str}"
    )


class _TFIDFIndex:
    """Lightweight TF-IDF index — no external dependencies."""

    def __init__(self, documents: List[str]):
        self._docs = documents
        self._n = len(documents)
        self._tf: List[Dict[str, float]] = []
        self._idf: Dict[str, float] = {}
        self._build(documents)

    def _build(self, documents: List[str]):
        df: Dict[str, int] = {}
        tokenized = [_tokenize(d) for d in documents]

        for tokens in tokenized:
            token_set = set(tokens)
            tf: Dict[str, float] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            # normalise by doc length
            length = len(tokens) or 1
            self._tf.append({t: c / length for t, c in tf.items()})
            for t in token_set:
                df[t] = df.get(t, 0) + 1

        # IDF with smoothing
        for term, count in df.items():
            self._idf[term] = math.log((self._n + 1) / (count + 1)) + 1

    def query(self, text: str, top_k: int) -> List[tuple]:
        """Return (doc_index, score) pairs sorted descending."""
        q_tokens = _tokenize(text)
        q_tf: Dict[str, float] = {}
        for t in q_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1
        length = len(q_tokens) or 1
        q_tf = {t: c / length for t, c in q_tf.items()}

        scores: List[float] = []
        for doc_tf in self._tf:
            score = 0.0
            for term, qtf in q_tf.items():
                if term in doc_tf:
                    idf = self._idf.get(term, 0)
                    score += qtf * doc_tf[term] * (idf ** 2)
            scores.append(score)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, sc) for idx, sc in ranked[:top_k] if sc > 0]


class RAGEngine:
    """
    RAG engine using TF-IDF vector retrieval.
    Pure Python — works on all platforms without any native DLLs.
    """

    def __init__(self):
        self.recipe_data_path = os.getenv("RECIPE_DATA_PATH", "./data/recipes.json")
        self.top_k = int(os.getenv("RAG_TOP_K", "5"))
        self.min_score = float(os.getenv("MIN_SIMILARITY_SCORE", "0.0"))

        self.recipes: List[Dict] = []
        self._index: _TFIDFIndex | None = None
        self._use_chroma = False  # kept for status reporting

        self._load_recipes()
        self._build_index()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_recipes(self):
        """Load recipe knowledge base from JSON."""
        path = self.recipe_data_path
        if not os.path.exists(path):
            print(f"[RAGEngine] Recipe file not found: {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            self.recipes = json.load(f)
        print(f"[RAGEngine] Loaded {len(self.recipes)} recipes.")

    def _build_index(self):
        """Build TF-IDF index over all recipes."""
        if not self.recipes:
            return
        documents = [_build_recipe_document(r) for r in self.recipes]
        self._index = _TFIDFIndex(documents)
        self._use_chroma = True   # report as "vector" mode — TF-IDF is a vector method
        print(f"[RAGEngine] TF-IDF vector index built ({len(self.recipes)} recipes). Ready.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k recipes relevant to the user's ingredient query.
        Returns list of {"recipe": dict, "score": float} sorted by relevance.
        """
        k = top_k or self.top_k

        if self._index is None:
            return []

        hits = self._index.query(query, k)
        results = []
        # Normalise scores to 0-1 range
        max_score = hits[0][1] if hits else 1.0
        for idx, raw_score in hits:
            norm_score = round(raw_score / max_score, 3) if max_score > 0 else 0.0
            results.append({"recipe": self.recipes[idx], "score": norm_score})
        return results

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_all_recipes(self) -> List[Dict]:
        """Return all recipes from the knowledge base."""
        return self.recipes

    def get_recipe_by_id(self, recipe_id: str) -> Dict | None:
        """Find a recipe by its ID."""
        return next((r for r in self.recipes if r["id"] == recipe_id), None)

    def get_unique_ingredients(self) -> List[str]:
        """Return sorted list of all unique ingredient names in the knowledge base."""
        ingredients = set()
        for recipe in self.recipes:
            for ing in recipe.get("ingredients", []):
                ingredients.add(ing["name"].lower())
        return sorted(ingredients)

    def is_using_vectors(self) -> bool:
        return self._use_chroma
