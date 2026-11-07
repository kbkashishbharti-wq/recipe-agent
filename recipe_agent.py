"""
Recipe Preparation Agent — Core Logic
Orchestrates RAG retrieval + IBM Granite LLM to answer cooking queries.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from rag_engine import RAGEngine
from granite_client import GraniteClient

load_dotenv()


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

RECIPE_PROMPT_TEMPLATE = """You are a professional chef and cooking assistant powered by IBM Granite.
A user has the following ingredients available: {ingredients}

Based on the retrieved recipe context below, provide:
1. Which recipe(s) best match their available ingredients
2. Step-by-step cooking instructions adapted to what they have
3. Smart substitutions for any missing ingredients
4. Useful cooking tips
5. Dietary information and adjustments if relevant

User request: {user_query}

Retrieved Recipe Context:
{context}

{dietary_note}

Provide a helpful, practical, and encouraging cooking guide. Be specific with quantities and timings.
Format your response clearly with sections. Start directly with the recipe recommendation.
"""

SUBSTITUTION_PROMPT_TEMPLATE = """You are a professional chef. A user is missing the following ingredient: "{missing_ingredient}"
from the recipe "{recipe_name}".

Known substitutions for this recipe: {known_substitutions}

User's available ingredients: {available_ingredients}

Suggest the best substitution from what the user actually has, explaining:
- How to substitute (ratio/quantity adjustments)
- How it will affect the dish's flavor/texture
- Any technique modifications needed

Be concise and practical.
"""

TIPS_PROMPT_TEMPLATE = """You are a professional chef. For the recipe "{recipe_name}", 
provide 3 expert cooking tips that will help a home cook get the best results.
Consider: technique, common mistakes to avoid, flavor enhancement, and presentation.
Keep each tip to 1-2 sentences. Be specific and actionable.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RecipeAgent:
    """
    The main Recipe Preparation Agent.
    Combines RAG retrieval with IBM Granite generation.
    """

    def __init__(self):
        print("[RecipeAgent] Initializing...")
        self.rag = RAGEngine()
        self.llm = GraniteClient()
        print(
            f"[RecipeAgent] Ready. "
            f"Vector search: {self.rag.is_using_vectors()}, "
            f"IBM Granite: {self.llm.is_ready()}"
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def find_recipes(
        self,
        ingredients: List[str],
        dietary_preferences: List[str] = None,
        user_query: str = "",
    ) -> Dict[str, Any]:
        """
        Given a list of ingredients and optional preferences, retrieve relevant recipes
        and generate AI-powered cooking guidance.

        Returns a structured response dict.
        """
        if not ingredients:
            return {
                "success": False,
                "error": "Please provide at least one ingredient.",
                "recipes": [],
                "ai_response": "",
            }

        # Build enriched search query
        query = self._build_search_query(ingredients, dietary_preferences, user_query)

        # --- RAG Retrieval ---
        retrieved = self.rag.retrieve(query)

        if not retrieved:
            return {
                "success": True,
                "recipes": [],
                "ai_response": (
                    "I couldn't find any recipes matching your available ingredients. "
                    "Try adding more common pantry items like garlic, onion, oil, or salt."
                ),
                "ingredients_used": ingredients,
                "rag_mode": self.rag.is_using_vectors(),
            }

        # --- Build LLM context ---
        context = self._build_context(retrieved)
        dietary_note = self._build_dietary_note(dietary_preferences)

        prompt = RECIPE_PROMPT_TEMPLATE.format(
            ingredients=", ".join(ingredients),
            user_query=user_query or f"What can I make with {', '.join(ingredients[:3])}?",
            context=context,
            dietary_note=dietary_note,
        )

        # --- IBM Granite Generation ---
        ai_response = self.llm.generate(prompt)

        # --- Compute ingredient match scores ---
        recipes_with_scores = self._score_ingredient_matches(retrieved, ingredients)

        return {
            "success": True,
            "recipes": recipes_with_scores,
            "ai_response": ai_response,
            "ingredients_used": ingredients,
            "dietary_preferences": dietary_preferences or [],
            "rag_mode": "semantic" if self.rag.is_using_vectors() else "keyword",
            "granite_connected": self.llm.is_ready(),
            "model_info": self.llm.get_model_info(),
        }

    # ------------------------------------------------------------------
    # Substitution advisor
    # ------------------------------------------------------------------

    def get_substitution(
        self,
        recipe_id: str,
        missing_ingredient: str,
        available_ingredients: List[str],
    ) -> Dict[str, Any]:
        """Get AI-powered substitution advice for a missing ingredient."""
        recipe = self.rag.get_recipe_by_id(recipe_id)
        if not recipe:
            return {"success": False, "error": "Recipe not found."}

        # Get known substitutions from knowledge base
        known_subs = recipe.get("substitutions", {}).get(missing_ingredient.lower(), [])
        known_subs_str = ", ".join(known_subs) if known_subs else "none documented"

        prompt = SUBSTITUTION_PROMPT_TEMPLATE.format(
            missing_ingredient=missing_ingredient,
            recipe_name=recipe["name"],
            known_substitutions=known_subs_str,
            available_ingredients=", ".join(available_ingredients) if available_ingredients else "not specified",
        )

        ai_response = self.llm.generate(prompt)

        return {
            "success": True,
            "recipe_name": recipe["name"],
            "missing_ingredient": missing_ingredient,
            "known_substitutions": known_subs,
            "ai_advice": ai_response,
        }

    # ------------------------------------------------------------------
    # Cooking tips
    # ------------------------------------------------------------------

    def get_cooking_tips(self, recipe_id: str) -> Dict[str, Any]:
        """Get AI-enhanced cooking tips for a specific recipe."""
        recipe = self.rag.get_recipe_by_id(recipe_id)
        if not recipe:
            return {"success": False, "error": "Recipe not found."}

        prompt = TIPS_PROMPT_TEMPLATE.format(recipe_name=recipe["name"])
        ai_tips = self.llm.generate(prompt)

        return {
            "success": True,
            "recipe_name": recipe["name"],
            "knowledge_base_tips": recipe.get("tips", []),
            "ai_tips": ai_tips,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_search_query(
        self, ingredients: List[str], preferences: List[str], user_query: str
    ) -> str:
        parts = []
        if user_query:
            parts.append(user_query)
        parts.extend(ingredients)
        if preferences:
            parts.extend(preferences)
        return " ".join(parts)

    def _build_context(self, retrieved: List[Dict]) -> str:
        """Serialize retrieved recipes into LLM-readable context."""
        sections = []
        for i, item in enumerate(retrieved[:4], 1):  # max 4 in context to stay within token limit
            r = item["recipe"]
            score = item["score"]
            ingredients = ", ".join(
                f"{ing['name']} ({ing['quantity']})" for ing in r.get("ingredients", [])
            )
            instructions = "\n  ".join(
                f"{j+1}. {step}" for j, step in enumerate(r.get("instructions", []))
            )
            subs = r.get("substitutions", {})
            subs_str = "; ".join(f"{k}: {', '.join(v)}" for k, v in subs.items())
            tips = " | ".join(r.get("tips", []))

            sections.append(
                f"[Recipe {i} — Relevance: {score}]\n"
                f"Name: {r['name']} ({r.get('cuisine', '')}) | "
                f"Diet: {', '.join(r.get('diet', []))} | "
                f"Time: {r.get('time_minutes')} min | "
                f"Difficulty: {r.get('difficulty', '')}\n"
                f"Ingredients: {ingredients}\n"
                f"Instructions:\n  {instructions}\n"
                f"Substitutions: {subs_str}\n"
                f"Tips: {tips}\n"
            )
        return "\n---\n".join(sections)

    def _build_dietary_note(self, preferences: List[str]) -> str:
        if not preferences:
            return ""
        return f"User dietary preferences/restrictions: {', '.join(preferences)}. Please tailor advice accordingly."

    def _score_ingredient_matches(
        self, retrieved: List[Dict], user_ingredients: List[str]
    ) -> List[Dict]:
        """Enrich retrieved recipes with ingredient match metadata."""
        user_set = {ing.lower().strip() for ing in user_ingredients}
        result = []

        for item in retrieved:
            recipe = item["recipe"]
            recipe_ings = {ing["name"].lower() for ing in recipe.get("ingredients", [])}

            matched = user_set & recipe_ings
            missing = recipe_ings - user_set

            result.append(
                {
                    **recipe,
                    "relevance_score": item["score"],
                    "matched_ingredients": sorted(matched),
                    "missing_ingredients": sorted(missing),
                    "match_percentage": round(
                        len(matched) / len(recipe_ings) * 100 if recipe_ings else 0, 1
                    ),
                }
            )

        return result

    def get_status(self) -> Dict[str, Any]:
        """Return agent health and configuration status."""
        return {
            "recipes_loaded": len(self.rag.get_all_recipes()),
            "vector_search": self.rag.is_using_vectors(),
            "granite_connected": self.llm.is_ready(),
            "model": self.llm.get_model_info(),
        }
