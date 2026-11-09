"""
Recipe Preparation Agent — Streamlit UI
IBM Granite + RAG powered cooking assistant.
Deploy: https://streamlit.io/cloud
"""

import os
import streamlit as st

# ── Inject Streamlit secrets into env BEFORE importing the agent ──────────────
# Streamlit Cloud stores secrets in st.secrets; locally you can use a .env file.
for key in [
    "WATSONX_API_KEY",
    "WATSONX_PROJECT_ID",
    "WATSONX_URL",
    "GRANITE_MODEL_ID",
    "MAX_NEW_TOKENS",
    "LLM_TEMPERATURE",
    "RAG_TOP_K",
    "MIN_SIMILARITY_SCORE",
    "RECIPE_DATA_PATH",
]:
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

from recipe_agent import RecipeAgent  # noqa: E402  (must come after env setup)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recipe Preparation Agent",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; }
        .recipe-card {
            background: #f7f8fa;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
        }
        .match-badge {
            display: inline-block;
            background: #3b82d4;
            color: white;
            border-radius: 4px;
            padding: 1px 8px;
            font-size: 0.78rem;
            margin-right: 4px;
        }
        .missing-badge {
            display: inline-block;
            background: #e5e7eb;
            color: #57606a;
            border-radius: 4px;
            padding: 1px 8px;
            font-size: 0.78rem;
            margin-right: 4px;
        }
        .status-dot-green { color: #16a34a; font-size: 1.1rem; }
        .status-dot-red   { color: #dc2626; font-size: 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached agent initialisation ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising Recipe Agent…")
def get_agent() -> RecipeAgent:
    return RecipeAgent()


agent = get_agent()
status = agent.get_status()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🍳 Recipe Agent")
    st.caption("IBM Granite · RAG powered")

    st.divider()

    # Status indicators
    st.markdown("**Agent Status**")
    granite_ok = status.get("granite_connected", False)
    vector_ok  = status.get("vector_search", False)
    recipes_n  = status.get("recipes_loaded", 0)

    col1, col2 = st.columns(2)
    col1.metric("Recipes", recipes_n)
    col2.metric("RAG Mode", "Vector" if vector_ok else "Keyword")

    if granite_ok:
        st.markdown('<span class="status-dot-green">●</span> IBM Granite connected', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot-red">●</span> Demo mode (no API key)', unsafe_allow_html=True)
        st.info(
            "Add **WATSONX_API_KEY** and **WATSONX_PROJECT_ID** to Streamlit secrets "
            "to enable live IBM Granite responses.",
            icon="ℹ️",
        )

    st.divider()

    # Dietary preferences
    st.markdown("**Dietary Preferences**")
    dietary_options = ["vegetarian", "vegan", "gluten-free", "dairy-free", "keto", "halal"]
    dietary = st.multiselect(
        "Select all that apply",
        dietary_options,
        label_visibility="collapsed",
    )

    st.divider()

    # Navigation
    page = st.radio(
        "Navigate",
        ["🔍 Find Recipes", "🔄 Ingredient Substitution", "💡 Cooking Tips", "📖 Browse All Recipes"],
        label_visibility="collapsed",
    )


# ── Helper: render recipe card ────────────────────────────────────────────────
def render_recipe_card(recipe: dict, idx: int):
    match_pct = recipe.get("match_percentage", 0)
    matched   = recipe.get("matched_ingredients", [])
    missing   = recipe.get("missing_ingredients", [])

    with st.container():
        st.markdown(
            f"""<div class="recipe-card">
            <strong>#{idx} {recipe['name']}</strong>
            &nbsp;·&nbsp; {recipe.get('cuisine','')}
            &nbsp;·&nbsp; {recipe.get('difficulty','').capitalize()}
            &nbsp;·&nbsp; ⏱ {recipe.get('time_minutes','')} min
            &nbsp;·&nbsp; 🥗 {', '.join(recipe.get('diet',[]))}
            <br/><br/>
            <strong>Match:</strong> {match_pct}% &nbsp;
            {"".join(f'<span class="match-badge">{i}</span>' for i in matched)}
            {"".join(f'<span class="missing-badge">missing: {i}</span>' for i in missing)}
            </div>""",
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Find Recipes
# ════════════════════════════════════════════════════════════════════════════
if page == "🔍 Find Recipes":
    st.title("🔍 Find Recipes")
    st.caption("Enter the ingredients you have and get AI-powered recipe recommendations.")

    # Ingredient input
    all_ingredients = agent.rag.get_unique_ingredients()
    selected_ings = st.multiselect(
        "Pick from known ingredients",
        all_ingredients,
        placeholder="Type to search…",
    )

    extra_raw = st.text_input(
        "…or type additional ingredients (comma-separated)",
        placeholder="e.g. eggs, butter, lemon",
    )
    extra_ings = [i.strip() for i in extra_raw.split(",") if i.strip()]
    ingredients = list(dict.fromkeys(selected_ings + extra_ings))  # dedup, preserve order

    user_query = st.text_input(
        "Optional: describe what you'd like to cook",
        placeholder="e.g. quick weeknight dinner, something spicy, healthy breakfast",
    )

    search_btn = st.button("🍽️ Find Recipes", type="primary", use_container_width=True)

    if search_btn:
        if not ingredients:
            st.warning("Please select or type at least one ingredient.")
        else:
            with st.spinner("Searching recipes and generating AI guidance…"):
                result = agent.find_recipes(
                    ingredients=ingredients,
                    dietary_preferences=dietary if dietary else None,
                    user_query=user_query,
                )

            if not result.get("success"):
                st.error(result.get("error", "Something went wrong."))
            else:
                recipes = result.get("recipes", [])
                ai_resp = result.get("ai_response", "")

                # ── AI response ──
                st.subheader("🤖 IBM Granite AI Guidance")
                st.markdown(ai_resp)

                # ── Matched recipes ──
                if recipes:
                    st.subheader(f"📋 Matched Recipes ({len(recipes)})")
                    for i, r in enumerate(recipes, 1):
                        render_recipe_card(r, i)
                        with st.expander(f"Full instructions — {r['name']}"):
                            for j, step in enumerate(r.get("instructions", []), 1):
                                st.markdown(f"**{j}.** {step}")
                            if r.get("tips"):
                                st.markdown("**Tips:**")
                                for tip in r["tips"]:
                                    st.markdown(f"- {tip}")
                            if r.get("nutrition"):
                                n = r["nutrition"]
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("Calories", n.get("calories", "—"))
                                c2.metric("Protein",  n.get("protein", "—"))
                                c3.metric("Carbs",    n.get("carbs", "—"))
                                c4.metric("Fat",      n.get("fat", "—"))
                else:
                    st.info("No recipes found. Try adding more common pantry items like garlic, oil, or onion.")

                # ── Meta info ──
                with st.expander("ℹ️ Search metadata"):
                    st.json({
                        "ingredients_used": result.get("ingredients_used"),
                        "dietary_preferences": result.get("dietary_preferences"),
                        "rag_mode": result.get("rag_mode"),
                        "granite_connected": result.get("granite_connected"),
                        "model": result.get("model_info", {}).get("model_id"),
                    })


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Ingredient Substitution
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Ingredient Substitution":
    st.title("🔄 Ingredient Substitution")
    st.caption("Missing an ingredient? Get AI-powered swap recommendations.")

    all_recipes_list = agent.rag.get_all_recipes()
    recipe_names = {r["id"]: r["name"] for r in all_recipes_list}

    selected_recipe_id = st.selectbox(
        "Choose a recipe",
        list(recipe_names.keys()),
        format_func=lambda rid: recipe_names[rid],
    )

    missing_ing = st.text_input("Missing ingredient", placeholder="e.g. parsley")

    available_raw = st.text_input(
        "Ingredients you have available (comma-separated)",
        placeholder="e.g. basil, chives, cilantro",
    )
    available_ings = [i.strip() for i in available_raw.split(",") if i.strip()]

    sub_btn = st.button("💡 Get Substitution", type="primary", use_container_width=True)

    if sub_btn:
        if not missing_ing.strip():
            st.warning("Please enter the missing ingredient.")
        else:
            with st.spinner("Consulting IBM Granite for substitution advice…"):
                result = agent.get_substitution(
                    recipe_id=selected_recipe_id,
                    missing_ingredient=missing_ing.strip(),
                    available_ingredients=available_ings,
                )

            if not result.get("success"):
                st.error(result.get("error"))
            else:
                st.subheader(f"Substitution for **{missing_ing}** in *{result['recipe_name']}*")

                if result.get("known_substitutions"):
                    st.markdown("**Known substitutions from the knowledge base:**")
                    st.markdown(
                        " · ".join(f"`{s}`" for s in result["known_substitutions"])
                    )

                st.subheader("🤖 IBM Granite Advice")
                st.markdown(result.get("ai_advice", ""))


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Cooking Tips
# ════════════════════════════════════════════════════════════════════════════
elif page == "💡 Cooking Tips":
    st.title("💡 Cooking Tips")
    st.caption("Get expert AI-generated tips for any recipe.")

    all_recipes_list = agent.rag.get_all_recipes()
    recipe_names = {r["id"]: r["name"] for r in all_recipes_list}

    selected_recipe_id = st.selectbox(
        "Choose a recipe",
        list(recipe_names.keys()),
        format_func=lambda rid: recipe_names[rid],
    )

    tips_btn = st.button("✨ Get Cooking Tips", type="primary", use_container_width=True)

    if tips_btn:
        with st.spinner("Generating expert cooking tips…"):
            result = agent.get_cooking_tips(selected_recipe_id)

        if not result.get("success"):
            st.error(result.get("error"))
        else:
            st.subheader(f"Tips for *{result['recipe_name']}*")

            if result.get("knowledge_base_tips"):
                st.markdown("**Knowledge base tips:**")
                for tip in result["knowledge_base_tips"]:
                    st.markdown(f"- {tip}")

            st.subheader("🤖 IBM Granite Expert Tips")
            st.markdown(result.get("ai_tips", ""))


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Browse All Recipes
# ════════════════════════════════════════════════════════════════════════════
elif page == "📖 Browse All Recipes":
    st.title("📖 Browse All Recipes")
    st.caption("Explore the full recipe knowledge base.")

    all_recipes_list = agent.rag.get_all_recipes()

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    cuisine_options = sorted({r.get("cuisine", "") for r in all_recipes_list if r.get("cuisine")})
    diet_options    = sorted({d for r in all_recipes_list for d in r.get("diet", [])})
    diff_options    = sorted({r.get("difficulty", "") for r in all_recipes_list if r.get("difficulty")})

    filter_cuisine = col_f1.multiselect("Cuisine", cuisine_options)
    filter_diet    = col_f2.multiselect("Diet", diet_options)
    filter_diff    = col_f3.multiselect("Difficulty", diff_options)

    # Apply filters
    filtered = all_recipes_list
    if filter_cuisine:
        filtered = [r for r in filtered if r.get("cuisine") in filter_cuisine]
    if filter_diet:
        filtered = [r for r in filtered if any(d in r.get("diet", []) for d in filter_diet)]
    if filter_diff:
        filtered = [r for r in filtered if r.get("difficulty") in filter_diff]

    st.markdown(f"**Showing {len(filtered)} / {len(all_recipes_list)} recipes**")

    for recipe in filtered:
        with st.expander(
            f"**{recipe['name']}** · {recipe.get('cuisine','')} · "
            f"{recipe.get('difficulty','').capitalize()} · ⏱ {recipe.get('time_minutes','')} min"
        ):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("**Ingredients:**")
                for ing in recipe.get("ingredients", []):
                    st.markdown(f"- {ing['name']} — {ing['quantity']}")
            with c2:
                st.markdown("**Diet:** " + ", ".join(recipe.get("diet", [])))
                st.markdown(f"**Servings:** {recipe.get('servings', '—')}")
                if recipe.get("nutrition"):
                    n = recipe["nutrition"]
                    st.markdown(
                        f"**Nutrition:** {n.get('calories','')} kcal · "
                        f"P {n.get('protein','')} · C {n.get('carbs','')} · F {n.get('fat','')}"
                    )
            st.markdown("**Instructions:**")
            for j, step in enumerate(recipe.get("instructions", []), 1):
                st.markdown(f"{j}. {step}")
            if recipe.get("tips"):
                st.markdown("**Tips:** " + " | ".join(recipe["tips"]))


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Recipe Preparation Agent · IBM Granite + RAG · Built for AICTE")
