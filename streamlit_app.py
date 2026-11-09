"""
Recipe Preparation Agent — Streamlit UI
IBM Granite + RAG powered cooking assistant.
Deploy: https://streamlit.io/cloud
"""

import os
import streamlit as st

# ── Page config — MUST be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Recipe Preparation Agent",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject Streamlit secrets into env BEFORE importing the agent ──────────────
# Streamlit Cloud stores secrets in st.secrets; locally python-dotenv reads .env
_SECRET_KEYS = [
    "WATSONX_API_KEY",
    "WATSONX_PROJECT_ID",
    "WATSONX_URL",
    "GRANITE_MODEL_ID",
    "MAX_NEW_TOKENS",
    "LLM_TEMPERATURE",
    "RAG_TOP_K",
    "MIN_SIMILARITY_SCORE",
    "RECIPE_DATA_PATH",
]
try:
    for key in _SECRET_KEYS:
        val = st.secrets.get(key)
        if val:
            os.environ[key] = str(val)
except Exception:
    # No secrets file — fine locally (python-dotenv / .env will be used instead)
    pass

from recipe_agent import RecipeAgent  # noqa: E402  (must come after env setup)

# ── Custom CSS — uses [data-testid] selectors that survive Streamlit Cloud ────
st.markdown(
    """
    <style>
        /* tighten top padding */
        [data-testid="stAppViewContainer"] > .main .block-container {
            padding-top: 1.2rem;
        }
        /* recipe card */
        .recipe-card {
            background: #f7f8fa;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
            font-family: inherit;
        }
        /* ingredient badges */
        .match-badge {
            display: inline-block;
            background: #1d6ae5;
            color: #ffffff !important;
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 0.75rem;
            margin: 2px 3px 2px 0;
        }
        .missing-badge {
            display: inline-block;
            background: #e5e7eb;
            color: #374151 !important;
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 0.75rem;
            margin: 2px 3px 2px 0;
        }
        /* status dots */
        .dot-green { color: #16a34a; font-weight: bold; }
        .dot-red   { color: #dc2626; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached agent — shared across all reruns, never reloaded on Cloud ──────────
@st.cache_resource(show_spinner="⏳ Initialising Recipe Agent…")
def get_agent() -> RecipeAgent:
    return RecipeAgent()


agent = get_agent()
status = agent.get_status()

granite_ok = status.get("granite_connected", False)
vector_ok  = status.get("vector_search", False)
recipes_n  = status.get("recipes_loaded", 0)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🍳 Recipe Agent")
    st.caption("IBM Granite · RAG powered")
    st.divider()

    # Status
    st.markdown("**Agent Status**")
    c1, c2 = st.columns(2)
    c1.metric("Recipes", recipes_n)
    c2.metric("Search", "Vector" if vector_ok else "Keyword")

    if granite_ok:
        st.markdown('<p class="dot-green">● IBM Granite connected</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="dot-red">● Demo mode (no API key)</p>', unsafe_allow_html=True)
        st.info(
            "Add **WATSONX_API_KEY** and **WATSONX_PROJECT_ID** in "
            "Streamlit Cloud → App Settings → Secrets to enable live AI.",
            icon="ℹ️",
        )

    st.divider()

    # Dietary preferences
    st.markdown("**Dietary Preferences**")
    dietary = st.multiselect(
        "diet",
        ["vegetarian", "vegan", "gluten-free", "dairy-free", "keto", "halal"],
        label_visibility="collapsed",
    )

    st.divider()

    # Navigation
    page = st.radio(
        "nav",
        ["🔍 Find Recipes", "🔄 Substitution", "💡 Cooking Tips", "📖 Browse Recipes"],
        label_visibility="collapsed",
    )


# ── Helper: render a recipe result card ──────────────────────────────────────
def recipe_card(recipe: dict, idx: int):
    name       = recipe.get("name", "")
    cuisine    = recipe.get("cuisine", "")
    difficulty = recipe.get("difficulty", "").capitalize()
    time_m     = recipe.get("time_minutes", "")
    diet       = ", ".join(recipe.get("diet", []))
    match_pct  = recipe.get("match_percentage", 0)
    matched    = recipe.get("matched_ingredients", [])
    missing    = recipe.get("missing_ingredients", [])

    matched_html = "".join(f'<span class="match-badge">{i}</span>' for i in matched)
    missing_html = "".join(f'<span class="missing-badge">✗ {i}</span>' for i in missing)

    st.markdown(
        f"""<div class="recipe-card">
        <strong>#{idx}&nbsp; {name}</strong>
        &nbsp;·&nbsp; {cuisine}
        &nbsp;·&nbsp; {difficulty}
        &nbsp;·&nbsp; ⏱&nbsp;{time_m} min
        &nbsp;·&nbsp; 🥗&nbsp;{diet}
        <br/><br/>
        <strong>Ingredient match:&nbsp;{match_pct}%</strong><br/>
        {matched_html}
        {missing_html}
        </div>""",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Find Recipes
# ════════════════════════════════════════════════════════════════════════════
if page == "🔍 Find Recipes":
    st.title("🔍 Find Recipes")
    st.caption("Enter the ingredients you have and get AI-powered recipe recommendations.")

    all_ing = agent.rag.get_unique_ingredients()

    col_a, col_b = st.columns([3, 2])
    with col_a:
        selected_ings = st.multiselect(
            "Pick from known ingredients",
            all_ing,
            placeholder="Type to search…",
        )
    with col_b:
        extra_raw = st.text_input(
            "Or type extra ingredients (comma-separated)",
            placeholder="eggs, butter, lemon",
        )

    extra_ings  = [i.strip() for i in extra_raw.split(",") if i.strip()]
    ingredients = list(dict.fromkeys(selected_ings + extra_ings))

    user_query = st.text_input(
        "Describe what you'd like to cook (optional)",
        placeholder="quick weeknight dinner, something spicy, healthy breakfast…",
    )

    if st.button("🍽️ Find Recipes", type="primary", use_container_width=True):
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

                st.subheader("🤖 IBM Granite AI Guidance")
                st.markdown(ai_resp)

                if recipes:
                    st.subheader(f"📋 Matched Recipes ({len(recipes)})")
                    for i, r in enumerate(recipes, 1):
                        recipe_card(r, i)
                        with st.expander(f"Full instructions — {r['name']}"):
                            for j, step in enumerate(r.get("instructions", []), 1):
                                st.markdown(f"**{j}.** {step}")
                            if r.get("tips"):
                                st.markdown("**Tips:**")
                                for tip in r["tips"]:
                                    st.markdown(f"- {tip}")
                            if r.get("nutrition"):
                                n = r["nutrition"]
                                nc1, nc2, nc3, nc4 = st.columns(4)
                                nc1.metric("Calories", n.get("calories", "—"))
                                nc2.metric("Protein",  n.get("protein",  "—"))
                                nc3.metric("Carbs",    n.get("carbs",    "—"))
                                nc4.metric("Fat",      n.get("fat",      "—"))
                else:
                    st.info("No recipes found. Try adding more common pantry items like garlic, oil, or onion.")

                with st.expander("ℹ️ Search metadata"):
                    st.json({
                        "ingredients_used":    result.get("ingredients_used"),
                        "dietary_preferences": result.get("dietary_preferences"),
                        "rag_mode":            result.get("rag_mode"),
                        "granite_connected":   result.get("granite_connected"),
                        "model":               result.get("model_info", {}).get("model_id"),
                    })


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Ingredient Substitution
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Substitution":
    st.title("🔄 Ingredient Substitution")
    st.caption("Missing an ingredient? Get AI-powered swap recommendations.")

    all_recipes_list = agent.rag.get_all_recipes()
    recipe_map       = {r["id"]: r["name"] for r in all_recipes_list}

    selected_id = st.selectbox(
        "Choose a recipe",
        list(recipe_map.keys()),
        format_func=lambda rid: recipe_map[rid],
    )
    missing_ing = st.text_input("Missing ingredient", placeholder="e.g. parsley")
    avail_raw   = st.text_input(
        "Ingredients you have (comma-separated)",
        placeholder="basil, chives, cilantro",
    )
    avail_ings = [i.strip() for i in avail_raw.split(",") if i.strip()]

    if st.button("💡 Get Substitution", type="primary", use_container_width=True):
        if not missing_ing.strip():
            st.warning("Please enter the missing ingredient.")
        else:
            with st.spinner("Consulting IBM Granite…"):
                result = agent.get_substitution(
                    recipe_id=selected_id,
                    missing_ingredient=missing_ing.strip(),
                    available_ingredients=avail_ings,
                )
            if not result.get("success"):
                st.error(result.get("error"))
            else:
                st.subheader(f"Substitution for **{missing_ing}** in *{result['recipe_name']}*")
                if result.get("known_substitutions"):
                    st.markdown("**Known substitutions (knowledge base):**")
                    st.markdown(" · ".join(f"`{s}`" for s in result["known_substitutions"]))
                st.subheader("🤖 IBM Granite Advice")
                st.markdown(result.get("ai_advice", ""))


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Cooking Tips
# ════════════════════════════════════════════════════════════════════════════
elif page == "💡 Cooking Tips":
    st.title("💡 Cooking Tips")
    st.caption("Get expert AI-generated tips for any recipe.")

    all_recipes_list = agent.rag.get_all_recipes()
    recipe_map       = {r["id"]: r["name"] for r in all_recipes_list}

    selected_id = st.selectbox(
        "Choose a recipe",
        list(recipe_map.keys()),
        format_func=lambda rid: recipe_map[rid],
    )

    if st.button("✨ Get Cooking Tips", type="primary", use_container_width=True):
        with st.spinner("Generating expert cooking tips…"):
            result = agent.get_cooking_tips(selected_id)
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
elif page == "📖 Browse Recipes":
    st.title("📖 Browse All Recipes")
    st.caption("Explore the full recipe knowledge base.")

    all_recipes_list = agent.rag.get_all_recipes()

    cf1, cf2, cf3 = st.columns(3)
    cuisine_opts = sorted({r.get("cuisine", "") for r in all_recipes_list if r.get("cuisine")})
    diet_opts    = sorted({d for r in all_recipes_list for d in r.get("diet", [])})
    diff_opts    = sorted({r.get("difficulty", "") for r in all_recipes_list if r.get("difficulty")})

    f_cuisine = cf1.multiselect("Cuisine",    cuisine_opts)
    f_diet    = cf2.multiselect("Diet",       diet_opts)
    f_diff    = cf3.multiselect("Difficulty", diff_opts)

    filtered = all_recipes_list
    if f_cuisine: filtered = [r for r in filtered if r.get("cuisine")    in f_cuisine]
    if f_diet:    filtered = [r for r in filtered if any(d in r.get("diet", []) for d in f_diet)]
    if f_diff:    filtered = [r for r in filtered if r.get("difficulty") in f_diff]

    st.markdown(f"**Showing {len(filtered)} / {len(all_recipes_list)} recipes**")

    for recipe in filtered:
        with st.expander(
            f"**{recipe['name']}**  ·  {recipe.get('cuisine','')}  ·  "
            f"{recipe.get('difficulty','').capitalize()}  ·  ⏱ {recipe.get('time_minutes','')} min"
        ):
            rc1, rc2 = st.columns([2, 1])
            with rc1:
                st.markdown("**Ingredients:**")
                for ing in recipe.get("ingredients", []):
                    st.markdown(f"- {ing['name']} — {ing['quantity']}")
            with rc2:
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
                st.markdown("**Tips:**")
                for tip in recipe["tips"]:
                    st.markdown(f"- {tip}")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Recipe Preparation Agent · IBM Granite + RAG · Built for AICTE")
