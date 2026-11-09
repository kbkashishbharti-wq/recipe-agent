"""
Recipe Preparation Agent — Streamlit UI
IBM Granite + RAG powered cooking assistant.
Deploy: https://streamlit.io/cloud

NEW:
  • 🎥 YouTube video support (per recipe + a dedicated Video Guides page)
  • 👩‍🍳 Community Kitchen (users upload photos + feedback of dishes they made)
"""

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

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

# ── Storage for new features ──────────────────────────────────────────────────
# NOTE: Streamlit Cloud's disk is temporary — files here are wiped when the app
# restarts/redeploys. Fine for demos; for permanent storage use Supabase /
# Firebase / Google Sheets + Cloudinary.
DATA_DIR    = Path("community_data")
IMAGE_DIR   = DATA_DIR / "images"
POSTS_FILE  = DATA_DIR / "posts.json"
VIDEOS_FILE = DATA_DIR / "videos.json"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_MB = 5
YT_PATTERN = re.compile(
    r"^(https?://)?(www\.|m\.)?"
    r"(youtube\.com/(watch\?(.*&)?v=|shorts/|embed/)|youtu\.be/)[\w-]{11}",
    re.IGNORECASE,
)


def load_json(path: Path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_youtube_url(url: str) -> bool:
    return bool(YT_PATTERN.match((url or "").strip()))


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
        [
            "🔍 Find Recipes",
            "🔄 Substitution",
            "💡 Cooking Tips",
            "📖 Browse Recipes",
            "🎥 Video Guides",
            "👩‍🍳 Community Kitchen",
        ],
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


# ── Helper: YouTube video block for a recipe ─────────────────────────────────
def video_section(recipe: dict, key_prefix: str):
    """Shows the recipe's video (if any), a YouTube search link, and a box to add a link."""
    rid  = str(recipe.get("id", recipe.get("name", "")))
    name = recipe.get("name", "")

    saved_links = load_json(VIDEOS_FILE, {})
    url = recipe.get("youtube_url") or saved_links.get(rid)

    st.markdown("**🎥 Video tutorial**")
    if url and is_youtube_url(url):
        st.video(url)
    else:
        st.caption("No video added for this recipe yet.")

    search_url = f"https://www.youtube.com/results?search_query={quote_plus(name + ' recipe')}"
    st.markdown(f"[🔎 Search “{name}” on YouTube]({search_url})")

    new_url = st.text_input(
        "Paste a YouTube link for this recipe",
        placeholder="https://www.youtube.com/watch?v=...",
        key=f"{key_prefix}_yt_{rid}",
    )
    if st.button("Save video link", key=f"{key_prefix}_ytbtn_{rid}"):
        if is_youtube_url(new_url):
            saved_links[rid] = new_url.strip()
            save_json(VIDEOS_FILE, saved_links)
            st.success("Video saved! It will show up next time this recipe is opened.")
        else:
            st.error("That doesn't look like a valid YouTube link.")


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
                            st.divider()
                            video_section(r, key_prefix=f"find{i}")
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
            st.divider()
            video_section(recipe, key_prefix="browse")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Video Guides (NEW)
# ════════════════════════════════════════════════════════════════════════════
elif page == "🎥 Video Guides":
    st.title("🎥 Video Guides")
    st.caption("Watch a cooking video for any recipe, or paste any YouTube link to play it here.")

    tab_recipe, tab_any = st.tabs(["Recipe videos", "Play any YouTube link"])

    with tab_recipe:
        all_recipes_list = agent.rag.get_all_recipes()
        by_id = {r["id"]: r for r in all_recipes_list}
        vid_id = st.selectbox(
            "Choose a recipe",
            list(by_id.keys()),
            format_func=lambda rid: by_id[rid]["name"],
            key="video_recipe_select",
        )
        video_section(by_id[vid_id], key_prefix="videopage")

    with tab_any:
        link = st.text_input(
            "YouTube link",
            placeholder="https://www.youtube.com/watch?v=...",
            key="any_yt_link",
        )
        if link:
            if is_youtube_url(link):
                st.video(link.strip())
            else:
                st.error("Please enter a valid YouTube link (youtube.com or youtu.be).")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Community Kitchen (NEW)
# ════════════════════════════════════════════════════════════════════════════
elif page == "👩‍🍳 Community Kitchen":
    st.title("👩‍🍳 Community Kitchen")
    st.caption("Made a dish using our app? Share a photo and your feedback with everyone!")

    all_recipes_list = agent.rag.get_all_recipes()
    recipe_names = sorted(r["name"] for r in all_recipes_list)

    # ── Submit form ──────────────────────────────────────────────────────────
    with st.form("community_form", clear_on_submit=True):
        st.subheader("📸 Share your creation")
        fc1, fc2 = st.columns(2)
        author = fc1.text_input("Your name", max_chars=40, placeholder="e.g. Priya")
        recipe_choice = fc2.selectbox("Which recipe did you make?", recipe_names)

        rating = st.slider("Your rating", 1, 5, 5)
        feedback = st.text_area(
            "Your feedback",
            max_chars=500,
            placeholder="How did it turn out? Any changes you made?",
        )
        photo = st.file_uploader(
            f"Photo of your dish (JPG/PNG, max {MAX_IMAGE_MB} MB)",
            type=["jpg", "jpeg", "png"],
        )
        submitted = st.form_submit_button("🚀 Post to Community", type="primary")

    if submitted:
        if not author.strip():
            st.warning("Please enter your name.")
        elif not feedback.strip() and photo is None:
            st.warning("Please add a photo or some feedback.")
        elif photo is not None and photo.size > MAX_IMAGE_MB * 1024 * 1024:
            st.error(f"Photo is too large. Please keep it under {MAX_IMAGE_MB} MB.")
        else:
            image_name = None
            if photo is not None:
                ext = Path(photo.name).suffix.lower()
                if ext not in (".jpg", ".jpeg", ".png"):
                    ext = ".jpg"
                image_name = f"{uuid.uuid4().hex}{ext}"
                with open(IMAGE_DIR / image_name, "wb") as f:
                    f.write(photo.getbuffer())

            posts = load_json(POSTS_FILE, [])
            posts.append({
                "id":        uuid.uuid4().hex,
                "author":    author.strip(),
                "recipe":    recipe_choice,
                "rating":    rating,
                "feedback":  feedback.strip(),
                "image":     image_name,
                "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            })
            save_json(POSTS_FILE, posts)
            st.success("🎉 Thanks for sharing! Your post is live below.")

    # ── Gallery ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🍽️ Community Gallery")

    posts = load_json(POSTS_FILE, [])
    posts = list(reversed(posts))  # newest first

    if posts:
        recipes_with_posts = sorted({p["recipe"] for p in posts})
        gallery_filter = st.multiselect("Filter by recipe", recipes_with_posts)
        if gallery_filter:
            posts = [p for p in posts if p["recipe"] in gallery_filter]

        st.markdown(f"**{len(posts)} post(s)**")

        cols = st.columns(3)
        for idx, p in enumerate(posts):
            with cols[idx % 3]:
                with st.container(border=True):
                    img_path = IMAGE_DIR / p["image"] if p.get("image") else None
                    if img_path and img_path.exists():
                        st.image(str(img_path), use_container_width=True)
                    st.markdown(f"**{p['recipe']}**")
                    st.markdown("⭐" * int(p.get("rating", 0)))
                    if p.get("feedback"):
                        st.write(p["feedback"])
                    st.caption(f"👤 {p['author']} · {p['timestamp']}")
    else:
        st.info("No posts yet — be the first to share your dish! 🍳")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Recipe Preparation Agent · IBM Granite + RAG · Built for AICTE")



