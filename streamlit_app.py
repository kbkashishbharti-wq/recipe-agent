"""
Recipe Preparation Agent — Streamlit UI
IBM Granite + RAG powered cooking assistant.
Deploy: https://streamlit.io/cloud

FEATURES
  • 🏠 Home with Recipe of the Day
  • 🔍 Find Recipes · 🔄 Substitution · 💡 Cooking Tips · 📖 Browse Recipes
  • 👨‍🍳 Cooking Mode (step-by-step, big text, built-in timer)
  • 💬 Chat with the Chef (Granite)
  • 📅 Meal Planner (Mon–Sun) → 🛒 Shopping List (tick off + download)
  • 🎯 Nutrition Goal tracker
  • ⭐ My Favorites
  • 🔢 Serving size scaler
  • 🎥 YouTube video guides
  • 👩‍🍳 Community Kitchen (photos, feedback, ❤️ likes, top-rated dishes)
"""

import html
import json
import os
import re
import uuid
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote_plus

import streamlit as st
import streamlit.components.v1 as components

# ── Page config — MUST be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Recipe Preparation Agent",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject Streamlit secrets into env BEFORE importing the agent ──────────────
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
    pass

from recipe_agent import RecipeAgent  # noqa: E402  (must come after env setup)

# ── Storage ───────────────────────────────────────────────────────────────────
# NOTE: Streamlit Cloud's disk is temporary — files are wiped on restart/redeploy.
# Favorites, meal plan and shopping list live in st.session_state (per visitor,
# reset on browser refresh). Community posts/videos are saved to disk (shared).
DATA_DIR    = Path("community_data")
IMAGE_DIR   = DATA_DIR / "images"
POSTS_FILE  = DATA_DIR / "posts.json"
VIDEOS_FILE = DATA_DIR / "videos.json"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_MB = 5
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
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


# ── Quantity helpers (serving scaler + shopping list) ─────────────────────────
_NUM = re.compile(r"^\s*(\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)\s*(.*)$")


def _to_float(s: str) -> float:
    s = s.strip()
    if " " in s:
        whole, frac = s.split()
        n, d = frac.split("/")
        return int(whole) + int(n) / int(d)
    if "/" in s:
        n, d = s.split("/")
        return int(n) / int(d)
    return float(s)


def parse_qty(q):
    """'1 1/2 cups' -> (1.5, 'cups'); 'to taste' -> (None, 'to taste')."""
    text = str(q or "").strip()
    m = _NUM.match(text)
    if not m:
        return None, text
    try:
        return _to_float(m.group(1)), m.group(2).strip()
    except (ValueError, ZeroDivisionError):
        return None, text


def fmt_num(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".") if abs(x - round(x)) > 0.01 else str(int(round(x)))


def scale_qty(q, factor: float) -> str:
    n, unit = parse_qty(q)
    if n is None:
        return str(q or "")
    return f"{fmt_num(n * factor)} {unit}".strip()


def base_servings(recipe: dict) -> int:
    m = re.search(r"\d+", str(recipe.get("servings", "")))
    return max(int(m.group()), 1) if m else 4


def servings_picker(recipe: dict, key: str) -> int:
    return int(
        st.number_input(
            "Servings",
            min_value=1,
            max_value=24,
            value=base_servings(recipe),
            step=1,
            key=key,
        )
    )


def build_shopping_list(recipes: list, servings: int) -> dict:
    """Combine ingredients of several recipes -> {name: 'quantity text'}."""
    agg = {}
    for r in recipes:
        factor = servings / base_servings(r)
        for ing in r.get("ingredients", []):
            name = str(ing.get("name", "")).strip().lower()
            if not name:
                continue
            n, unit = parse_qty(ing.get("quantity", ""))
            agg.setdefault(name, []).append((None if n is None else n * factor, unit))

    result = {}
    for name, items in agg.items():
        sums, raws = {}, []
        for n, unit in items:
            if n is None:
                if unit and unit not in raws:
                    raws.append(unit)
            else:
                sums[unit.lower()] = sums.get(unit.lower(), 0) + n
        parts = [f"{fmt_num(v)} {u}".strip() for u, v in sums.items()] + raws
        result[name] = " + ".join(parts)
    return result


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] > .main .block-container {
            padding-top: 1.2rem;
        }
        .recipe-card {
            background: #f7f8fa;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
            font-family: inherit;
        }
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
        .dot-green { color: #16a34a; font-weight: bold; }
        .dot-red   { color: #dc2626; font-weight: bold; }
        .cook-step {
            font-size: 1.7rem;
            line-height: 1.6;
            padding: 1.2rem 1.4rem;
            border-radius: 10px;
            border: 1px solid #e5e7eb;
            background: #f7f8fa;
            color: #111827;
            margin: 0.6rem 0 1rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached agent ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Initialising Recipe Agent…")
def get_agent() -> RecipeAgent:
    return RecipeAgent()


agent = get_agent()
status = agent.get_status()

granite_ok = status.get("granite_connected", False)
vector_ok  = status.get("vector_search", False)
recipes_n  = status.get("recipes_loaded", 0)

# Session-state defaults
st.session_state.setdefault("favorites", [])
st.session_state.setdefault("shop_recipe_ids", [])
st.session_state.setdefault("meal_plan", {})
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("liked_posts", [])


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🍳 Recipe Agent")
    st.caption("IBM Granite · RAG powered")
    st.divider()

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

    st.markdown("**Dietary Preferences**")
    dietary = st.multiselect(
        "diet",
        ["vegetarian", "vegan", "gluten-free", "dairy-free", "keto", "halal"],
        label_visibility="collapsed",
    )

    st.divider()

    page = st.radio(
        "nav",
        [
            "🏠 Home",
            "🔍 Find Recipes",
            "🔄 Substitution",
            "💡 Cooking Tips",
            "📖 Browse Recipes",
            "👨‍🍳 Cooking Mode",
            "💬 Chat with the Chef",
            "📅 Meal Planner",
            "🛒 Shopping List",
            "🎯 Nutrition Goal",
            f"⭐ My Favorites ({len(st.session_state['favorites'])})",
            "🎥 Video Guides",
            "👩‍🍳 Community Kitchen",
        ],
        label_visibility="collapsed",
    )
    if page.startswith("⭐ My Favorites"):
        page = "⭐ My Favorites"


# ── Helpers: recipe card ──────────────────────────────────────────────────────
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


# ── Helpers: favorites + shopping buttons ────────────────────────────────────
def recipe_actions(recipe: dict, key_prefix: str):
    """⭐ favorite toggle and 🛒 add-to-shopping-list buttons."""
    rid = recipe.get("id")
    if rid is None:
        return
    favs = st.session_state["favorites"]
    shop = st.session_state["shop_recipe_ids"]

    a1, a2 = st.columns(2)
    if a1.button(
        "★ Remove from favorites" if rid in favs else "☆ Save to favorites",
        key=f"{key_prefix}_fav_{rid}",
        use_container_width=True,
    ):
        if rid in favs:
            favs.remove(rid)
        else:
            favs.append(rid)
        st.rerun()

    if rid in shop:
        a2.button("🛒 In shopping list ✓", key=f"{key_prefix}_shop_{rid}",
                  disabled=True, use_container_width=True)
    elif a2.button("🛒 Add to shopping list", key=f"{key_prefix}_shop_{rid}",
                   use_container_width=True):
        shop.append(rid)
        st.rerun()


# ── Helpers: YouTube video block ─────────────────────────────────────────────
def video_section(recipe: dict, key_prefix: str):
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


# ── Helper: countdown timer (runs in the browser) ────────────────────────────
def timer_widget(minutes: float):
    secs = max(int(minutes * 60), 1)
    code = """
    <div style="font-family:sans-serif;text-align:center;background:#f7f8fa;color:#111827;
                border:1px solid #e5e7eb;border-radius:10px;padding:12px;">
      <div id="t" style="font-size:46px;font-weight:700;">00:00</div>
      <button onclick="start()" style="padding:6px 14px;margin:2px;">▶ Start</button>
      <button onclick="pause()" style="padding:6px 14px;margin:2px;">⏸ Pause</button>
      <button onclick="reset()" style="padding:6px 14px;margin:2px;">↺ Reset</button>
      <div id="m" style="margin-top:6px;color:#16a34a;font-weight:600;min-height:20px;"></div>
    </div>
    <script>
      const total = __SECS__;
      let left = total, id = null;
      function show(){
        const m = String(Math.floor(left/60)).padStart(2,'0');
        const s = String(left%60).padStart(2,'0');
        document.getElementById('t').textContent = m + ':' + s;
      }
      function done(){
        document.getElementById('m').textContent = "⏰ Time's up!";
        try {
          const c = new (window.AudioContext || window.webkitAudioContext)();
          const o = c.createOscillator(); o.connect(c.destination); o.start();
          setTimeout(() => o.stop(), 800);
        } catch(e) {}
      }
      function start(){
        if (id || left <= 0) return;
        id = setInterval(() => {
          left--; show();
          if (left <= 0){ clearInterval(id); id = null; done(); }
        }, 1000);
      }
      function pause(){ clearInterval(id); id = null; }
      function reset(){
        pause(); left = total;
        document.getElementById('m').textContent = '';
        show();
      }
      show();
    </script>
    """.replace("__SECS__", str(secs))
    components.html(code, height=170)


# ── Helper: ask the chef (uses the existing agent) ───────────────────────────
def ask_chef(question: str) -> str:
    try:
        known = agent.rag.get_unique_ingredients()
        q = question.lower()
        found = [i for i in known if str(i).lower() in q]
        result = agent.find_recipes(
            ingredients=found or ["salt", "oil"],
            dietary_preferences=dietary if dietary else None,
            user_query=question,
        )
        if result.get("success"):
            return result.get("ai_response") or "Sorry, I couldn't come up with an answer."
        return result.get("error", "Sorry, something went wrong.")
    except Exception as exc:  # noqa: BLE001
        return f"Sorry, the chef is unavailable right now ({exc})."


def calories_of(recipe: dict):
    n = recipe.get("nutrition") or {}
    m = re.search(r"\d+", str(n.get("calories", "")))
    return int(m.group()) if m else None


# ════════════════════════════════════════════════════════════════════════════
# HOME — Recipe of the Day
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🍳 Recipe Preparation Agent")
    st.caption("Find recipes from what you have, cook step by step, and share with the community.")

    all_recipes_list = agent.rag.get_all_recipes()
    if all_recipes_list:
        r = all_recipes_list[date.today().toordinal() % len(all_recipes_list)]
        st.subheader("🌟 Recipe of the Day")
        with st.container(border=True):
            st.markdown(f"### {r['name']}")
            st.caption(
                f"{r.get('cuisine','')} · {r.get('difficulty','').capitalize()} · "
                f"⏱ {r.get('time_minutes','')} min · 🥗 {', '.join(r.get('diet', []))}"
            )
            hc1, hc2 = st.columns([1, 2])
            with hc1:
                st.markdown("**Ingredients**")
                for ing in r.get("ingredients", []):
                    st.markdown(f"- {ing['name']} — {ing['quantity']}")
            with hc2:
                st.markdown("**Instructions**")
                for j, step in enumerate(r.get("instructions", []), 1):
                    st.markdown(f"{j}. {step}")
            recipe_actions(r, "home")

    posts = load_json(POSTS_FILE, [])
    m1, m2, m3 = st.columns(3)
    m1.metric("Recipes", recipes_n)
    m2.metric("Community posts", len(posts))
    m3.metric("Your favorites", len(st.session_state["favorites"]))

    st.info("Use the menu on the left: try **Cooking Mode**, **Meal Planner** or **Chat with the Chef**.", icon="👈")


# ════════════════════════════════════════════════════════════════════════════
# PAGE — Find Recipes
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Find Recipes":
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
                st.session_state["find_result"] = agent.find_recipes(
                    ingredients=ingredients,
                    dietary_preferences=dietary if dietary else None,
                    user_query=user_query,
                )

    # Results are kept in session_state so buttons inside them (favorite,
    # shopping list, save video) don't make the results disappear.
    result = st.session_state.get("find_result")
    if result:
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
                        recipe_actions(r, f"find{i}")
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
# PAGE — Ingredient Substitution
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
# PAGE — Cooking Tips
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
# PAGE — Browse All Recipes (with serving scaler)
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
                servings = servings_picker(recipe, key=f"browse_serv_{recipe['id']}")
                factor = servings / base_servings(recipe)
                st.markdown(f"**Ingredients (for {servings}):**")
                for ing in recipe.get("ingredients", []):
                    st.markdown(f"- {ing['name']} — {scale_qty(ing.get('quantity',''), factor)}")
            with rc2:
                st.markdown("**Diet:** " + ", ".join(recipe.get("diet", [])))
                st.markdown(f"**Original servings:** {recipe.get('servings', '—')}")
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
            recipe_actions(recipe, "browse")
            st.divider()
            video_section(recipe, key_prefix="browse")


# ════════════════════════════════════════════════════════════════════════════
# PAGE — Cooking Mode (step by step + timer)
# ════════════════════════════════════════════════════════════════════════════
elif page == "👨‍🍳 Cooking Mode":
    st.title("👨‍🍳 Cooking Mode")
    st.caption("One step at a time, in big text, with a timer.")

    all_recipes_list = agent.rag.get_all_recipes()
    by_id = {r["id"]: r for r in all_recipes_list}

    rid = st.selectbox(
        "Choose a recipe to cook",
        list(by_id.keys()),
        format_func=lambda x: by_id[x]["name"],
        key="cook_recipe_select",
    )
    recipe = by_id[rid]
    steps = recipe.get("instructions", [])

    if st.session_state.get("cook_recipe") != rid:
        st.session_state["cook_recipe"] = rid
        st.session_state["cook_step"] = 0

    with st.expander("🧺 Ingredients (adjust servings)"):
        servings = servings_picker(recipe, key=f"cook_serv_{rid}")
        factor = servings / base_servings(recipe)
        for ing in recipe.get("ingredients", []):
            st.markdown(f"- {ing['name']} — {scale_qty(ing.get('quantity',''), factor)}")

    if not steps:
        st.info("This recipe has no instructions yet.")
    else:
        step = min(st.session_state.get("cook_step", 0), len(steps) - 1)
        st.session_state["cook_step"] = step

        st.progress((step + 1) / len(steps), text=f"Step {step + 1} of {len(steps)}")
        st.markdown(
            f'<div class="cook-step">{html.escape(str(steps[step]))}</div>',
            unsafe_allow_html=True,
        )

        b1, b2, b3 = st.columns(3)
        if b1.button("⬅️ Back", disabled=step == 0, use_container_width=True):
            st.session_state["cook_step"] = step - 1
            st.rerun()
        if b2.button("🔁 Restart", use_container_width=True):
            st.session_state["cook_step"] = 0
            st.rerun()
        if b3.button("Next ➡️", type="primary", disabled=step == len(steps) - 1,
                     use_container_width=True):
            st.session_state["cook_step"] = step + 1
            st.rerun()

        if step == len(steps) - 1:
            st.success("🎉 All done! Enjoy your meal — share a photo in **Community Kitchen**.")

        st.markdown("**⏱ Timer for this step**")
        guess = re.search(r"(\d+)\s*(?:to|-|–)?\s*(?:\d+\s*)?min", str(steps[step]), re.I)
        default_min = int(guess.group(1)) if guess else 5
        minutes = st.number_input(
            "Minutes",
            min_value=1,
            max_value=240,
            value=min(default_min, 240),
            key=f"timer_min_{rid}_{step}",
        )
        timer_widget(minutes)


# ════════════════════════════════════════════════════════════════════════════
# PAGE — Chat with the Chef
# ════════════════════════════════════════════════════════════════════════════
elif page == "💬 Chat with the Chef":
    st.title("💬 Chat with the Chef")
    st.caption("Ask any cooking question — “Why is my dough sticky?”, “How do I fix salty curry?”")

    if st.button("🧹 Clear chat"):
        st.session_state["chat_history"] = []
        st.rerun()

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask the chef…")
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Chef is thinking…"):
                answer = ask_chef(question)
            st.markdown(answer)
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})


# ════════════════════════════════════════════════════════════════════════════
# PAGE — Meal Planner
# ════════════════════════════════════════════════════════════════════════════
elif page == "📅 Meal Planner":
    st.title("📅 Weekly Meal Planner")
    st.caption("Pick one recipe per day, then send everything to your shopping list.")

    all_recipes_list = agent.rag.get_all_recipes()
    names = [r["name"] for r in all_recipes_list]
    name_to_recipe = {r["name"]: r for r in all_recipes_list}
    options = ["—"] + names

    plan = st.session_state["meal_plan"]
    cols = st.columns(2)
    for idx, day in enumerate(DAYS):
        current = plan.get(day, "—")
        choice = cols[idx % 2].selectbox(
            day,
            options,
            index=options.index(current) if current in options else 0,
            key=f"plan_select_{day}",
        )
        plan[day] = choice

    planned = [name_to_recipe[plan[d]] for d in DAYS if plan.get(d) in name_to_recipe]
    st.divider()
    if planned:
        total_cal = [calories_of(r) for r in planned if calories_of(r) is not None]
        st.markdown(f"**{len(planned)} meal(s) planned**"
                    + (f" · average ≈ {sum(total_cal)//len(total_cal)} kcal per meal" if total_cal else ""))
        if st.button("🛒 Send plan to Shopping List", type="primary"):
            ids = []
            for r in planned:
                if r.get("id") is not None and r["id"] not in ids:
                    ids.append(r["id"])
            st.session_state["shop_recipe_ids"] = ids
            st.success("Done! Open **🛒 Shopping List** from the menu.")
    else:
        st.info("Choose at least one recipe above.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE — Shopping List
# ════════════════════════════════════════════════════════════════════════════
elif page == "🛒 Shopping List":
    st.title("🛒 Shopping List")
    st.caption("Combined ingredients of your chosen recipes. Tick items you already have.")

    all_recipes_list = agent.rag.get_all_recipes()
    by_id = {r["id"]: r for r in all_recipes_list}

    default_ids = [i for i in st.session_state["shop_recipe_ids"] if i in by_id]
    chosen_ids = st.multiselect(
        "Recipes",
        list(by_id.keys()),
        default=default_ids,
        format_func=lambda rid: by_id[rid]["name"],
    )
    st.session_state["shop_recipe_ids"] = chosen_ids

    people = st.number_input("Cooking for how many people?", 1, 24, 4)

    if chosen_ids:
        shopping = build_shopping_list([by_id[i] for i in chosen_ids], int(people))
        st.markdown(f"**{len(shopping)} items**")

        needed = []
        for name in sorted(shopping):
            qty = shopping[name]
            have = st.checkbox(
                f"{name.title()} — {qty}" if qty else name.title(),
                key=f"shop_have_{name}",
            )
            if not have:
                needed.append(f"[ ] {name.title()} — {qty}" if qty else f"[ ] {name.title()}")

        st.divider()
        st.markdown(f"**Still to buy: {len(needed)}**")
        text = "Shopping List\n" + "=" * 13 + "\n" + "\n".join(needed)
        st.download_button(
            "⬇️ Download list (.txt)",
            data=text,
            file_name="shopping_list.txt",
            mime="text/plain",
            disabled=not needed,
        )
    else:
        st.info("Add recipes here, from the Meal Planner, or with the 🛒 button on any recipe.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE — Nutrition Goal
# ════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Nutrition Goal":
    st.title("🎯 Nutrition Goal")
    st.caption("Set a daily calorie goal and see which recipes fit one meal.")

    g1, g2 = st.columns(2)
    goal  = g1.number_input("Daily calorie goal (kcal)", 800, 5000, 2000, step=50)
    meals = g2.slider("Meals per day", 2, 6, 3)
    per_meal = goal / meals
    st.info(f"Target per meal: about **{int(per_meal)} kcal** (±10% counts as a fit).")

    rows = []
    for r in agent.rag.get_all_recipes():
        cal = calories_of(r)
        if cal is None:
            verdict = "No data"
        elif cal <= per_meal * 1.1:
            verdict = "✅ Fits"
        else:
            verdict = "⚠️ Over"
        rows.append({
            "Recipe": r["name"],
            "Calories": cal if cal is not None else "—",
            "Verdict": verdict,
        })

    only_fit = st.checkbox("Show only recipes that fit", value=False)
    if only_fit:
        rows = [row for row in rows if row["Verdict"] == "✅ Fits"]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("Calories are per serving as stored in the recipe data.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE — My Favorites
# ════════════════════════════════════════════════════════════════════════════
elif page == "⭐ My Favorites":
    st.title("⭐ My Favorites")
    st.caption("Saved for this browser session. Refreshing the page clears them.")

    all_recipes_list = agent.rag.get_all_recipes()
    by_id = {r["id"]: r for r in all_recipes_list}
    fav_recipes = [by_id[i] for i in st.session_state["favorites"] if i in by_id]

    if not fav_recipes:
        st.info("No favorites yet. Tap **☆ Save to favorites** on any recipe.")
    for recipe in fav_recipes:
        with st.expander(f"**{recipe['name']}** · {recipe.get('cuisine','')} · ⏱ {recipe.get('time_minutes','')} min"):
            st.markdown("**Ingredients:**")
            for ing in recipe.get("ingredients", []):
                st.markdown(f"- {ing['name']} — {ing['quantity']}")
            st.markdown("**Instructions:**")
            for j, step in enumerate(recipe.get("instructions", []), 1):
                st.markdown(f"{j}. {step}")
            recipe_actions(recipe, "favpage")


# ════════════════════════════════════════════════════════════════════════════
# PAGE — Video Guides
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
# PAGE — Community Kitchen
# ════════════════════════════════════════════════════════════════════════════
elif page == "👩‍🍳 Community Kitchen":
    st.title("👩‍🍳 Community Kitchen")
    st.caption("Made a dish using our app? Share a photo and your feedback with everyone!")

    all_recipes_list = agent.rag.get_all_recipes()
    recipe_names = sorted(r["name"] for r in all_recipes_list)

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
                "likes":     0,
                "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            })
            save_json(POSTS_FILE, posts)
            st.success("🎉 Thanks for sharing! Your post is live below.")

    all_posts = load_json(POSTS_FILE, [])

    # ── Top rated dishes ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("🏆 Top Rated Dishes")
    if all_posts:
        stats = {}
        for p in all_posts:
            s = stats.setdefault(p["recipe"], {"ratings": [], "likes": 0})
            s["ratings"].append(int(p.get("rating", 0)))
            s["likes"] += int(p.get("likes", 0))
        top_rows = [
            {
                "Recipe": name,
                "Avg rating": round(sum(s["ratings"]) / len(s["ratings"]), 1),
                "Posts": len(s["ratings"]),
                "❤️ Likes": s["likes"],
            }
            for name, s in stats.items()
        ]
        top_rows.sort(key=lambda row: (row["Avg rating"], row["❤️ Likes"]), reverse=True)
        st.dataframe(top_rows[:5], use_container_width=True, hide_index=True)
    else:
        st.caption("Rankings appear once people start posting.")

    # ── Gallery ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🍽️ Community Gallery")

    posts = list(reversed(all_posts))  # newest first

    if posts:
        sort_mode = st.radio("Sort by", ["Newest", "Most liked"], horizontal=True)
        if sort_mode == "Most liked":
            posts.sort(key=lambda p: int(p.get("likes", 0)), reverse=True)

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

                    already = p["id"] in st.session_state["liked_posts"]
                    likes = int(p.get("likes", 0))
                    if st.button(
                        f"❤️ {likes}" + (" · Liked" if already else ""),
                        key=f"like_{p['id']}",
                        disabled=already,
                    ):
                        fresh = load_json(POSTS_FILE, [])
                        for q in fresh:
                            if q["id"] == p["id"]:
                                q["likes"] = int(q.get("likes", 0)) + 1
                        save_json(POSTS_FILE, fresh)
                        st.session_state["liked_posts"].append(p["id"])
                        st.rerun()
    else:
        st.info("No posts yet — be the first to share your dish! 🍳")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Recipe Preparation Agent · IBM Granite + RAG · Built for AICTE")



