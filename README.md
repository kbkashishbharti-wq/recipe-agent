# Recipe Preparation Agent
### IBM Granite + RAG-powered Cooking Assistant | AICTE Problem Statement #16

---

## Overview

The **Recipe Preparation Agent** helps users cook meals using only the ingredients they have on hand. It uses a **RAG (Retrieval-Augmented Generation)** architecture powered by **IBM Granite** (via IBM watsonx.ai) to retrieve relevant recipes and generate personalised, step-by-step cooking instructions.

**Key features:**
- Input ingredients and get tailored recipe suggestions
- IBM Granite LLM generates contextualised cooking guidance
- TF-IDF vector RAG search (pure Python, no native dependencies)
- Smart ingredient substitution advisor
- AI-powered cooking tips
- Dietary adjustments (vegan, gluten-free, keto, etc.)
- Reduces food waste by using what you already have

---

## ▶️ Deploy on Streamlit Cloud (One-Click)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)

### Steps

1. **Fork / clone this repo** to your GitHub account.

2. **Go to** [share.streamlit.io](https://share.streamlit.io) → **New app**.

3. Fill in:
   | Field | Value |
   |---|---|
   | Repository | `your-github-username/your-repo-name` |
   | Branch | `main` |
   | Main file path | `streamlit_app.py` |

4. **Add Secrets** — click **Advanced settings → Secrets** and paste:
   ```toml
   WATSONX_API_KEY      = "your_ibm_cloud_api_key"
   WATSONX_PROJECT_ID   = "your_watsonx_project_id"
   WATSONX_URL          = "https://us-south.ml.cloud.ibm.com"
   GRANITE_MODEL_ID     = "ibm/granite-4-h-small"
   ```
   > **Without these**, the app still works in **demo mode** (RAG retrieval fully functional, LLM responses are placeholders).

5. Click **Deploy!** — your app will be live in ~60 seconds.

---

## Architecture

```
User Input (ingredients)
        |
        v
+----------------------+
|   Streamlit UI       |  streamlit_app.py
|  Find Recipes        |
|  Substitution        |
|  Cooking Tips        |
|  Browse Recipes      |
+--------+-------------+
         |
    +----+------------------------------+
    |         RecipeAgent              |  recipe_agent.py
    |  Orchestrates RAG + LLM          |
    +----+-------------+---------------+
         |             |
         v             v
+-------------+  +----------------------+
|  RAGEngine  |  |   GraniteClient      |
| rag_engine  |  |  granite_client.py   |
|             |  |                      |
|  TF-IDF     |  |  IBM watsonx.ai      |
|  vector idx |  |  granite-4-h-small   |
| recipes.json|  |  (Lite plan free)    |
+-------------+  +----------------------+
         |
         v
  data/recipes.json  (12 recipes knowledge base)
```

---

## Project Structure

```
AICTE-RECIPE PREPARATION AGENT/
+-- .env.example               <- Template: copy to ".env" and fill in credentials
+-- .env                       <- Your real credentials (git-ignored, never committed)
+-- .gitignore                 <- Excludes .env, __pycache__, secrets, etc.
+-- requirements.txt           <- Python dependencies (streamlit + ibm-watsonx-ai)
+-- streamlit_app.py           <- ★ Streamlit UI — deploy THIS file
+-- app.py                     <- Flask REST API (local dev / non-Streamlit deployments)
+-- recipe_agent.py            <- Core agent logic (RAG + LLM orchestration)
+-- rag_engine.py              <- TF-IDF RAG engine (pure Python, no native deps)
+-- granite_client.py          <- IBM Granite LLM client (watsonx.ai chat API)
+-- .streamlit/
|   +-- secrets.toml           <- Local secrets template (git-ignored)
+-- data/
|   +-- recipes.json           <- Recipe knowledge base (12 recipes)
+-- README.md
```

---

## Local Development

### 1. Prerequisites
- Python 3.9+
- IBM Cloud account (free Lite plan works)
- IBM watsonx.ai project

### 2. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure IBM Cloud credentials

**Option A — `.env` file (for Flask / local):**
```bash
cp .env.example .env
# Edit .env and fill in WATSONX_API_KEY and WATSONX_PROJECT_ID
```

**Option B — `.streamlit/secrets.toml` (for local Streamlit):**
```bash
# Edit .streamlit/secrets.toml and fill in the same values
```

### 5. Run locally

**Streamlit (recommended):**
```bash
streamlit run streamlit_app.py
```
Open: **http://localhost:8501**

**Flask (REST API only):**
```bash
python app.py
```
Open: **http://localhost:5000**

---

## Usage

1. Type ingredients you have in your pantry (pick from list or free-type)
2. Optionally select dietary preferences (vegan, keto, gluten-free, etc.)
3. Optionally type a free-text query (e.g. "quick breakfast", "something spicy")
4. Click **"Find Recipes"**
5. View RAG-retrieved recipes with ingredient match scores
6. Read the IBM Granite AI response with step-by-step instructions
7. Use **Ingredient Substitution** for AI advice on missing items
8. Use **Cooking Tips** for expert Granite-generated tips
9. Use **Browse All Recipes** to filter the full knowledge base

---

## Recipe Knowledge Base

| ID   | Recipe                          | Cuisine       | Diet          | Time   |
|------|---------------------------------|---------------|---------------|--------|
| R001 | Classic Spaghetti Aglio e Olio  | Italian       | vegan         | 20 min |
| R002 | Chickpea and Spinach Curry      | Indian        | vegan, GF     | 30 min |
| R003 | Classic Omelette                | French        | keto          | 10 min |
| R004 | Vegetable Fried Rice            | Chinese       | vegan         | 20 min |
| R005 | Banana Pancakes                 | American      | vegetarian    | 15 min |
| R006 | Lemon Garlic Roasted Chicken    | Mediterranean | keto, GF      | 75 min |
| R007 | Tomato Basil Soup               | Italian       | vegan, GF     | 35 min |
| R008 | Greek Salad                     | Greek         | keto, GF      | 10 min |
| R009 | Dal (Red Lentil Soup)           | Indian        | vegan, GF     | 30 min |
| R010 | Avocado Toast with Poached Egg  | Modern        | vegetarian    | 15 min |
| R011 | Beef Stir-Fry with Vegetables   | Chinese       | GF            | 25 min |
| R012 | Simple French Toast             | French        | vegetarian    | 15 min |

---

## Technology Stack

| Component      | Technology                                                   |
|----------------|--------------------------------------------------------------|
| LLM            | IBM Granite 4 Small (`ibm/granite-4-h-small`) via watsonx.ai |
| RAG Retrieval  | Pure-Python TF-IDF vector index (zero native dependencies)   |
| UI             | Streamlit (`streamlit_app.py`)                               |
| REST API       | Python 3.9+ / Flask (`app.py`)                               |
| IBM SDK        | `ibm-watsonx-ai`                                             |

---

## Demo Mode

If IBM credentials are not configured, the application runs in **demo mode**:
- TF-IDF RAG retrieval works fully
- All recipe data, ingredient matching, and substitution lists work
- LLM responses show a placeholder explaining how to connect Granite

---

## AICTE Problem Statement #16

This project addresses the challenge of **reducing food waste** and making cooking **smarter and more sustainable** by:
- Turning available pantry items into practical meal solutions
- Providing intelligent substitutions to avoid unnecessary grocery trips
- Offering dietary adjustments for different health requirements
- Delivering step-by-step guidance adapted to available ingredients
