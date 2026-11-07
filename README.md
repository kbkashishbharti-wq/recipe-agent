# Recipe Preparation Agent 🍳
### IBM Granite + RAG-powered Cooking Assistant | AICTE Problem Statement #16

---

## 📋 Overview

The **Recipe Preparation Agent** helps users cook meals using only the ingredients they have on hand. It uses a **RAG (Retrieval-Augmented Generation)** architecture powered by **IBM Granite** (via IBM watsonx.ai) to retrieve relevant recipes and generate personalised, step-by-step cooking instructions.

**Key features:**
- 🥦 Input ingredients → get tailored recipe suggestions
- 🤖 IBM Granite LLM generates contextualised cooking guidance
- 🔍 TF-IDF vector RAG search (pure Python — no native dependencies)
- 🔄 Smart ingredient substitution advisor
- 🍴 AI-powered cooking tips
- 🌱 Dietary adjustments (vegan, gluten-free, keto, etc.)
- 💡 Reduces food waste by using what you already have

---

## 🏗️ Architecture

```
User Input (ingredients)
        │
        ▼
┌─────────────────────┐
│   Flask REST API    │  app.py
│  /api/recipes       │
│  /api/substitution  │
│  /api/tips          │
└────────┬────────────┘
         │
    ┌────┴─────────────────────────┐
    │         RecipeAgent          │  recipe_agent.py
    │  Orchestrates RAG + LLM      │
    └────┬─────────────┬───────────┘
         │             │
         ▼             ▼
┌─────────────┐  ┌──────────────────────┐
│  RAGEngine  │  │   GraniteClient      │
│ rag_engine  │  │  granite_client.py   │
│             │  │                      │
│  TF-IDF     │  │  IBM watsonx.ai      │
│  vector idx │  │  granite-4-h-small   │
│ recipes.json│  │  (Lite plan free)    │
└─────────────┘  └──────────────────────┘
         │
         ▼
  data/recipes.json  (12 recipes knowledge base)
```

---

## 📁 Project Structure

```
AICTE-RECIPE PREPARATION AGENT/
├── .env.example            # ← Template: copy to ".env" and fill in credentials
├── .env                    # ← Your real credentials (git-ignored, never committed)
├── .gitignore              # Excludes .env, __pycache__, etc.
├── requirements.txt        # Python dependencies
├── app.py                  # Flask REST API server
├── recipe_agent.py         # Core agent logic (RAG + LLM orchestration)
├── rag_engine.py           # TF-IDF RAG engine (pure Python, no native deps)
├── granite_client.py       # IBM Granite LLM client (watsonx.ai chat API)
├── static/
│   └── index.html          # Frontend web UI (single-page app)
├── data/
│   └── recipes.json        # Recipe knowledge base (12 recipes)
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.9+
- IBM Cloud account (free Lite plan works)
- IBM watsonx.ai project

### 2. Clone the repository

```bash
git clone https://github.com/kbkashishbharti-wq/recipe-agent.git
cd recipe-agent
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure IBM Cloud credentials

Copy the example environment file and fill in your real credentials:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
WATSONX_API_KEY=your_actual_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_actual_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
GRANITE_MODEL_ID=ibm/granite-4-h-small
```

> ⚠️ **Never commit `.env` to version control.** It is listed in `.gitignore`.

**How to get credentials:**
1. Sign up at [cloud.ibm.com](https://cloud.ibm.com) (free)
2. Navigate to **watsonx.ai** → create a project
3. Go to **Manage → Access (IAM)** → create an API key
4. Copy your **Project ID** from the project settings page

### 5. Run the application

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 🚀 Usage

1. **Type ingredients** you have in your pantry into the tag input (or use quick-add chips)
2. Optionally select **dietary preferences** (vegan, keto, gluten-free, etc.)
3. Optionally type a **free-text query** (e.g. "quick breakfast", "something spicy")
4. Click **"Find Recipes with IBM Granite"**
5. View **RAG-retrieved recipes** with ingredient match scores
6. Read the **IBM Granite AI response** with step-by-step instructions
7. Click **"Substitutions"** for any recipe to get AI advice on missing ingredients
8. Click **"AI Cooking Tips"** for expert Granite-generated tips

---

## 🔌 API Reference

### `POST /api/recipes`
Find recipes based on ingredients.
```json
{
  "ingredients": ["eggs", "flour", "milk", "butter"],
  "dietary_preferences": ["vegetarian"],
  "query": "quick breakfast"
}
```

### `POST /api/substitution`
Get substitution advice for a missing ingredient.
```json
{
  "recipe_id": "R001",
  "missing_ingredient": "parsley",
  "available_ingredients": ["basil", "chives"]
}
```

### `POST /api/tips`
Get AI cooking tips for a recipe.
```json
{ "recipe_id": "R003" }
```

### `GET /api/status`
Health check + agent configuration.

### `GET /api/all-recipes`
Browse all recipes in the knowledge base.

### `GET /api/ingredients`
All unique ingredient names (for autocomplete).

---

## 🍽️ Recipe Knowledge Base

| ID | Recipe | Cuisine | Diet | Time |
|----|--------|---------|------|------|
| R001 | Classic Spaghetti Aglio e Olio | Italian | vegan | 20 min |
| R002 | Chickpea and Spinach Curry | Indian | vegan, GF | 30 min |
| R003 | Classic Omelette | French | keto | 10 min |
| R004 | Vegetable Fried Rice | Chinese | vegan | 20 min |
| R005 | Banana Pancakes | American | vegetarian | 15 min |
| R006 | Lemon Garlic Roasted Chicken | Mediterranean | keto, GF | 75 min |
| R007 | Tomato Basil Soup | Italian | vegan, GF | 35 min |
| R008 | Greek Salad | Greek | keto, GF | 10 min |
| R009 | Dal (Red Lentil Soup) | Indian | vegan, GF | 30 min |
| R010 | Avocado Toast with Poached Egg | Modern | vegetarian | 15 min |
| R011 | Beef Stir-Fry with Vegetables | Chinese | GF | 25 min |
| R012 | Simple French Toast | French | vegetarian | 15 min |

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| LLM | IBM Granite 4 Small (`ibm/granite-4-h-small`) via watsonx.ai |
| RAG Retrieval | Pure-Python TF-IDF vector index (zero native dependencies) |
| Backend | Python 3.9+ / Flask |
| Frontend | Vanilla HTML / CSS / JavaScript (single-page app) |
| IBM SDK | `ibm-watsonx-ai` |

---

## 🔄 Demo Mode

If IBM credentials are not configured, the application runs in **demo mode**:
- ✅ TF-IDF RAG retrieval works fully
- ✅ All recipe data, ingredient matching, and substitution lists work
- ⚠️ LLM responses show a placeholder explaining how to connect Granite

---

## 🌱 AICTE Problem Statement #16

This project addresses the challenge of **reducing food waste** and making cooking **smarter and more sustainable** by:
- Turning available pantry items into practical meal solutions
- Providing intelligent substitutions to avoid unnecessary grocery trips
- Offering dietary adjustments for different health requirements
- Delivering step-by-step guidance adapted to available ingredients
