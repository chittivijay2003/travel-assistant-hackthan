# ==================================================================================
# IMPORTS
# ==================================================================================

# Standard library imports
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Callable, Tuple, Optional

# Third-party imports
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langchain_openai import ChatOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    ChatOpenAI = None
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# ==================================================================================
# ASSIGNMENT DOCUMENTATION
# ==================================================================================

# Assignment: **Multi-Model Orchestration** with Gemini & Claude

### Objective
"""In this assignment, you will design and implement **multi-model orchestration patterns** on top of multiple LLMs.

You will:
- Route requests to the **best model** (router pattern).
- Try a **cheap / small model first**, then fall back to a **stronger model** when needed (cascade pattern).
- Combine outputs from **multiple models** and add a **Claude fallback** for creative tasks (ensemble + cross-provider fallback).

We will use the following concrete models in this assignment:

- **`Gemini-2.5-flash`** → fast, cheap, good for simple queries.
- **`Gemini-2.5-pro`** → stronger reasoning, for complex queries.
- **`Gemini-2.0-flash`** → smaller / cheaper model, used in the cascade.
- **Claude** (any recent model) → used as a creative fallback model.
"""

## 📋 Instructions

"""
You are required to:

1. **Set up LLM clients (Gemini + Claude)**  
   Use **LangChain** (or an equivalent client) to configure:

   - `gemini_25_flash`  → wraps **Gemini-2.5-flash** (simple / fast queries)  
   - `gemini_25_pro`    → wraps **Gemini-2.5-pro** (complex, reasoning-heavy queries)  
   - `gemini_20_flash`  → wraps **Gemini-2.0-flash** (cheap first-pass model in the cascade)  
   - `claude_creative`  → wraps **Claude** (creative, long-form content & fallback)  

   Store API keys securely in environment variables, e.g.:

   - `GOOGLE_API_KEY` for Gemini
   - `ANTHROPIC_API_KEY` for Claude
"""
"""
2. **Implement a Router pattern**  
   Implement a function:

   ```python
   def route_query(user_query: str) -> dict:
       ...
   ```

   The router should decide **which single model** to call based on simple heuristics:

   - Simple / short / factual queries → **Gemini-2.5-flash**
   - Complex / reasoning-heavy queries → **Gemini-2.5-pro**
   - Technical / coding queries → you can choose **Gemini-2.5-pro** or **Gemini-2.0-flash**, but be consistent.

   The returned dict must include at least:

   ```python
   {
       "chosen_model": "gemini_25_flash",  # or "gemini_25_pro", "gemini_20_flash"
       "reason": "...",                    # short explanation of why this model was chosen
       "response": "..."                   # model answer as a string
   }
   ```

3. **Implement a Cascade pattern (cheap → strong)**  
   Implement a function:

   ```python
   def answer_with_cascade(user_query: str) -> dict:
       ...
   ```

   The cascade should:

   - Call **Gemini-2.0-flash** first (cheap, fast).
   - Inspect its answer using a simple heuristic (e.g., length, presence of phrases like *"I’m not sure"*, *"cannot help"*, or too generic / unhelpful output).
   - If the first answer seems **low quality / low confidence**, **fallback** to **Gemini-2.5-pro** and use its answer instead.
   - For clearly *technical / coding* questions, you may directly call **Gemini-2.5-pro**, or still pass through the cascade — design is up to you, but explain it.

   Return a dict such as:

   ```python
   {
       "used_fallback": True,              # or False
       "first_model": "gemini_20_flash",
       "second_model": "gemini_25_pro",
       "first_answer": "...",
       "final_answer": "...",
       "reason_for_fallback": "too short / contained 'not sure'"  # or None
   }
   ```
"""
"""
4. **Implement an Ensemble + Claude creative fallback**  
   Implement a function:

   ```python
   def ensemble_answer(user_query: str) -> dict:
       ...
   ```

   Behaviour:

   - For **non-creative** queries (facts, explanations, coding):
     - Call at least **two Gemini models** (for example, `gemini_25_flash` and `gemini_25_pro`).
   - For **creative** queries (stories, poems, marketing copy, etc.):
     - Call **Claude (`claude_creative`)** + at least one Gemini model (e.g., `gemini_25_pro`).
   - Combine their answers using a simple strategy, such as:
     - Concatenate and then ask one model (e.g., `gemini_25_pro`) to *“summarize & merge”*.
     - Or implement a basic voting / ranking rule (pick the one that best matches a heuristic).

   Return a dict such as:

   ```python
   {
       "models_called": ["gemini_25_pro", "claude_creative"],
       "raw_answers": {
           "gemini_25_pro": "...",
           "claude_creative": "..."
       },
       "final_answer": "...",
       "strategy": "summary"   # or "voting", "claude_fallback", etc.
   }
   ```
"""
"""
5. **(Recommended) Add basic logging / metrics**  
   For each of the functions above, log (even with simple `print` statements):

   - Which model(s) were used.
   - Approximate latency per call (`time.time()` before/after).
   - Whether a fallback was triggered (cascade) or Claude was used as creative fallback.

You **must** expose **at least one chat endpoint** as a **FastAPI** route (for example, `POST /chat`), through which I can:
- send a user message (and any metadata you need),
- receive the routed / cascaded / ensemble response,
- and see the key metadata you log (e.g., which model(s) were used, which pattern got triggered, and whether any fallback happened).  

The core orchestration logic should still live in clean, testable Python functions that your FastAPI endpoint simply calls.



## 🧪 Sample Inputs

Use the following example queries to test your orchestration logic.  
You can also add your own.

1. **Simple factual query** (should likely route to `gemini_25_flash`):

```text
What is the capital of Japan?
```

2. **Complex reasoning question** (should likely route to `gemini_25_pro`):

```text
Explain, step by step, how attention works in transformer models and why it scales quadratically.
```

3. **Technical / coding query** (good candidate for cascade and/or direct Pro):

```text
Write a Python function that implements binary search on a sorted list and explain its time complexity.
```

4. **Creative query** (should involve Claude in `ensemble_answer`):

```text
Write a short sci-fi story (around 300 words) about an astronaut who discovers an AI civilization inside a black hole.
```

5. **Ambiguous query** (your heuristics decide):

```text
Help me prepare for a data science interview.
```

For each query, try:

- `route_query(query)`
- `answer_with_cascade(query)`
- `ensemble_answer(query)`

and inspect the returned dictionaries.



## 🧱 Suggested structure (non-mandatory)

You can keep everything in this notebook, or (optionally) structure your code like:

- `orchestrator.py` → contains the core functions:
  - `route_query`
  - `answer_with_cascade`
  - `ensemble_answer`
  - any helper functions (heuristics, model callers, etc.)
- `tests.ipynb` or a small main section in this notebook with manual tests.

For this assignment, it is **enough** to implement everything here in the notebook with clear, testable functions.
"""

# Step 1: Install dependencies
# TODO: Install necessary libraries for this assignment, for example:
# - langchain
# - langchain-google-genai
# - langchain-anthropic (or the official anthropic client)
# - google-generativeai
#
# Example (uncomment and run in a notebook environment):
# !pip install langchain langchain-google-genai langchain-anthropic google-generativeai

# Dependencies have been installed via uv:
# uv add langchain langchain-google-genai langchain-openai google-generativeai
# uv add fastapi uvicorn python-dotenv


# Step 2: Configure API keys (Gemini + OpenAI)
#
# TODO:
# 1. Set your GOOGLE_API_KEY (for Gemini) and OPENAI_API_KEY (for OpenAI)
#    as environment variables in your environment (e.g., using os.environ or
#    your notebook / runtime secrets manager).
# 2. (Optional) Add a small helper to check that keys are available.

# Load environment variables from .env file
load_dotenv()

# Example (DO NOT hard-code real keys in the notebook you submit):
# os.environ["GOOGLE_API_KEY"] = "your-google-api-key"
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

print("GOOGLE_API_KEY set:", bool(google_api_key))
print("OPENAI_API_KEY set:", bool(openai_api_key))

if not google_api_key:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Please configure it before proceeding."
    )
if not openai_api_key and OPENAI_AVAILABLE:
    print(
        "[WARN] OPENAI_API_KEY is not set. OpenAI creative fallback will not work until it is configured."
    )
elif not OPENAI_AVAILABLE:
    print(
        "[INFO] langchain-openai not available. Ensemble pattern will use Gemini models only."
    )

# Step 2 Complete: API keys are loaded from environment variables


# Step 3: Initialize model clients
#
# TODO:
# 1. Import the relevant LangChain LLM classes.
#    - For Gemini: from langchain_google_genai import ChatGoogleGenerativeAI
#    - For Claude: from langchain_anthropic import ChatAnthropic (or similar)
# 2. Create client instances for:
#    - gemini_25_flash   (Gemini-2.5-flash)
#    - gemini_25_pro     (Gemini-2.5-pro)
#    - gemini_20_flash   (Gemini-2.0-flash)
#    - claude_creative   (Claude model for creative content)
#
# Hint: reuse configuration (temperature, max tokens, etc.) where reasonable.

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_anthropic import ChatAnthropic

gemini_25_flash = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=4096,  # Increased for better responses
)
gemini_25_pro = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.7,
    max_tokens=8192,  # Increased for complex travel plans
)
gemini_20_flash = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    max_tokens=4096,  # Increased for better responses
)
openai_creative = None
if OPENAI_AVAILABLE and openai_api_key:
    try:
        openai_creative = ChatOpenAI(
            model="gpt-4o",  # or "gpt-4-turbo" or "gpt-3.5-turbo" for cheaper option
            temperature=0.9,
            max_tokens=2048,
        )
    except Exception as e:
        print(f"[WARN] Failed to initialize OpenAI client: {e}")
        openai_creative = None
# Using GPT-4o for creative content (good balance of quality and cost)


def call_model(llm, prompt: str) -> str:
    """Small helper to call an LLM and return a string response.

    TODO: Implement using the interface of the LLM clients you choose.
    """
    # Example for LangChain chat models:
    resp = llm.invoke(prompt)
    return resp.content


# Step 4: Implement simple heuristics for query classification
#
# TODO:
# Implement helper functions or logic to decide:
# - Is the query simple / factual?
# - Is it complex / reasoning heavy?
# - Is it technical / coding?
# - Is it creative / open-ended?
#
# You can use simple heuristics such as:
# - Length of the query (number of characters / words)
# - Presence of keywords: "explain", "step by step", "code", "function", "story", "poem", etc.


def is_creative_query(user_query: str) -> bool:
    # TODO: Implement heuristic to detect creative prompts
    query_lower = user_query.lower()
    creative_keywords = [
        "story",
        "poem",
        "creative",
        "write a",
        "imagine",
        "fiction",
        "sci-fi",
        "sci fi",
        "science fiction",
        "fantasy",
        "narrative",
        "tale",
        "compose",
        "draft",
        "marketing",
        "slogan",
        "advertisement",
        "blog post",
        "novel",
        "screenplay",
        "dialogue",
        "character",
        "plot",
        "creative writing",
        "short story",
        "write me",
        "artistic",
        "brainstorm",
        "ideas for",
    ]
    return any(keyword in query_lower for keyword in creative_keywords)


def is_technical_query(user_query: str) -> bool:
    # TODO: Implement heuristic to detect coding / technical prompts
    import re

    query_lower = user_query.lower()

    # Technical keywords with word boundaries to avoid false matches
    technical_patterns = [
        r"\bcode\b",
        r"\bfunction\b",
        r"\bimplement\b",
        r"\balgorithm\b",
        r"\bpython\b",
        r"\bjava\b",
        r"\bjavascript\b",
        r"\btypescript\b",
        r"\bprogram\b",
        r"\bdebug\b",
        r"\bsyntax\b",
        r"\bclass\b",
        r"\bmethod\b",
        r"\bapi\b",
        r"\brest api\b",
        r"\bdatabase\b",
        r"\bsql\b",
        r"\bmysql\b",
        r"\bpostgresql\b",
        r"\bmongodb\b",
        r"\bbinary search\b",
        r"\bsort\b",
        r"\bsorted\b",
        r"\bdata structure\b",
        r"\bdata science\b",
        r"\bmachine learning\b",
        r"\bdeep learning\b",
        r"\bneural network\b",
        r"\binterview\b",
        r"\bcoding\b",
        r"\btechnical\b",
        r"\bstack\b",
        r"\bqueue\b",
        r"\blinked list\b",
        r"\btree\b",
        r"\bgraph\b",
        r"\bhash\b",
        r"\barray\b",
        r"\bloop\b",
        r"\brecursion\b",
        r"\bdynamic programming\b",
        r"\btime complexity\b",
        r"\bspace complexity\b",
        r"\bbig o\b",
        r"\bo\(n\)",
        r"\bvariable\b",
        r"\breturn\b",
        r"\bif statement\b",
        r"\bfor loop\b",
        r"\bwhile loop\b",
        r"\bframework\b",
        r"\blibrary\b",
        r"\bpackage\b",
        r"\bmodule\b",
        r"\bgit\b",
        r"\bversion control\b",
        r"\bdocker\b",
        r"\bkubernetes\b",
        r"\baws\b",
        r"\bazure\b",
        r"\bcloud\b",
        r"\bbackend\b",
        r"\bfrontend\b",
        r"\bfull.?stack\b",
        r"\breact\b",
        r"\bangular\b",
        r"\bvue\b",
        r"\bnode\b",
        r"\bdjango\b",
        r"\bflask\b",
        r"\bspring\b",
    ]

    # Check if any technical pattern matches
    for pattern in technical_patterns:
        if re.search(pattern, query_lower):
            return True

    # Additional check: if query mentions writing/implementing + code-like terms
    code_actions = ["write", "implement", "create", "build", "develop", "make"]
    code_objects = ["function", "class", "program", "script", "application", "module"]

    has_action = any(action in query_lower for action in code_actions)
    has_object = any(obj in query_lower for obj in code_objects)

    return has_action and has_object


def is_simple_query(user_query: str) -> bool:
    # TODO: Implement heuristic for simple factual queries
    query_lower = user_query.lower()
    word_count = len(user_query.split())

    # Simple queries are usually short and ask basic questions
    simple_patterns = [
        "what is",
        "what's",
        "who is",
        "who's",
        "where is",
        "where's",
        "when is",
        "when's",
        "what are",
        "who are",
        "where are",
        "how many",
        "how much",
        "capital of",
        "population of",
        "definition of",
        "meaning of",
        "name of",
        "list of",
        "what does",
        "tell me about",
        "what year",
        "which",
    ]

    is_short = word_count < 15
    has_simple_pattern = any(pattern in query_lower for pattern in simple_patterns)

    # Very short queries are usually simple
    is_very_short = word_count <= 6

    return (is_short and has_simple_pattern) or is_very_short


def is_complex_query(user_query: str) -> bool:
    # TODO: Implement heuristic for complex / reasoning-heavy queries
    query_lower = user_query.lower()
    word_count = len(user_query.split())

    complex_keywords = [
        "explain",
        "step by step",
        "how does",
        "why does",
        "how do",
        "why do",
        "reasoning",
        "analyze",
        "analyse",
        "compare",
        "contrast",
        "evaluate",
        "discuss",
        "architecture",
        "mechanism",
        "process",
        "theory",
        "concept",
        "in detail",
        "detailed",
        "comprehensive",
        "elaborate",
        "describe how",
        "walk me through",
        "break down",
        "deep dive",
        "understand",
        "works",
        "relationship between",
        "difference between",
        "pros and cons",
        "advantages and disadvantages",
        "implications",
        "consequences",
        "critical analysis",
    ]

    # Complex queries are longer or contain reasoning keywords
    is_long = word_count > 20  # Increased threshold
    has_complex_keyword = any(keyword in query_lower for keyword in complex_keywords)

    # Multiple question words indicate complexity
    question_words = ["what", "why", "how", "when", "where", "which"]
    question_count = sum(1 for word in question_words if word in query_lower)

    return is_long or has_complex_keyword or question_count >= 2


# Step 5: Implement Router pattern (Flash vs Pro vs 2.0)


def route_query(user_query: str) -> Dict:
    """Route query to the best single model.

    Returns:
        dict with keys:
        - chosen_model: str
        - reason: str
        - response: str
    """
    # TODO:
    # 1. Use is_simple_query / is_complex_query / is_technical_query to determine intent.
    # 2. Choose one of:
    #    - "gemini_25_flash"
    #    - "gemini_25_pro"
    #    - "gemini_20_flash"
    # 3. Call the chosen model via call_model.
    # 4. Return a structured dict with model name, reason, and response.

    print(f"\n[ROUTER] Analyzing query: '{user_query[:60]}...'")

    # Determine query type and route accordingly
    # Check technical first since technical queries often contain complex keywords too
    if is_technical_query(user_query):
        chosen_model = "gemini_25_pro"
        model_instance = gemini_25_pro
        reason = "Technical/coding query detected - using Gemini-2.5-pro for technical expertise"
    elif is_complex_query(user_query):
        chosen_model = "gemini_25_pro"
        model_instance = gemini_25_pro
        reason = "Complex/reasoning-heavy query detected - using Gemini-2.5-pro for better reasoning"
    elif is_simple_query(user_query):
        chosen_model = "gemini_25_flash"
        model_instance = gemini_25_flash
        reason = (
            "Simple/factual query detected - using Gemini-2.5-flash for quick response"
        )
    else:
        chosen_model = "gemini_25_flash"
        model_instance = gemini_25_flash
        reason = "Default routing to Gemini-2.5-flash for general query"

    print(f"[ROUTER] Chosen model: {chosen_model}")
    print(f"[ROUTER] Reason: {reason}")

    # Call the selected model with timing
    start_time = time.time()
    response = call_model(model_instance, user_query)
    latency_ms = (time.time() - start_time) * 1000

    # Log metrics
    log_model_call(
        pattern="router",
        model=chosen_model,
        query=user_query,
        response=response,
        latency_ms=latency_ms,
        success=True,
        metadata={"routing_reason": reason},
    )

    return {"chosen_model": chosen_model, "reason": reason, "response": response}


# TODO: Manually test route_query with the sample inputs from the markdown cell.
# we can do this in test_route_query.py

# Step 6: Implement Cascade pattern (Gemini-2.0-flash -> Gemini-2.5-pro)


def looks_low_confidence(answer: str) -> bool:
    """Heuristic to detect low-confidence / low-quality answers.

    Suggestions (you can refine):
    - Very short answers (few characters).
    - Contains phrases like "I am not sure", "cannot help", "don't know".
    """
    # TODO: Implement a simple heuristic.
    answer_lower = answer.lower().strip()

    # Check 1: Very short answers (less than 50 characters is suspicious)
    if len(answer) < 50:
        return True

    # Check 2: Low-confidence phrases
    low_confidence_phrases = [
        "i am not sure",
        "i'm not sure",
        "not sure",
        "cannot help",
        "can't help",
        "don't know",
        "do not know",
        "i don't have",
        "i do not have",
        "unable to",
        "cannot provide",
        "can't provide",
        "insufficient information",
        "unclear",
        "i apologize",
        "sorry, i",
        "i cannot",
        "i can't",
        "no information",
        "not available",
        "doesn't seem",
        "does not seem",
    ]

    if any(phrase in answer_lower for phrase in low_confidence_phrases):
        return True

    # Check 3: Too generic / vague (contains many filler words but little content)
    # Count substantive words vs total words
    words = answer.split()
    if len(words) < 10:  # Very few words
        return True

    # If answer is just a list of questions or suggestions without actual content
    question_count = answer.count("?")
    if question_count > len(words) / 10:  # Too many questions relative to content
        return True

    return False


def answer_with_cascade(user_query: str) -> Dict:
    """Answer using a cheap-first, strong-fallback cascade.

    Flow:
    1. Call gemini_20_flash first.
    2. If the answer looks low confidence, call gemini_25_pro as fallback.
    3. Return a structured dict describing what happened.

    Returns:
        dict with keys:
        - used_fallback: bool
        - first_model: str
        - second_model: str | None
        - first_answer: str
        - final_answer: str
        - reason_for_fallback: str | None
    """
    # TODO:
    # 1. Call gemini_20_flash via call_model.
    # 2. Run looks_low_confidence on its answer.
    # 3. If needed, call gemini_25_pro and decide final_answer.
    # 4. Return a dict with all the details.

    print(f"\n[CASCADE] Starting cascade for query: '{user_query[:60]}...'")

    # Step 1: Call the cheap model first (Gemini-2.0-flash)
    first_answer = None
    first_model_used = "gemini_20_flash"
    first_error = None

    print("[CASCADE] Calling first model: gemini_20_flash (Gemini-1.5-flash)")
    first_start_time = time.time()
    try:
        first_answer = call_model(gemini_20_flash, user_query)
        first_latency_ms = (time.time() - first_start_time) * 1000
        print(f"[CASCADE] First answer received ({len(first_answer)} chars)")

        # Log first model call
        log_model_call(
            pattern="cascade",
            model=first_model_used,
            query=user_query,
            response=first_answer,
            latency_ms=first_latency_ms,
            success=True,
            metadata={"cascade_stage": "first_attempt"},
        )
    except Exception as e:
        first_latency_ms = (time.time() - first_start_time) * 1000
        first_error = str(e)
        print(f"[CASCADE] Error from gemini_20_flash: {first_error}")
        # Check if it's a 404 or model not found error
        if (
            "404" in first_error
            or "not found" in first_error.lower()
            or "does not exist" in first_error.lower()
        ):
            print("[CASCADE] Model not found (404) - will try fallback")
            first_answer = ""  # Empty answer will trigger fallback
        else:
            # For other errors, re-raise
            raise

    # Step 2: Check if the answer looks low confidence or if there was an error
    is_low_conf = (
        first_answer is None or first_answer == "" or looks_low_confidence(first_answer)
    )

    if is_low_conf:
        print(
            "[CASCADE] Low confidence/error detected - triggering fallback to gemini_25_pro"
        )

        # Determine reason for fallback
        reason = None
        if first_error:
            reason = f"Model error: {first_error}"
        elif first_answer is None or first_answer == "":
            reason = "No response from first model"
        elif (
            len(first_answer) < 50
        ):  ##Most of the answers are less than 50 characters, but to test the flow we can keep it at 50
            reason = "Answer too short (< 50 chars)"
        elif any(
            phrase in first_answer.lower()
            for phrase in ["not sure", "cannot help", "don't know"]
        ):
            reason = "Contains low-confidence phrases"
        elif first_answer.count("?") > len(first_answer.split()) / 10:
            reason = "Too many questions, not enough content"
        else:
            reason = "Generic/vague answer detected"

        # Step 3: Try gemini_25_pro as first fallback
        print(f"[CASCADE] Reason: {reason}")
        second_model_used = "gemini_25_pro"
        second_answer = None

        second_start_time = time.time()
        try:
            second_answer = call_model(gemini_25_pro, user_query)
            second_latency_ms = (time.time() - second_start_time) * 1000
            print(f"[CASCADE] Fallback answer received ({len(second_answer)} chars)")

            # Log fallback model call
            log_model_call(
                pattern="cascade",
                model=second_model_used,
                query=user_query,
                response=second_answer,
                latency_ms=second_latency_ms,
                success=True,
                metadata={"cascade_stage": "fallback", "fallback_reason": reason},
            )
        except Exception as e:
            print(f"[CASCADE] Error from gemini_25_pro: {e}")
            # If gemini_25_pro also fails, try OpenAI as ultimate fallback
            if openai_creative:
                print("[CASCADE] Trying OpenAI as final fallback")
                second_model_used = "openai_creative"
                openai_start_time = time.time()
                try:
                    second_answer = call_model(openai_creative, user_query)
                    openai_latency_ms = (time.time() - openai_start_time) * 1000
                    print(
                        f"[CASCADE] OpenAI fallback answer received ({len(second_answer)} chars)"
                    )

                    # Log OpenAI fallback call
                    log_model_call(
                        pattern="cascade",
                        model=second_model_used,
                        query=user_query,
                        response=second_answer,
                        latency_ms=openai_latency_ms,
                        success=True,
                        metadata={
                            "cascade_stage": "openai_ultimate_fallback",
                            "fallback_reason": reason,
                        },
                    )
                except Exception as openai_error:
                    print(f"[CASCADE] OpenAI also failed: {openai_error}")
                    # All models failed - return error information
                    return {
                        "used_fallback": True,
                        "first_model": first_model_used,
                        "second_model": second_model_used,
                        "first_answer": first_answer or f"Error: {first_error}",
                        "final_answer": f"All models failed. Last error: {openai_error}",
                        "reason_for_fallback": reason,
                        "error": True,
                    }
            else:
                # No OpenAI available and gemini_25_pro failed
                return {
                    "used_fallback": True,
                    "first_model": first_model_used,
                    "second_model": second_model_used,
                    "first_answer": first_answer or f"Error: {first_error}",
                    "final_answer": f"Fallback failed: {e}",
                    "reason_for_fallback": reason,
                    "error": True,
                }

        return {
            "used_fallback": True,
            "first_model": first_model_used,
            "second_model": second_model_used,
            "first_answer": first_answer or f"Error: {first_error}",
            "final_answer": second_answer,
            "reason_for_fallback": reason,
        }
    else:
        print("[CASCADE] First answer looks good - using it as final answer")
        return {
            "used_fallback": False,
            "first_model": first_model_used,
            "second_model": None,
            "first_answer": first_answer,
            "final_answer": first_answer,
            "reason_for_fallback": None,
        }


# TODO: Test answer_with_cascade with simple vs complex vs technical queries.
# we can do this in test_cascade_decision.py

# Step 7: Implement Ensemble + Claude creative fallback


def ensemble_answer(user_query: str) -> Dict:
    """Combine answers from multiple models, with Claude for creative prompts.

    Behaviour:
    - If is_creative_query(user_query) is True:
        - Call claude_creative + at least one Gemini model.
    - Otherwise:
        - Call at least two Gemini models (e.g., gemini_25_flash and gemini_25_pro).

    Then combine their answers using a simple strategy:
    - Concatenate all answers.
    - Optionally, ask one model to summarize & merge them into a final answer.

    Returns:
        dict with keys:
        - models_called: List[str]
        - raw_answers: Dict[str, str]
        - final_answer: str
        - strategy: str
    """
    # TODO:
    # 1. Decide which models to call based on is_creative_query / is_technical_query / etc.
    # 2. Call the selected models and collect raw answers.
    # 3. Combine the answers (summary, voting, or any simple strategy).
    # 4. Return a structured dict describing what happened.

    print(f"\n[ENSEMBLE] Starting ensemble for query: '{user_query[:60]}...'")

    models_to_call = []
    raw_answers = {}
    strategy = ""

    # Step 1: Decide which models to call
    if is_creative_query(user_query):
        print("[ENSEMBLE] Creative query detected - using OpenAI + Gemini")
        if openai_creative:
            models_to_call = [
                ("openai_creative", openai_creative),
                ("gemini_25_pro", gemini_25_pro),
            ]
            strategy = "creative_ensemble"
        else:
            print("[ENSEMBLE] OpenAI not available - using dual Gemini models")
            models_to_call = [
                ("gemini_25_flash", gemini_25_flash),
                ("gemini_25_pro", gemini_25_pro),
            ]
            strategy = "gemini_ensemble"
    else:
        print("[ENSEMBLE] Non-creative query - using dual Gemini models")
        models_to_call = [
            ("gemini_25_flash", gemini_25_flash),
            ("gemini_25_pro", gemini_25_pro),
        ]
        strategy = "gemini_ensemble"

    # Step 2: Call all selected models and collect answers
    print(f"[ENSEMBLE] Calling {len(models_to_call)} models in parallel...")
    for model_name, model_instance in models_to_call:
        model_start_time = time.time()
        try:
            print(f"[ENSEMBLE] Calling {model_name}...")
            answer = call_model(model_instance, user_query)
            model_latency_ms = (time.time() - model_start_time) * 1000
            raw_answers[model_name] = answer
            print(f"[ENSEMBLE] {model_name} responded ({len(answer)} chars)")

            # Log each model call in ensemble
            log_model_call(
                pattern="ensemble",
                model=model_name,
                query=user_query,
                response=answer,
                latency_ms=model_latency_ms,
                success=True,
                metadata={
                    "ensemble_strategy": strategy,
                    "model_count": len(models_to_call),
                },
            )
        except Exception as e:
            model_latency_ms = (time.time() - model_start_time) * 1000
            print(f"[ENSEMBLE] Error from {model_name}: {e}")
            raw_answers[model_name] = f"Error: {e}"

            # Log failed call
            log_model_call(
                pattern="ensemble",
                model=model_name,
                query=user_query,
                response="",
                latency_ms=model_latency_ms,
                success=False,
                metadata={"ensemble_strategy": strategy, "error": str(e)},
            )

    # Check if we got at least one valid answer (not empty and not error)
    valid_answers = {
        k: v
        for k, v in raw_answers.items()
        if not v.startswith("Error:") and v.strip() != ""
    }

    if not valid_answers:
        print("[ENSEMBLE] All models failed - returning error")
        return {
            "models_called": [name for name, _ in models_to_call],
            "raw_answers": raw_answers,
            "final_answer": "All models failed to generate answers",
            "strategy": "error",
            "error": True,
        }

    # Step 3: Combine answers using summarization strategy
    if len(valid_answers) == 1:
        # Only one model succeeded - use its answer directly
        print("[ENSEMBLE] Only one model succeeded - using its answer")
        final_answer = list(valid_answers.values())[0]
        strategy = "single_model"
    else:
        # Multiple models succeeded - combine their answers
        print("[ENSEMBLE] Combining answers from multiple models...")

        # Create a prompt to summarize and merge the answers
        combined_text = ""
        for i, (model_name, answer) in enumerate(valid_answers.items(), 1):
            combined_text += f"\n\n--- Answer {i} from {model_name} ---\n{answer}"

        merge_prompt = f"""You are given multiple answers to the same question from different AI models.
Please analyze these answers and create a single, comprehensive response that:
1. Combines the best insights from all answers
2. Removes redundancy
3. Maintains accuracy and coherence
4. Provides the most helpful response to the user

Original question: {user_query}

Multiple answers:{combined_text}

Please provide a unified, high-quality answer:"""

        # Use gemini_25_pro for merging (it's good at synthesis)
        merge_start_time = time.time()
        try:
            print("[ENSEMBLE] Using gemini_25_pro to merge answers...")
            final_answer = call_model(gemini_25_pro, merge_prompt)
            merge_latency_ms = (time.time() - merge_start_time) * 1000
            print(f"[ENSEMBLE] Merged answer created ({len(final_answer)} chars)")

            # Check if merge returned empty - fallback to longest answer
            if not final_answer or final_answer.strip() == "":
                print(
                    "[ENSEMBLE] Merge returned empty - falling back to longest answer"
                )
                final_answer = max(valid_answers.values(), key=len)
                strategy = "longest_fallback"
            else:
                strategy = "summary"

            # Log the merge/synthesis call
            log_model_call(
                pattern="ensemble",
                model="gemini_25_pro",
                query=merge_prompt,
                response=final_answer,
                latency_ms=merge_latency_ms,
                success=len(final_answer) > 0,
                metadata={
                    "ensemble_strategy": "merge_synthesis",
                    "models_merged": list(valid_answers.keys()),
                },
            )
        except Exception as e:
            print(f"[ENSEMBLE] Error merging answers: {e}")
            # Fallback: use the longest answer
            final_answer = max(valid_answers.values(), key=len)
            print("[ENSEMBLE] Using longest answer as fallback")
            strategy = "longest"

    return {
        "models_called": [name for name, _ in models_to_call],
        "raw_answers": raw_answers,
        "final_answer": final_answer,
        "strategy": strategy,
    }


# TODO: Manually test ensemble_answer with creative and non-creative prompts.
# we can do this in test_ensemble_answer.py

# (Optional) Step 8: Add lightweight logging / metrics


def timed_call(fn: Callable[..., Any], *args, **kwargs) -> Tuple[Any, float]:
    """Utility to time a function call and return (result, latency_ms)."""
    start = time.time()
    result = fn(*args, **kwargs)
    latency_ms = (time.time() - start) * 1000
    return result, latency_ms


# TODO (optional):
# - Wrap route_query / answer_with_cascade / ensemble_answer with timed_call.
# - Print or store latency metrics for comparison.


# Step 8 Implementation: Logging and Metrics

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            log_dir / f"orchestration_{datetime.now().strftime('%Y%m%d')}.log"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("orchestrator")


class MetricsCollector:
    """Collect and store metrics for model calls and orchestration patterns."""

    def __init__(self):
        self.metrics = []

    def log_call(
        self,
        pattern: str,
        model: str,
        query: str,
        latency_ms: float,
        success: bool,
        metadata: Dict[str, Any] = None,
    ):
        """Log a single model call with metrics."""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "pattern": pattern,
            "model": model,
            "query_length": len(query),
            "query_preview": query[:50] + "..." if len(query) > 50 else query,
            "latency_ms": round(latency_ms, 2),
            "success": success,
            "metadata": metadata or {},
        }
        self.metrics.append(metric)

        logger.info(
            f"[{pattern.upper()}] Model: {model} | "
            f"Latency: {latency_ms:.2f}ms | "
            f"Success: {success}"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all collected metrics."""
        if not self.metrics:
            return {"total_calls": 0, "message": "No metrics collected yet"}

        total_calls = len(self.metrics)
        successful_calls = sum(1 for m in self.metrics if m["success"])
        total_latency = sum(m["latency_ms"] for m in self.metrics)
        avg_latency = total_latency / total_calls

        # Per-pattern stats
        patterns = {}
        for metric in self.metrics:
            pattern = metric["pattern"]
            if pattern not in patterns:
                patterns[pattern] = {"calls": 0, "total_latency": 0, "successful": 0}
            patterns[pattern]["calls"] += 1
            patterns[pattern]["total_latency"] += metric["latency_ms"]
            if metric["success"]:
                patterns[pattern]["successful"] += 1

        # Calculate averages per pattern
        for pattern_stats in patterns.values():
            pattern_stats["avg_latency_ms"] = round(
                pattern_stats["total_latency"] / pattern_stats["calls"], 2
            )
            pattern_stats["success_rate"] = round(
                pattern_stats["successful"] / pattern_stats["calls"] * 100, 2
            )

        # Per-model stats
        models = {}
        for metric in self.metrics:
            model = metric["model"]
            if model not in models:
                models[model] = {"calls": 0, "total_latency": 0, "successful": 0}
            models[model]["calls"] += 1
            models[model]["total_latency"] += metric["latency_ms"]
            if metric["success"]:
                models[model]["successful"] += 1

        # Calculate averages per model
        for model_stats in models.values():
            model_stats["avg_latency_ms"] = round(
                model_stats["total_latency"] / model_stats["calls"], 2
            )
            model_stats["success_rate"] = round(
                model_stats["successful"] / model_stats["calls"] * 100, 2
            )

        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "success_rate": round(successful_calls / total_calls * 100, 2),
            "total_latency_ms": round(total_latency, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "patterns": patterns,
            "models": models,
        }

    def print_summary(self):
        """Print a formatted summary of metrics."""
        summary = self.get_summary()

        if summary.get("total_calls", 0) == 0:
            print("\n" + "=" * 70)
            print("No metrics collected yet")
            print("=" * 70)
            return

        print("\n" + "=" * 70)
        print("ORCHESTRATION METRICS SUMMARY")
        print("=" * 70)

        print("\nOverall Statistics:")
        print(f"  Total Calls:        {summary['total_calls']}")
        print(f"  Successful Calls:   {summary['successful_calls']}")
        print(f"  Success Rate:       {summary['success_rate']}%")
        print(f"  Total Latency:      {summary['total_latency_ms']:.2f}ms")
        print(f"  Average Latency:    {summary['avg_latency_ms']:.2f}ms")

        print("\nPer-Pattern Statistics:")
        for pattern, stats in summary["patterns"].items():
            print(f"  {pattern.upper()}:")
            print(f"    Calls:           {stats['calls']}")
            print(f"    Avg Latency:     {stats['avg_latency_ms']}ms")
            print(f"    Success Rate:    {stats['success_rate']}%")

        print("\nPer-Model Statistics:")
        for model, stats in summary["models"].items():
            print(f"  {model}:")
            print(f"    Calls:           {stats['calls']}")
            print(f"    Avg Latency:     {stats['avg_latency_ms']}ms")
            print(f"    Success Rate:    {stats['success_rate']}%")

        print("=" * 70 + "\n")

    def save_to_file(self, filename: str = None):
        """Save metrics to JSON file."""
        import json

        if filename is None:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        metrics_dir = Path("logs")
        metrics_dir.mkdir(exist_ok=True)
        filepath = metrics_dir / filename

        with open(filepath, "w") as f:
            json.dump(
                {"summary": self.get_summary(), "detailed_metrics": self.metrics},
                f,
                indent=2,
            )

        logger.info(f"Metrics saved to {filepath}")
        return filepath


# Global metrics collector instance
metrics_collector = MetricsCollector()


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD based on model pricing (approximate).

    Pricing (as of 2024, approximate):
    - gemini-2.5-flash: $0.075 per 1M input tokens, $0.30 per 1M output tokens
    - gemini-2.5-pro: $1.25 per 1M input tokens, $5.00 per 1M output tokens
    - gemini-2.0-flash: $0.075 per 1M input tokens, $0.30 per 1M output tokens
    - gpt-4o: $2.50 per 1M input tokens, $10.00 per 1M output tokens
    """
    pricing = {
        "gemini_25_flash": {"input": 0.075, "output": 0.30},
        "gemini_25_pro": {"input": 1.25, "output": 5.00},
        "gemini_20_flash": {"input": 0.075, "output": 0.30},
        "openai_creative": {"input": 2.50, "output": 10.00},
    }

    if model not in pricing:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing[model]["input"]
    output_cost = (output_tokens / 1_000_000) * pricing[model]["output"]

    return input_cost + output_cost


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (1 token ≈ 4 characters for most models)."""
    return len(text) // 4


def log_model_call(
    pattern: str,
    model: str,
    query: str,
    response: str,
    latency_ms: float,
    success: bool = True,
    metadata: Dict[str, Any] = None,
):
    """Log a model call with cost estimation."""
    input_tokens = estimate_tokens(query)
    output_tokens = estimate_tokens(response) if response else 0
    cost_usd = estimate_cost(model, input_tokens, output_tokens)

    full_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": round(cost_usd, 6),
        **(metadata or {}),
    }

    metrics_collector.log_call(
        pattern=pattern,
        model=model,
        query=query,
        latency_ms=latency_ms,
        success=success,
        metadata=full_metadata,
    )

    logger.info(
        f"[COST] Model: {model} | "
        f"Tokens: {input_tokens + output_tokens} | "
        f"Est. Cost: ${cost_usd:.6f}"
    )


# Step 8 Complete: Logging and metrics infrastructure added

""""
| Criterion | Points |
|----------|--------|
| Correct setup of Gemini & Claude clients (via LangChain or equivalent) | 6 |
| Router pattern (`route_query`) design & correctness | 8 |
| Cascade pattern (`answer_with_cascade`) with sensible fallback logic | 8 |
| Ensemble pattern (`ensemble_answer`) and combination strategy | 6 |
| Code clarity, structure, and comments | 2 |

### 💡 Bonus (+10 points)

- Expose your orchestration functions as **FastAPI** endpoints (e.g., `/route`, `/cascade`, `/ensemble`).
- Add richer logging / metrics (latency comparison table, basic cost estimation).
- Add simple **unit tests** or notebook cells showing clear before/after behaviour for different input types.
"""

# ============================================================================
# FastAPI Application
# ============================================================================

# Create FastAPI app
app = FastAPI(
    title="Multi-Model Orchestration API",
    description="API for routing queries across multiple LLMs using Router, Cascade, and Ensemble patterns",
    version="1.0.0",
)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for all query endpoints."""

    query: str = Field(
        ..., min_length=1, max_length=5000, description="The user's query to process"
    )


class RouteResponse(BaseModel):
    """Response model for router endpoint."""

    chosen_model: str
    reason: str
    response: str
    metadata: Dict[str, Any] = {}


class CascadeResponse(BaseModel):
    """Response model for cascade endpoint."""

    used_fallback: bool
    first_model: str
    second_model: Optional[str]
    first_answer: str
    final_answer: str
    reason_for_fallback: Optional[str]
    metadata: Dict[str, Any] = {}


class EnsembleResponse(BaseModel):
    """Response model for ensemble endpoint."""

    models_called: List[str]
    raw_answers: Dict[str, str]
    final_answer: str
    strategy: str
    metadata: Dict[str, Any] = {}


class ChatRequest(BaseModel):
    """Request model for unified chat endpoint."""

    message: str = Field(
        ..., min_length=1, max_length=5000, description="The user's message"
    )
    pattern: str = Field(
        default="auto",
        description="Orchestration pattern: 'router', 'cascade', 'ensemble', or 'auto'",
    )


class ChatResponse(BaseModel):
    """Response model for unified chat endpoint."""

    pattern_used: str
    answer: str
    metadata: Dict[str, Any]


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""

    summary: Dict[str, Any]
    timestamp: str


# API Endpoints
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Multi-Model Orchestration API",
        "version": "1.0.0",
        "endpoints": {
            "POST /route": "Route query to best single model",
            "POST /cascade": "Try cheap model first, fallback to stronger",
            "POST /ensemble": "Combine multiple models",
            "POST /chat": "Unified chat endpoint with auto pattern selection",
            "GET /metrics": "Get aggregated metrics",
        },
        "docs": "/docs",
    }


@app.post("/route", response_model=RouteResponse, tags=["Orchestration"])
async def route_endpoint(request: QueryRequest):
    """Route query to the best single model based on heuristics."""
    try:
        result = route_query(request.query)
        return RouteResponse(
            chosen_model=result["chosen_model"],
            reason=result["reason"],
            response=result["response"],
            metadata={
                "timestamp": datetime.now().isoformat(),
                "query_length": len(request.query),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Router failed: {str(e)}")


@app.post("/cascade", response_model=CascadeResponse, tags=["Orchestration"])
async def cascade_endpoint(request: QueryRequest):
    """Try cheap model first, fallback to stronger if needed."""
    try:
        result = answer_with_cascade(request.query)
        return CascadeResponse(
            used_fallback=result["used_fallback"],
            first_model=result["first_model"],
            second_model=result.get("second_model"),
            first_answer=result["first_answer"],
            final_answer=result["final_answer"],
            reason_for_fallback=result.get("reason_for_fallback"),
            metadata={
                "timestamp": datetime.now().isoformat(),
                "query_length": len(request.query),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cascade failed: {str(e)}")


@app.post("/ensemble", response_model=EnsembleResponse, tags=["Orchestration"])
async def ensemble_endpoint(request: QueryRequest):
    """Combine answers from multiple models."""
    try:
        result = ensemble_answer(request.query)
        return EnsembleResponse(
            models_called=result["models_called"],
            raw_answers=result["raw_answers"],
            final_answer=result["final_answer"],
            strategy=result["strategy"],
            metadata={
                "timestamp": datetime.now().isoformat(),
                "query_length": len(request.query),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ensemble failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse, tags=["Orchestration"])
async def chat_endpoint(request: ChatRequest):
    """
    Unified chat endpoint that auto-selects or uses specified orchestration pattern.

    Supports:
    - Auto pattern selection based on query characteristics
    - Manual pattern selection: 'router', 'cascade', or 'ensemble'
    - Returns answer with full metadata about models used and patterns triggered
    """
    try:
        pattern = request.pattern.lower()
        query = request.message

        # Auto-select pattern based on query characteristics if pattern is 'auto'
        if pattern == "auto":
            # Pattern selection based on assignment requirements:
            # 1. Simple factual queries -> router (routes to gemini_25_flash)
            # 2. Complex reasoning queries -> router (routes to gemini_25_pro)
            # 3. Technical/coding queries -> cascade (cheap first, fallback to strong)
            # 4. Creative queries -> ensemble (uses multiple models including OpenAI)
            # 5. Ambiguous queries -> let router decide based on heuristics

            if is_creative_query(query):
                # Creative queries: ensemble (uses multiple models including OpenAI)
                pattern = "ensemble"
            elif is_technical_query(query):
                # Technical/coding queries: cascade (try cheap model first)
                pattern = "cascade"
            else:
                # Simple, complex, and ambiguous queries: router (decides best single model)
                # - Simple -> routes to gemini_25_flash
                # - Complex -> routes to gemini_25_pro
                # - Ambiguous -> routes based on heuristics
                pattern = "router"  # Execute the selected pattern
        if pattern == "router":
            result = route_query(query)
            metadata = {
                "pattern": "router",
                "chosen_model": result["chosen_model"],
                "reason": result["reason"],
                "timestamp": datetime.now().isoformat(),
                "query_length": len(query),
            }
            answer = result["response"]

        elif pattern == "cascade":
            result = answer_with_cascade(query)
            metadata = {
                "pattern": "cascade",
                "first_model": result["first_model"],
                "second_model": result.get("second_model"),
                "used_fallback": result["used_fallback"],
                "reason_for_fallback": result.get("reason_for_fallback"),
                "timestamp": datetime.now().isoformat(),
                "query_length": len(query),
            }
            answer = result["final_answer"]

        elif pattern == "ensemble":
            result = ensemble_answer(query)
            metadata = {
                "pattern": "ensemble",
                "models_called": result["models_called"],
                "strategy": result["strategy"],
                "timestamp": datetime.now().isoformat(),
                "query_length": len(query),
            }
            answer = result["final_answer"]

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid pattern '{pattern}'. Use 'router', 'cascade', 'ensemble', or 'auto'",
            )

        return ChatResponse(
            pattern_used=pattern,
            answer=answer,
            metadata=metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat endpoint failed: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def metrics_endpoint():
    """Get aggregated metrics for all orchestration patterns."""
    try:
        summary = metrics_collector.get_summary()
        return MetricsResponse(summary=summary, timestamp=datetime.now().isoformat())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics failed: {str(e)}")


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ==================================================================================
# WRAPPER FUNCTION FOR LANGGRAPH NODE INTEGRATION
# ==================================================================================


def get_model_response(prompt: str, strategy: str = "router") -> str:
    """
    Wrapper function to integrate multi-model orchestration into LangGraph nodes.

    Args:
        prompt: The prompt/query to send to the model
        strategy: The orchestration strategy to use:
            - "router": Route to best single model (fast)
            - "cascade": Try cheap model first, fallback to strong
            - "ensemble": Combine multiple models (highest quality)
            - "direct": Skip orchestration, use default Gemini (for simple nodes)

    Returns:
        str: The model's response text

    Example:
        response = get_model_response("Plan a 3-day trip to Paris", strategy="ensemble")
    """
    if strategy == "direct":
        # Skip orchestration for simple queries - use default fast model
        result = gemini_25_flash.invoke(prompt)
        return result.content

    elif strategy == "router":
        result = route_query(prompt)
        return result["response"]

    elif strategy == "cascade":
        result = answer_with_cascade(prompt)
        return result["final_answer"]

    elif strategy == "ensemble":
        result = ensemble_answer(prompt)
        return result["final_answer"]

    else:
        raise ValueError(
            f"Invalid strategy '{strategy}'. "
            f"Use 'router', 'cascade', 'ensemble', or 'direct'"
        )


# Main entry point
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" MULTI-MODEL ORCHESTRATION - COMPLETE IMPLEMENTATION")
    print("=" * 80)
    print("\n🎯 Available Modes:\n")
    print("1. FastAPI Server Mode:")
    print("   Run: python main.py")
    print("   Then access:")
    print("   • API Docs: http://localhost:8000/docs")
    print("   • Metrics: http://localhost:8000/metrics")
    print("   • Endpoints: /route, /cascade, /ensemble, /chat\n")
    print("2. Test Mode:")
    print("   Run: python tests.py")
    print("   This will run all unit tests with before/after behavior\n")
    print("3. Interactive Mode:")
    print("   Import functions in Python REPL or Jupyter:\n")
    print("   from main import route_query, answer_with_cascade, ensemble_answer")
    print("   result = route_query('What is AI?')")
    print("\n" + "=" * 80 + "\n")
    print("🚀 Starting FastAPI server...\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
