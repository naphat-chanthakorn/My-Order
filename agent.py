import os
import sys
import logging
import wikipedia # อย่าลืม pip install wikipedia
from datetime import datetime

sys.path.append("..")
from callback_logging import log_query_to_model, log_model_response
from dotenv import load_dotenv
import google.cloud.logging
from google.adk import Agent
from google.genai import types
from typing import Optional, List, Dict

from google.adk.tools.tool_context import ToolContext

load_dotenv()

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

# ==========================================
# 🛠️ 1. Tools Definition
# ==========================================

def search_wikipedia(query: str) -> str:
    """Searches Wikipedia for a specific query to find historical facts.
    
    Args:
        query (str): The search keyword (e.g., 'Genghis Khan achievements').
    
    Returns:
        str: A summary of the wikipedia page.
    """
    try:
        # Limit sentences to avoid token overflow
        return wikipedia.summary(query, sentences=3)
    except wikipedia.exceptions.PageError:
        return "Page not found."
    except wikipedia.exceptions.DisambiguationError:
        return "Topic is ambiguous, please refine keyword."
    except Exception as e:
        return f"Error searching: {str(e)}"

def save_evidence_to_state(
    tool_context: ToolContext,
    content: str,
    evidence_type: str
) -> dict[str, str]:
    """Saves the research findings to the session state (pos_data or neg_data).

    Args:
        content (str): The text content found from research.
        evidence_type (str): Must be either 'positive' or 'negative'.

    Returns:
        dict: Status message.
    """
    # Determine the key based on agent type
    key = "pos_data" if evidence_type == "positive" else "neg_data"
    
    # Get existing data
    current_data = tool_context.state.get(key, [])
    if isinstance(current_data, str): # Handle if ADK stored it as string
        current_data = [current_data]
        
    # Append new evidence
    current_data.append(content)
    
    # Update state
    tool_context.state[key] = current_data
    
    return {"status": "success", "message": f"Saved to {key}"}

def set_topic(
    tool_context: ToolContext,
    topic: str
) -> dict[str, str]:
    """Sets the historical topic for the trial.
    
    Args:
        topic (str): The name of the person or event (e.g., 'Genghis Khan').
    """
    tool_context.state["topic"] = topic
    # Initialize empty evidence lists
    tool_context.state["pos_data"] = []
    tool_context.state["neg_data"] = []
    return {"status": "success", "topic": topic}

def deliver_verdict(
    tool_context: ToolContext,
    final_verdict: str
) -> dict[str, str]:
    """Saves the final verdict to a text file and ends the trial.
    
    Args:
        final_verdict (str): The structured text of the final judgement.
    """
    topic = tool_context.state.get("topic", "Unknown_Topic")
    filename = f"Verdict_{topic.replace(' ', '_')}.txt"
    
    header = f"=== ⚖️ VERDICT: {topic} ===\nDate: {datetime.now()}\n{'='*30}\n\n"
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(header + final_verdict)
        return {"status": "success", "message": f"Verdict saved to {filename}. CASE CLOSED."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==========================================
# 🤖 2. Agents Definition
# ==========================================

# --- Agent A: The Admirer ---
admirer_agent = Agent(
    name="The_Admirer",
    model=os.getenv("MODEL"),
    description="Researches positive aspects, achievements, and legacies.",
    instruction="""
    You are 'The Admirer'.
    1. Look at the topic: { topic? }
    2. Use 'search_wikipedia' with keywords like 'achievements', 'success', 'legacy'.
    3. IMPORTANT: Once you find info, use 'save_evidence_to_state' with evidence_type='positive'.
    4. Report back briefly what you found.
    """,
    tools=[search_wikipedia, save_evidence_to_state],
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
)

# --- Agent B: The Critic ---
critic_agent = Agent(
    name="The_Critic",
    model=os.getenv("MODEL"),
    description="Researches negative aspects, controversies, and failures.",
    instruction="""
    You are 'The Critic'.
    1. Look at the topic: { topic? }
    2. Use 'search_wikipedia' with keywords like 'controversy', 'criticism', 'failures', 'crimes'.
    3. IMPORTANT: Once you find info, use 'save_evidence_to_state' with evidence_type='negative'.
    4. Report back briefly what you found.
    """,
    tools=[search_wikipedia, save_evidence_to_state],
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
)

# --- Agent C: The Judge (Root Agent) ---
# ทำหน้าที่เป็น Orchestrator คุม Loop และตัดสิน
root_agent = Agent(
    name="The_Judge",
    model=os.getenv("MODEL"), 
    # ใช้ Model ที่ฉลาดหน่อย (เช่น gemini-1.5-pro) ถ้า config ใน env ได้
    description="The Chief Justice who controls the trial flow.",
    instruction="""
    You are 'The Judge' of the Historical Court.
    
    Your Goal: Create a balanced report on the topic: { topic? }
    
    Current Evidence State:
    [PROS]: { pos_data? }
    [CONS]: { neg_data? }
    
    Procedure:
    1. If { topic? } is empty, ask the user for a topic and use 'set_topic'.
    2. If evidence is missing or unbalanced:
       - Send 'The_Admirer' to find pros.
       - Send 'The_Critic' to find cons.
    3. Review the gathered data. If it's still shallow, order them to search specifically again.
    4. When evidence is sufficient, write a Final Verdict comprising:
       - Executive Summary
       - The Admiration (Pros)
       - The Criticism (Cons)
       - Final Judgement
    5. CALL 'deliver_verdict' tool to save the file.
    """,
    tools=[set_topic, deliver_verdict],
    sub_agents=[admirer_agent, critic_agent],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2, # ต่ำเพื่อให้คุม Logic ได้ดี
    ),
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
)
