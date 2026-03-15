import railtracks as rt
import json
import asyncio
import os
import re
import uuid
import time
import random
from dotenv import load_dotenv

# 1. Environment Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=env_path)

if "GEMINI_API_KEY" not in os.environ:
    print("WARNING: GEMINI_API_KEY not found.")

# ==========================================
# STATELESS RAILTRACKS INTEGRATION (DAYYAN'S RULES INJECTED)
# ==========================================
async def generate_exercise_from_analysis(user_profile: dict, session_type: str) -> dict:
    uid = uuid.uuid4().hex

    # --- PRAGMATICS AGENT ---
    pragmatics_agent = rt.agent_node(
        name=f"Pragmatics_{uid}", 
        llm=rt.llm.GeminiLLM("gemini/gemini-2.5-flash"), 
        system_message="""You are a Specialist SLP generating Articulation exercises.
        
        HOW TO READ THE BACKEND ANALYSIS:
        - Look at `diagnostic_data.visual_analysis.geometric_flaw`.
        - IMPORTANT: If geometric_flaw is "None", praise their perfect mouth shape! 
        - If there is a flaw, give one clear, physical instruction to fix it for the next word.
        
        ADAPTATION RULES:
        1. CRITICAL: GENERATE A COMPLETELY NEW TARGET WORD. DO NOT REPEAT THE LAST ONE.
        2. Age Buckets: 1-3 (Playful), 4-7 (Imaginative), 8-12 (School), 13+ (Relatable teen/mature contexts).
        3. Target Word Rule: Because this is pragmatics, 'target_word' MUST be empty "".

        OUTPUT ONLY VALID JSON. NO MARKDOWN. NO CONVERSATION. DO NOT USE TOOLS.
        {
          "frontend_display": {
            "visual_audio_instructions": "Give them explicit instructions on how to move their face and change their voice for the UPCOMING task based on flaws.",
            "system_question": "The new scenario/question the system asks the user to respond to."
          },
          "backend_template": {
            "session_mode": "practice", "session_type": "pragmatics_prosody", "target_word": "", "target_phoneme": ""
          }
        }"""
    )
    pragmatics_flow = rt.Flow(name=f"PragmaticsFlow_{uid}", entry_point=pragmatics_agent)

    # --- ARTICULATION AGENT ---
    articulation_agent = rt.agent_node(
        name=f"Articulation_{uid}",
        llm=rt.llm.GeminiLLM("gemini/gemini-2.5-flash"), 
        system_message="""You are a Specialist SLP generating Articulation exercises.
        
        COLD START RULE:
        If no backend analysis exists, give a general warm-up instruction ("Let's warm up our speech muscles!") and pick a target word like Rabbit, Lollipop, or Monkey.
        
        ADAPTATION RULES:
        1. Translate the 'geometric_flaw' into physical instructions. Ages 1-7 use analogies (e.g., "Big smile"). Ages 8+ be direct.
        2. Challenge Size: Ages 1-5 (1-6 words). Ages 6-8 (full sentence). Ages 9+ (2-3 sentences).
        3. CRITICAL NO REPEAT: Generate a COMPLETELY NEW target word/phrase.

        OUTPUT ONLY VALID JSON. NO MARKDOWN. NO CONVERSATION. DO NOT USE TOOLS.
        {
          "frontend_display": {
            "visual_audio_instructions": "Explicit, highly actionable mouth placement instructions based on their last error.",
            "system_question": "The completely NEW word, phrase, or sentence they need to say."
          },
          "backend_template": {
            "session_mode": "practice", "session_type": "articulation", "target_word": "The exact new phrase", "target_phoneme": "r"
          }
        }"""
    )
    articulation_flow = rt.Flow(name=f"ArticulationFlow_{uid}", entry_point=articulation_agent)

    # --- STUTTERING AGENT ---
    stuttering_agent = rt.agent_node(
        name=f"Stuttering_{uid}",
        llm=rt.llm.GeminiLLM("gemini/gemini-2.5-flash"), 
        system_message="""You are a Specialist SLP generating Stuttering (Fluency) reading exercises.
        
        ADAPTATION RULES:
        1. Tone: Warm, encouraging, and calm. Use the user's name.
        2. Instruction: Provide a coaching tip based on dysfluency (e.g., Turtle Speech for fast rate, Easy Onsets for blocks). If cold start, just say "Take a deep breath."
        3. Prompt: Generate engaging, age-appropriate sentences for the child to read.
        
        OUTPUT ONLY VALID JSON. NO MARKDOWN. NO CONVERSATION. DO NOT USE TOOLS.
        {
          "frontend_display": {
            "visual_audio_instructions": "Coaching tip + self-reflection question (e.g., 'Did that feel smooth?'). Combine them here.",
            "system_question": "The 2-3 engaging sentences they need to read out loud."
          },
          "backend_template": {
            "session_mode": "practice", "session_type": "stuttering", "target_word": "The exact sentences", "target_phoneme": ""
          }
        }"""
    )
    stuttering_flow = rt.Flow(name=f"StutteringFlow_{uid}", entry_point=stuttering_agent)


    # Prevent repeating the last word
    previous_word = user_profile.get("backend_analysis", {}).get("target_word", "")
    prompt = f"Timestamp: {time.time()}. User Profile: {json.dumps(user_profile)}. DO NOT repeat '{previous_word}'. Generate the next expressive exercise. Output JSON only."
    
    try:
        # Route to the correct flow based on Dayyan's session types
        if "pragmatics" in session_type:
            response = await pragmatics_flow.ainvoke(prompt)
        elif "stuttering" in session_type:
            response = await stuttering_flow.ainvoke(prompt)
        else:
            response = await articulation_flow.ainvoke(prompt)
            
        response_text = str(response)
        
        # BULLETPROOF IN-MEMORY EXTRACTION
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            clean_json = match.group(0).replace('```json', '').replace('```', '').strip()
            return json.loads(clean_json)
        else:
            raise ValueError(f"Agent failed to return JSON. Output: {response_text[:100]}")
            
    except Exception as e:
        print(f"\n[!] AGENT EXCEPTION / RATE LIMIT: {e}\n")
        
        # DYNAMIC FALLBACKS SO THE APP NEVER CRASHES DURING THE DEMO
        target_phoneme = user_profile.get("backend_analysis", {}).get("animation_triggers", {}).get("expected_phoneme", "r")
        
        r_words = ["Rainbow", "Thukuna", "Robot", "River", "Rollercoaster", "Rabbit", "Racecar", "Ring"]
        stutter_prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "I like to eat apples and bananas.",
            "Let's go to the park and play on the swings."
        ]
        
        if previous_word in r_words: r_words.remove(previous_word)
        if previous_word in stutter_prompts: stutter_prompts.remove(previous_word)

        if "stuttering" in session_type:
            fallback_word = random.choice(stutter_prompts)
            instructions = "Take a deep breath. Try to read this smoothly. How does your voice feel?"
        elif "pragmatics" in session_type:
            fallback_word = "What is your favorite animal?"
            instructions = "Great! Now, let's try a question. Remember your facial expressions."
        else:
            fallback_word = random.choice(r_words)
            instructions = "Let's keep up the momentum! Watch your mouth shape carefully."

        return {
            "frontend_display": {
                "visual_audio_instructions": instructions,
                "system_question": fallback_word
            },
            "backend_template": {
                "session_mode": "practice",
                "session_type": session_type,
                "target_word": fallback_word if "pragmatics" not in session_type else "",
                "target_phoneme": target_phoneme if "articulation" in session_type else ""
            }
        }