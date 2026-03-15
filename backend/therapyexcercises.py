import railtracks as rt
import json
import asyncio
import os
import re
from dotenv import load_dotenv

# 1. Environment Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=env_path)

if "GEMINI_API_KEY" not in os.environ:
    print("WARNING: GEMINI_API_KEY not found.")
    exit(1)

# ==========================================
# UNIFIED SAVING TOOL
# ==========================================
@rt.function_node
def save_next_exercise(json_string: str) -> str:
    """Saves ANY generated exercise (Pragmatics, Articulation, or Stuttering) to the same file."""
    try:
        clean_string = json_string.strip("`").replace("json\n", "")
        match = re.search(r'\{.*\}', clean_string, re.DOTALL)
        if match:
            clean_string = match.group(0)
            
        data = json.loads(clean_string)
        with open("next_exercise.json", "w") as f:
            json.dump(data, f, indent=2)
        return "SUCCESS"
    except Exception as e:
        print(f"Tool Error: {e}")
        return f"ERROR: {e}"

# ==========================================
# BRIDGE TO SERVER.PY (STATELESS FIX)
# ==========================================
async def generate_exercise_from_analysis(user_profile: dict, session_type: str) -> dict:
    """Instantiates fresh agents per request to prevent LLM memory poisoning."""
    
    # 1. Clean up stale files
    if os.path.exists("next_exercise.json"):
        os.remove("next_exercise.json")

    # 2. Recreate agents freshly every time with YOUR updated prompts
    
    # --- PRAGMATICS AGENT ---
    pragmatics_slp_agent = rt.agent_node(
        name="Pragmatics_SLP_Agent",
        tool_nodes=(save_next_exercise,),
        llm=rt.llm.GeminiLLM("gemini/gemini-2.5-flash"), 
        system_message="""You are a Specialist SLP generating Pragmatic Language exercises.
        You will receive a user's NAME, AGE, INTAKE GOALS, and their BACKEND ANALYSIS JSON.
        
        COLD START RULE (NO DATA):
        If `feedback_payload.feedback` is exactly "" (empty string), this is their first time. Do NOT invent past flaws.
        - Instructions: Give a warm, general welcome using their name.
        - Question Selection: 
          a) Check if the user profile provided an `intake_pragmatic_goal` (e.g., "asking for help"). If so, ask a baseline question matching that goal.
          b) If no intake goal exists, randomly select ONE of the following:
             1. "Can you ask me what my favorite game is?"
             2. "If the room is too dark, how would you ask someone to turn on the light?"
        
        HOW TO READ THE BACKEND ANALYSIS:
        - Look closely at `diagnostic_data.visual_analysis.geometric_flaw` or `feedback_payload.feedback` to figure out exactly what the user struggled with.
        
        ADAPTATION RULES:
        1. Tone: Use the user's NAME to build rapport. Adjust tone strictly based on age.
        2. Age Buckets:
           - 1-3: Simple words, very playful.
           - 4-7: Imaginative/Game themes (Detectives, Magic).
           - 8-12: School/Social life themes.
           - 13-15: Relatable teen social scenarios (Sarcasm, Politeness).
           - 15+: Professional or mature social contexts (Work, Interviews).
        3. Clinical Target: Focus on Interrogatives (questions) and Indirect Requests.
        4. Target Word Rule: Because this is pragmatics, 'target_word' MUST be empty "".

        JSON SCHEMA REQUIREMENTS:
        {
          "frontend_display": {
            "visual_audio_instructions": "Give them explicit instructions on how to move their face and change their voice for the UPCOMING task based on the flaws found in their backend analysis.",
            "system_question": "The new scenario/question the system asks the user to respond to."
          },
          "backend_template": {
            "session_mode": "practice",
            "session_type": "pragmatics_prosody",
            "target_word": "",
            "target_phoneme": ""
          }
        }
        Step 1: Save JSON via tool.
        Step 2: Return the exact string 'Exercise Ready.'
        """
    )
    pragmatics_flow = rt.Flow(name="PragmaticsFlow", entry_point=pragmatics_slp_agent)

    # --- ARTICULATION AGENT ---
    articulation_slp_agent = rt.agent_node(
        name="Articulation_SLP_Agent",
        tool_nodes=(save_next_exercise,), 
        llm=rt.llm.GeminiLLM("gemini/gemini-2.5-flash"), 
        system_message="""You are a Specialist SLP generating Articulation exercises.
        You will receive a user's NAME, AGE, INTAKE GOALS, and their BACKEND ANALYSIS JSON.
        
        COLD START RULE (NO DATA):
        If `feedback_payload.feedback` is exactly "" (empty string), this is their first time. Do NOT invent past flaws.
        - Instructions: Give a general warm-up instruction (e.g., "Let's warm up our speech muscles! Watch my mouth and try to copy me.").
        - Target Selection:
          a) Check if the user profile provided an `intake_target_phoneme` (e.g., "s"). If it exists, generate an age-appropriate target word that uses that phoneme.
          b) If no intake phoneme was provided, randomly select ONE of the following defaults: "Rabbit" (r), "Lollipop" (l), "Monkey" (m).
        
        HOW TO READ THE BACKEND ANALYSIS (IF DATA EXISTS):
        - Look at `expected_phoneme` vs `observed_phoneme`.
        - Look at `diagnostic_data.visual_analysis.geometric_flaw` to see HOW their mouth moved incorrectly (e.g., lips rounded instead of spread).
        
        ADAPTATION RULES:
        1. Tone: Use the user's NAME.
        2. Instructions: Translate the 'geometric_flaw' into crystal clear, physical instructions appropriate for their age. 
           - For young children (1-7), use easy to understand physical analogies (e.g., "Make a big smile", "Hide your tongue behind your top teeth"). Do NOT use clinical terms like 'elevation' or 'rounding'.
           - For older users (8+), be direct (e.g., "Make sure your lips are pulled back").
        3. Challenge Size: Based on age, provide a target word, phrase, or sentence that heavily features the target phoneme. (Ages 1-5: 1-6 words. Ages 6-8: a full sentence. Ages 9+: 2-3 full sentences). You may adjust the difficulty of reading with age
        4. Target Word Rule: You MUST fill in 'target_word' with the exact phrase you generated, and 'target_phoneme' with the sound they are practicing, so make sure there is 1 or more words that have the phoneme.

        JSON SCHEMA REQUIREMENTS:
        {
          "frontend_display": {
            "visual_audio_instructions": "Explicit, highly actionable mouth placement instructions for the new word based on their last error.",
            "system_question": "The new word, phrase, or sentence they need to say."
          },
          "backend_template": {
            "session_mode": "practice",
            "session_type": "articulation",
            "target_word": "The exact word/phrase generated in system_question",
            "target_phoneme": "The phoneme letter (e.g., 'r')"
          }
        }
        Step 1: Save JSON via tool.
        Step 2: Return the exact string 'Exercise Ready.'
        """
    )
    articulation_flow = rt.Flow(name="ArticulationFlow", entry_point=articulation_slp_agent)

    # --- STUTTERING AGENT ---
    stuttering_slp_agent = rt.agent_node(
        name="Stuttering_SLP_Agent",
        tool_nodes=(save_next_exercise,), 
        llm=rt.llm.GeminiLLM("gemini/gemini-2.5-flash"), 
        system_message="""You are a Specialist SLP generating Stuttering (Fluency) reading exercises for children, and occasionally adults.
        
        GUIDING PRINCIPLE:
        Use the provided User Profile and Backend Analysis as clinical evidence. Do not expect a fixed format in the 'feedback' field; use all available data to adapt the response.
        
        COLD START RULE (NO DATA):
        If `feedback_payload.feedback` is exactly "" (empty string), this is their first time.
        - Instruction: Give a general relaxing, calming instruction (e.g., "Take a deep breath and let's try some smooth, easy speech.").
        - Prompt: Provide a simple, fun 2-sentence baseline reading task.
        - Self-Reflection: Ask a general question about how their voice feels.

        ADAPTATION RULES (IF DATA EXISTS):
        1. Tone: Warm, encouraging, and calm. Use the user's name.
        2. Instruction: Provide a coaching tip based on previous dysfluency (e.g., Turtle Speech for fast rate, Easy Onsets for blocks).
        3. Prompt: Generate 2-3 engaging, age-appropriate sentences for the child to read, given their age.
        4. Self-Reflection: Formulate a question that helps the child check in with their speech (e.g., 'Did that feel smooth or bumpy?').
        5. Target Word Rule: 'target_word' in the backend_template must contain the reading prompt sentences.

        JSON SCHEMA REQUIREMENTS:
        {
          "frontend_display": {
            "instruction": "Coaching tip or warm welcome.",
            "prompt": "The sentences to read.",
            "self_reflection": "The self-monitoring question."
          },
          "backend_template": {
            "session_mode": "practice",
            "session_type": "stuttering_fluency",
            "target_word": "The exact prompt sentences.",
            "target_phoneme": ""
          }
        }
        Step 1: Save JSON via tool.
        Step 2: Return 'Exercise Ready.'
        """
    )
    stuttering_flow = rt.Flow(name="StutteringFlow", entry_point=stuttering_slp_agent)

    # 3. Execution Logic
    prompt = f"User Profile and Previous Attempt Data: {json.dumps(user_profile)}. Generate the next expressive exercise."
    
    try:
        # Route to the correct flow based on session_type
        if "pragmatics" in session_type:
            response = await pragmatics_flow.ainvoke(prompt)
        elif "stuttering" in session_type:
            response = await stuttering_flow.ainvoke(prompt)
        else:
            response = await articulation_flow.ainvoke(prompt)
            
        # Parse the output
        if os.path.exists("next_exercise.json"):
            with open("next_exercise.json", "r") as f:
                return json.load(f)
        else:
            match = re.search(r'\{.*\}', str(response), re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise Exception("Agent failed to use tool and returned invalid JSON.")
            
    except Exception as e:
        print(f"Error generating next exercise: {e}")
        target_phoneme = user_profile.get("backend_analysis", {}).get("animation_triggers", {}).get("expected_phoneme", "r")
        
        # Fallback JSON to prevent server crashes
        fallback_data = {
            "frontend_display": {
                "visual_audio_instructions": f"Great effort! I noticed your form was slightly off. Let's try another one targeting our {target_phoneme} sound.",
                "system_question": "Strawberry" if session_type == "articulation" else "What is your favorite food?"
            },
            "backend_template": {
                "session_mode": "practice",
                "session_type": session_type,
                "target_word": "Strawberry" if session_type == "articulation" else "",
                "target_phoneme": target_phoneme
            }
        }
        # Small adjustment for stuttering fallback
        if "stuttering" in session_type:
             fallback_data["frontend_display"]["instruction"] = "Take a deep breath and try this one."
             fallback_data["frontend_display"]["prompt"] = "The quick brown fox jumps over the lazy dog."
             fallback_data["frontend_display"]["self_reflection"] = "How did that feel?"
             fallback_data["backend_template"]["target_word"] = "The quick brown fox jumps over the lazy dog."
             
        return fallback_data