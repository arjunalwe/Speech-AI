import railtracks as rt
import json
import asyncio
from dotenv import load_dotenv
import os

# 1. Environment Setup
load_dotenv()
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
    """Saves ANY generated exercise (Pragmatics or Articulation) to the same file."""
    try:
        clean_json = json_string.strip("`").replace("json\n", "")
        data = json.loads(clean_json)
        # Always save to the exact same file so the frontend knows where to look
        with open("next_exercise.json", "w") as f:
            json.dump(data, f, indent=2)
        return "SUCCESS"
    except Exception as e:
        return f"ERROR: {e}"

# ==========================================
# PRAGMATICS AGENT
# ==========================================

pragmatics_slp_agent = rt.agent_node(
    name="Age_Aware_DLD_Agent",
    tool_nodes=(save_next_exercise,),
    llm=rt.llm.GeminiLLM("gemini/gemini-2.5-flash"), 
    system_message="""You are a Specialist SLP generating Pragmatic Language exercises.
    You will receive a user's NAME, AGE, and their BACKEND ANALYSIS JSON from their last attempt.
    
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

# ==========================================
# ARTICULATION AGENT 
# ==========================================

articulation_slp_agent = rt.agent_node(
    name="Articulation_SLP_Agent",
    tool_nodes=(save_next_exercise,), # Uses the same unified saving tool!
    llm=rt.llm.GeminiLLM("gemini/gemini-2.5-flash"), 
    system_message="""You are a Specialist SLP generating Articulation exercises.
    You will receive a user's NAME, AGE, and their BACKEND ANALYSIS JSON.
    
    HOW TO READ THE BACKEND ANALYSIS:
    - Look at `expected_phoneme` vs `observed_phoneme`.
    - Look at `diagnostic_data.visual_analysis.geometric_flaw` to see HOW their mouth moved incorrectly (e.g., lips rounded instead of spread).
    
    ADAPTATION RULES:
    1. Tone: Use the user's NAME.
    2. Instructions: Translate the 'geometric_flaw' into crystal clear, physical instructions appropriate for their age. 
       - For young children (1-7), use physical analogies (e.g., "Make a big Cheerio smile", "Hide your tongue behind your top teeth"). Do NOT use clinical terms like 'elevation' or 'rounding'.
       - For older users (8+), be direct (e.g., "Make sure your lips are pulled back").
    3. Challenge Size: Based on age, provide a target word, phrase, or sentence that heavily features the target phoneme. (Ages 1-5: 1 word. Ages 6-8: 2-3 words. Ages 9+: full sentence).
    4. Target Word Rule: You MUST fill in 'target_word' with the exact phrase you generated, and 'target_phoneme' with the sound they are practicing.

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

# ==========================================
# TEST FUNCTION
# ==========================================

async def run_test():
    # MOCK INPUT FROM FRONTEND (Articulation Scenario)
    mock_backend_analysis = {
      "status": "success",
      "session_type": "articulation_gliding",
      "nova_feedback": "Almost! I saw your lips turn into a little circle, which makes a 'W' sound. To make the 'R' sound, pull your lips back into a huge Cheerio smile!",
      "animation_triggers": {
        "target_word": "rabbit",
        "failed_word": "wabbit",
        "target_viseme": "liquid_r_shape",
        "observed_viseme": "labial_w_shape",
        "expected_phoneme": "r", 
        "observed_phoneme": "w"
      },
      "diagnostic_data": {
        "phoneme_analysis": {
          "word_score": 42.0,
          "error_type": "Substitution",
          "phoneme_accuracy": 15.0
        },
        "visual_analysis": {
          "pass": False,
          "geometric_flaw": "Lips were too rounded; required tongue-tip elevation not detected.",
          "metrics": {
            "peak_pucker": 0.82,
            "peak_smile": 0.15
          }
        }
      },
      "feedback_payload": {
        "status": "fail",
        "feedback": "goo goo gaa gaa"
      }
    }

    # Set age to 6 to test the "short phrase" generation rule and "kid-friendly" instructions
    user_profile = {
        "name": "Leo",
        "age": 6,
        "backend_analysis": mock_backend_analysis
    }
    
    prompt = f"User Profile and Previous Attempt Data: {json.dumps(user_profile)}. Generate the next expressive articulation exercise."
    
    print(f"Generating Articulation exercise for {user_profile['name']} ({user_profile['age']})...")
    
    result = await articulation_flow.ainvoke(prompt)
    print(f"Agent Response: {result}")
    print("Done! Check next_exercise.json")

if __name__ == "__main__":
    asyncio.run(run_test())