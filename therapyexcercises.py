import railtracks as rt
import json
import asyncio
from dotenv import load_dotenv
import os
load_dotenv()  # take environment variables from .env.
api_key = os.getenv("API_KEY")

# Ensure the Gemini API key is set in your environment variables.
# You can set it here for testing, but DO NOT commit this key to GitHub!
# os.environ["GEMINI_API_KEY"] = "your_actual_gemini_api_key_here"

if "GEMINI_API_KEY" not in os.environ:
    print("⚠️ WARNING: GEMINI_API_KEY environment variable is not set!")
    print("Please run: export GEMINI_API_KEY='your_key' before running this script.")
    exit(1)


# ==========================================
# 1. DEFINE THE TOOL
# ==========================================

@rt.function_node
def save_next_exercise(json_string: str) -> str:
    """Validates and saves the generated Pragmatics exercise to a JSON file."""
    try:
        # Strip markdown formatting if the LLM adds it (e.g., ```json ... ```)
        clean_json = json_string.strip("`").replace("json\n", "")
        data = json.loads(clean_json)
        
        # Validate critical Pydantic keys for the backend
        if "backend_template" not in data:
            raise ValueError("Missing 'backend_template'.")
            
        with open("next_pragmatics_exercise.json", "w") as f:
            json.dump(data, f, indent=2)
            
        return "SUCCESS: Pragmatics exercise saved to next_pragmatics_exercise.json"
    except Exception as e:
        return f"ERROR: Invalid JSON format. {e}"


# ==========================================
# 2. BUILD THE PRAGMATICS-ONLY AGENT
# ==========================================

pragmatics_slp_agent = rt.agent_node(
    name="Pragmatics_SLP_Agent",
    tool_nodes=(save_next_exercise,),
    # 👇 FIX: Changed OpenAILLM to GeminiLLM and used the strict LiteLLM Gemini format
    llm=rt.llm.GeminiLLM("gemini/gemini-2.5-flash"), 
    system_message="""You are an expert Speech-Language Pathologist specializing ONLY in Pragmatic Language and Prosody.
    Do NOT generate articulation or stuttering exercises.
    
    Based on the child's previous performance data, generate EXACTLY ONE practice exercise.
    Choose between two phases based on the DLD clinical study:
    - "Phase 1: Receptive" (Decoding meaning from Nova's flat vs. expressive voice)
    - "Phase 2: Expressive" (Child must use eyebrows and pitch contour to ask a question or make a request)
    
    You MUST output a single JSON object. It MUST contain two keys: "frontend_display" and "backend_template".
    
    SCHEMA REQUIREMENTS:
    {
      "frontend_display": {
        "phase": "1_receptive" OR "2_expressive",
        "kid_friendly_instructions": "What the UI text says to the child.",
        "nova_action": "What the 3D avatar should do (e.g., 'flat_voice_no_movement' or 'pleading_voice_head_tilt')",
        "options": ["Picture A Description", "Picture B Description"] // ONLY include if Phase 1. Empty array if Phase 2.
      },
      "backend_template": {
        "session_mode": "practice",
        "session_type": "pragmatics_prosody",
        "target_word": "The exact phrase to say (Leave blank if Phase 1)",
        "target_phoneme": ""  // CRITICAL: Always an empty string for Pragmatics!
      }
    }
    
    Pass the final JSON string into the `save_next_exercise` tool.
    """
)


# ==========================================
# 3. DEFINE THE FLOW
# ==========================================

pragmatics_flow = rt.Flow(
    name="Pragmatics Generation Flow",
    entry_point=pragmatics_slp_agent
)


# ==========================================
# 4. HOW TO TEST THIS LOCALLY
# ==========================================

async def run_test():
    # TEST CASE: Updated to perfectly match your partner's exact dictionary structure!
    # The child got the audio right (pitch was rising), but failed the visual (eyebrows flat).
    mock_partner_data = """
    {
      "status": "fail",
      "session_mode": "practice",
      "session_type": "pragmatics_prosody",
      "nova_feedback": "Hmm, I didn't understand. Remember, a good detective uses their eyebrows and makes their voice go UP at the end of a question!",
      "animation_triggers": {
        "target_word": "Is the monster gone?",
        "failed_word": "Is the monster gone?",
        "target_viseme": "",
        "observed_viseme": "",
        "expected_phoneme": "",
        "observed_phoneme": "error_detected"
      },
      "diagnostic_data": {
        "audio_analysis": {
          "prosody_score": 85.0,
          "audio_insight": "Pitch contour was perfectly rising.",
          "pass": true
        },
        "visual_analysis": {
          "pass": false,
          "geometric_flaw": "Eyebrows remained flat; interrogative facial affect not detected.",
          "metrics": {
            "browInnerUp": 0.1,
            "head_tilt_z": 0.05
          }
        }
      }
    }
    """
    
    print("🕵️‍♂️ Analyzing partner data and generating next Pragmatics exercise with Gemini...")
    result = await pragmatics_flow.ainvoke(f"Based on this result, generate a follow-up exercise:\n{mock_partner_data}")
    print("✅ Done! Check your local folder for 'next_pragmatics_exercise.json'")

if __name__ == "__main__":
    asyncio.run(run_test())