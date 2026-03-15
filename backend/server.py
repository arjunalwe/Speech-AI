import os
import base64
import asyncio
import tempfile
import subprocess
import re  # <-- Added for advanced stuttering text detection
import imageio_ffmpeg
import azure.cognitiveservices.speech as speechsdk
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import google.generativeai as gemini

# Import the agentic generation function from your teammate's file
from therapyexcercises import generate_exercise_from_analysis

load_dotenv()
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MOCK_DB = {
    "user_123": {
        "name": "Leo",
        "age": 6,
        "disorder_focus": "articulation"
    }
}

class Shapes(BaseModel):
    mouthClose: float = 0.0
    mouthPucker: float = 0.0
    mouthFunnel: float = 0.0
    mouthRollLower: float = 0.0
    mouthUpperUp: float = 0.0
    jawOpen: float = 0.0
    browInnerUp: float = 0.0
    mouthSmileLeft: float = 0.0
    mouthSmileRight: float = 0.0
    browDownLeft: float = 0.0
    browDownRight: float = 0.0
    eyeWideLeft: float = 0.0
    eyeWideRight: float = 0.0
    eyeBlinkLeft: float = 0.0
    eyeBlinkRight: float = 0.0

class Pose(BaseModel):
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0

class Frame(BaseModel):
    time_ms: int
    shapes: Shapes
    pose: Pose

class AudioMetrics(BaseModel):
    max_volume_spike: float = 0.0
    longest_silence_ms: int = 0

class Session(BaseModel):
    user_id: str
    name: str = "Buddy"
    age: int = 5
    session_mode: str
    session_type: str
    target_word: str
    target_phoneme: str
    audio_base64: str
    frames: List[Frame]
    audio_metrics: Optional[AudioMetrics] = None
    

async def convert_audio(base64_str: str) -> bytes:
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    base64_str += "=" * ((4 - len(base64_str) % 4) % 4)
    try:
        audio_bytes = base64.b64decode(base64_str)
    except Exception:
        raise HTTPException(status_code=400)
    if len(audio_bytes) < 100:
        return b"fake_audio_bytes"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_in:
        temp_in.write(audio_bytes)
        temp_in_path = temp_in.name
    temp_out_path = temp_in_path + "_out.wav"
    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        command = [ffmpeg_path, "-y", "-i", temp_in_path, "-ar", "16000", "-ac", "1", "-f", "wav", temp_out_path]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        with open(temp_out_path, "rb") as f:
            wav_bytes = f.read()
        return wav_bytes
    finally:
        if os.path.exists(temp_in_path):
            os.remove(temp_in_path)
        if os.path.exists(temp_out_path):
            os.remove(temp_out_path)

def call_azure_pronunciation(wav_bytes: bytes, reference_text: str) -> dict:
    if wav_bytes == b"fake_audio_bytes":
        return {"words": [], "prosody_score": 0}
    
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    stream_format = speechsdk.audio.AudioStreamFormat(16000, 16, 1)
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format)
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
    
    pron_config = speechsdk.PronunciationAssessmentConfig(reference_text, speechsdk.PronunciationAssessmentGradingSystem.HundredMark, speechsdk.PronunciationAssessmentGranularity.Phoneme)
    recognizer = speechsdk.SpeechRecognizer(speech_config, audio_config)
    
    pron_config.apply_to(recognizer)
    push_stream.write(wav_bytes)
    push_stream.close()
    result = recognizer.recognize_once_async().get()
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        p_res = speechsdk.PronunciationAssessmentResult(result)
        words = []
        for w in p_res.words:
            words.append({
                "word": w.word,
                "accuracy": w.accuracy_score,
                "phonemes": [{"text": ph.phoneme, "accuracy_score": ph.accuracy_score} for ph in w.phonemes]
            })
        return {
            "words": words, 
            "prosody_score": p_res.pronunciation_score,
            "recognized_text": result.text
        }
    return {"words": [], "prosody_score": 0}

def facial_evaluation(frames: List[Frame], session_type: str, target_phoneme: str) -> dict:
    if not frames:
        return {"pass": False, "target_viseme": "", "observed_viseme": "", "geometric_flaw": "No visual data recorded.", "metrics": {}}
    
    if session_type == "articulation":
        if target_phoneme.lower() in ["r", "l"]:
            pucker_scores = [f.shapes.mouthPucker for f in frames]
            max_pucker = max(pucker_scores) if pucker_scores else 0
            passed = max_pucker < 0.5 
            return {
                "pass": passed,
                "target_viseme": "liquid_r_shape",
                "observed_viseme": "liquid_r_shape" if passed else "labial_w_shape",
                "geometric_flaw": "None" if passed else "Lips were too rounded (made a W shape).",
                "metrics": {"pucker": round(max_pucker, 3)}
            }
        
        if target_phoneme.lower() in ["s", "z"]:
            jaw = max((f.shapes.jawOpen for f in frames), default=0)
            passed = jaw < 0.20
            return {
                "pass": passed,
                "target_viseme": "dental_s_shape",
                "observed_viseme": "dental_s_shape" if passed else "interdental_th_shape",
                "geometric_flaw": "None" if passed else "Jaw was too open",
                "metrics": {"jaw": round(jaw, 3)}
            }
            
    if "pragmatics" in session_type:
        brow_scores = [f.shapes.browInnerUp for f in frames]
        peak_brow = max(brow_scores) if brow_scores else 0
        passed = peak_brow > 0.05 
        
        return {
            "pass": passed,
            "target_viseme": "engaged_brow",
            "observed_viseme": "engaged_brow" if passed else "flat_affect",
            "geometric_flaw": "None" if passed else "I didn't see your eyebrows move! Try to be more expressive.",
            "metrics": {"peak_brow": round(peak_brow, 3)}
        }
        
    return {"pass": False, "target_viseme": "", "observed_viseme": "", "geometric_flaw": "Unknown phoneme type", "metrics": {}}

@app.post("/analyze")
async def analyze_attempt(session: Session):
    child_name = session.name
    child_age = session.age
    wav = await convert_audio(session.audio_base64)
    azure_data = {}
    
    # We call Azure for both Articulation (accuracy) and Pragmatics (prosody/tone)
    if session.session_type in ["articulation", "pragmatics", "stuttering"]:
        azure_data = await asyncio.to_thread(call_azure_pronunciation, wav, session.target_word)
    
    vis_data = facial_evaluation(session.frames, session.session_type, session.target_phoneme)
    
    score = 0
    insight = ""
    what_they_said = azure_data.get("recognized_text", "")

    if "articulation" in session.session_type:
        if azure_data.get("words"):
            for p in azure_data["words"][0].get("phonemes", []):
                if p["text"].lower() == session.target_phoneme.lower():
                    score = p.get("accuracy_score", 0)
                    break
        insight = f"Phoneme accuracy: {score}"
        a_pass = score >= 80

    elif "pragmatics" in session.session_type:
        score = azure_data.get("prosody_score", 0)
        if len(what_they_said) > 2:
            insight = f"Prosody score: {score}"
            a_pass = True # They spoke, so we pass them, but we will critique the tone
        else:
            insight = "No speech detected"
            a_pass = False

    elif "stuttering" in session.session_type:
        # BETTER STUTTER DETECTION:
        # 1. Lower silence threshold to 1000ms (1 second block)
        # 2. Use regex to find repeated words (e.g., "The the the dog")
        text_repetition = re.search(r'\b(\w+)\s+\1\b', what_they_said, re.IGNORECASE)
        
        if session.audio_metrics and session.audio_metrics.longest_silence_ms > 1000:
            score = 30
            insight = "Block or long hesitation detected"
        elif text_repetition or "-" in what_they_said:
            score = 30
            insight = "Repetition or prolongation detected in speech"
        else:
            score = 95
            insight = "Smooth flow"
        a_pass = score >= 80

    v_pass = vis_data.get("pass", False)
    
    # ==========================================
    # DE-WEIGHTING VISUALS FIX:
    # Visuals from webcams are buggy. We will NOT fail the child if v_pass is False.
    # We rely primarily on a_pass for success. Visuals just guide the feedback.
    # ==========================================
    overall_pass = a_pass 
    status = "success" if overall_pass else "fail"
    visual_flaw = vis_data.get("geometric_flaw", "No visual data")
    
    # ---------------------------------------------------------
    # CONVERSATIONAL FEEDBACK PROMPT
    # ---------------------------------------------------------
    conversational_feedback = "Great job!"
    
    if session.session_mode == "practice":
        try:
            model = gemini.GenerativeModel('gemini-2.5-flash')
            
            if "pragmatics" in session.session_type:
                if overall_pass and v_pass and score > 80:
                    prompt = f"The {child_age}-year-old child successfully answered a social question, had great tone, and used facial expressions! Give a 1-sentence energetic congratulation."
                else:
                    # TONE ANALYSIS FIX: Fed the prosody score into the prompt
                    prompt = f"""You are a pediatric speech therapist. The {child_age}-year-old patient is practicing social pragmatics.
                    - What they said: "{what_they_said}"
                    - Prosody (Tone) Score: {score}/100. (If below 60, they sounded monotone. If above 80, they sounded lively).
                    - Visual Diagnostic: {visual_flaw}
                    
                    Provide 3 sentences of empathetic feedback:
                    1. Acknowledge what they said.
                    2. Analyze their tone of voice based on the Prosody Score (e.g., "I loved your words, but your voice sounded a little sleepy!").
                    3. If the Visual Diagnostic mentions flat eyebrows, gently remind them to use facial expressions.
                    DO NOT sound out words or mention 'R' sounds. Do not use markdown."""
            
            elif "stuttering" in session.session_type:
                if overall_pass:
                    prompt = f"The {child_age}-year-old child read the passage smoothly! Give a 1-sentence energetic congratulation."
                else:
                    prompt = f"""You are a pediatric speech therapist. The {child_age}-year-old patient is practicing fluency (stuttering).
                    - What we heard: "{what_they_said}"
                    - Diagnostic: {insight}
                    
                    Provide 2-3 sentences of empathetic feedback. Acknowledge that they got a little bumpy or stuck, and remind them to take a deep breath and use "smooth, easy speech". Do not use markdown."""
            
            else: # Articulation
                if overall_pass:
                    prompt = f"The {child_age}-year-old child successfully said '{session.target_word}'. Give them a quick 1-sentence energetic congratulation! Do not use markdown."
                else:
                    prompt = f"""You an expert pediatric speech therapist. The patient is {child_age} years old. 
                    They are practicing an articulation exercise targeting the word '{session.target_word}'.
                    - Target Phoneme: /{session.target_phoneme}/
                    - What we heard: "{what_they_said}"
                    - Visual Diagnostic: {visual_flaw}

                    Provide 3 sentences of conversational feedback:
                    1. Acknowledge what they sounded like.
                    2. Sound out the target word phonetically with dashes (e.g., "RRRR-abbit").
                    3. Give ONE specific, anatomical piece of advice based on the Visual Diagnostic to fix their mouth shape.
                    Do not use bullet points or markdown."""
            
            response = await asyncio.to_thread(model.generate_content, prompt)
            conversational_feedback = response.text.strip().replace("*", "")
        except Exception as e:
            print(f"Feedback generation failed: {e}")
            conversational_feedback = "Good effort! Let's try that one more time."

    # ------------------------------------------------------------------
    # RAILTRACKS: Generate the NEXT exercise 
    # ------------------------------------------------------------------
    next_exercise_data = None
    if session.session_mode == "practice":
        backend_analysis_payload = {
            "status": status,
            "session_type": session.session_type,
            "target_word": session.target_word, 
            "animation_triggers": {
                "expected_phoneme": session.target_phoneme, 
                "observed_phoneme": "" if a_pass else "error",
                "what_child_said": what_they_said
            },
            "diagnostic_data": {
                "visual_analysis": {
                    "geometric_flaw": visual_flaw,
                }
            },
            "feedback": conversational_feedback # Sending the generated feedback to the Railtracks agent!
        }
        
        user_profile = {
            "name": child_name,
            "age": child_age,
            "backend_analysis": backend_analysis_payload
        }
        
        try:
            next_exercise_data = await generate_exercise_from_analysis(
                user_profile=user_profile, 
                session_type=session.session_type
            )
        except Exception as e:
            print(f"Railtracks generation failed: {e}")

    return {
        "status": status,
        "session_mode": session.session_mode,
        "session_type": session.session_type,
        "conversational_feedback": conversational_feedback,
        "next_turn": next_exercise_data, 
        "animation_triggers": {
            "target_word": session.target_word,
            "what_child_said": what_they_said, 
            "target_viseme": vis_data.get("target_viseme", ""),
            "observed_viseme": vis_data.get("observed_viseme", ""),
            "expected_phoneme": session.target_phoneme,
            "observed_phoneme": "" if a_pass else "error"
        },
        "diagnostic_data": {
            "audio_analysis": {
                "prosody_score": score,
                "audio_insight": insight,
                "pass": a_pass
            },
            "visual_analysis": {
                "pass": v_pass,
                "geometric_flaw": visual_flaw,
                "metrics": vis_data.get("metrics", {})
            }
        }
    }