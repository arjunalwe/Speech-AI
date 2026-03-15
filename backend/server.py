import os
import base64
import asyncio
import tempfile
import subprocess
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
            # Grab all pucker scores, default to 0 if empty
            pucker_scores = [f.shapes.mouthPucker for f in frames]
            max_pucker = max(pucker_scores) if pucker_scores else 0
            
            # Grab all smile scores
            smile_scores = [(f.shapes.mouthSmileLeft + f.shapes.mouthSmileRight) / 2 for f in frames]
            max_smile = max(smile_scores) if smile_scores else 0
            
            # THE DEMO FIX: Normal speech pucker is 0.1 - 0.3. 
            # We ONLY fail them if they make an extreme duck face (> 0.5).
            passed = max_pucker < 0.5 
            
            return {
                "pass": passed,
                "target_viseme": "liquid_r_shape",
                "observed_viseme": "liquid_r_shape" if passed else "labial_w_shape",
                "geometric_flaw": "None" if passed else "Lips were too rounded (made a W shape). Relax your lips!",
                "metrics": {"pucker": round(max_pucker, 3), "smile": round(max_smile, 3)}
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
            
    if session_type == "pragmatics":
        base = sum(f.shapes.browInnerUp for f in frames[:5]) / 5 if len(frames) >= 5 else 0
        peak = max((f.shapes.browInnerUp for f in frames), default=0)
        passed = peak >= (base + 0.15)
        return {
            "pass": passed,
            "target_viseme": "engaged_brow",
            "observed_viseme": "engaged_brow" if passed else "flat_affect",
            "geometric_flaw": "None" if passed else "No brow raise detected",
            "metrics": {"peak": round(peak, 3)}
        }
        
    return {"pass": False, "target_viseme": "", "observed_viseme": "", "geometric_flaw": "Unknown phoneme type", "metrics": {}}

@app.post("/analyze")
async def analyze_attempt(session: Session):
    # USE THE REAL DATA FROM REACT!
    child_name = session.name
    child_age = session.age
    wav = await convert_audio(session.audio_base64)
    azure_data = {}
    if session.session_type in ["articulation", "pragmatics"]:
        azure_data = await asyncio.to_thread(call_azure_pronunciation, wav, session.target_word)
    
    vis_data = facial_evaluation(session.frames, session.session_type, session.target_phoneme)
    
    score = 0
    insight = ""
    what_they_said = azure_data.get("recognized_text", "")

    if session.session_type == "articulation":
        if azure_data.get("words"):
            for p in azure_data["words"][0].get("phonemes", []):
                if p["text"].lower() == session.target_phoneme.lower():
                    score = p.get("accuracy_score", 0)
                    break
        insight = f"Phoneme accuracy: {score}"
    elif session.session_type == "pragmatics":
        score = azure_data.get("prosody_score", 0)
        insight = f"Prosody score: {score}"
    elif session.session_type == "stuttering":
        if session.audio_metrics:
            if session.audio_metrics.max_volume_spike > 0.8 or session.audio_metrics.longest_silence_ms > 2000:
                score = 30
                insight = "Block or hesitation detected"
            else:
                score = 95
                insight = "Smooth flow"

    a_pass = score >= 80
    v_pass = vis_data.get("pass", False)
    
    if session.session_type == "pragmatics":
        overall_pass = a_pass and v_pass
    else:
        overall_pass = a_pass
        
    status = "success" if overall_pass else "fail"
    visual_flaw = vis_data.get("geometric_flaw", "No visual data")
    
    # ------------------------------------------------------------------
    # RESTORED: Your exact code to generate the conversational feedback
    # ------------------------------------------------------------------
    conversational_feedback = "Great job!"
    if session.session_mode == "practice":
        if overall_pass:
            prompt = f"The {child_age}-year-old child successfully said '{session.target_word}'. Give them a quick 1-sentence energetic congratulation! Do not use markdown."
        else:
            prompt = f"""You an expert, highly encouraging pediatric speech therapist. 
            The patient is {child_age} years old. Adjust your vocabulary and tone perfectly for this age.
            They are practicing a '{session.session_type}' exercise targeting the word '{session.target_word}'.

            CLINICAL DATA:
            - Target Phoneme: /{session.target_phoneme}/
            - What the AI heard them say: "{what_they_said}"
            - Audio Diagnostic: {insight}
            - Visual Diagnostic (Webcam): {visual_flaw}

            YOUR TASK:
            The child did not pass. Provide 3-4 sentences of conversational feedback doing exactly this:
            1. Gently acknowledge what they actually sounded like based on the data.
            2. Sound out the target word phonetically with dashes (e.g., "RRRR-abbit") so they hear how to break it down.
            3. Use the Visual Diagnostic to give ONE specific, anatomical piece of advice on how to move their lips, jaw, or tongue to fix the specific error they made. 
            Make it empathetic and engaging! Do not use bullet points or markdown."""
        
        try:
            model = gemini.GenerativeModel('gemini-2.5-flash')
            response = await asyncio.to_thread(model.generate_content, prompt)
            conversational_feedback = response.text.strip()
        except Exception as e:
            print(f"RATE LIMIT OR SDK CRASH: {e}")
            import random
            fallbacks = [
                f"I heard you say '{what_they_said}'. Next time, make sure to keep your lips pulled back!",
                f"You're getting closer! Remember our trick: don't let your lips make an 'O' shape.",
                f"Good try! I saw your mouth move a little too much. Keep those lips spread wide like a smile!"
            ]
            conversational_feedback = random.choice(fallbacks) if not overall_pass else "Awesome job, your form was perfect!"
    # ------------------------------------------------------------------
    # RAILTRACKS: Generate the NEXT exercise 
    # ------------------------------------------------------------------
    # RAILTRACKS: Generate the NEXT exercise 
    next_exercise_data = None
    if session.session_mode == "practice":
        backend_analysis_payload = {
            "status": status,
            "session_type": session.session_type,
            "target_word": session.target_word, # <--- ADD THIS SO THE AI KNOWS WHAT THEY SAID
            "animation_triggers": {
                "expected_phoneme": session.target_phoneme, 
                "observed_phoneme": "" if a_pass else "error"
            },
            "diagnostic_data": {
                "visual_analysis": {
                    "geometric_flaw": visual_flaw,
                }
            }
        }
        
        user_profile = {
            "name": child_name,
            "age": child_age,
            "backend_analysis": backend_analysis_payload
        }
        
        next_exercise_data = await generate_exercise_from_analysis(
            user_profile=user_profile, 
            session_type=session.session_type
        )

    return {
        "status": status,
        "session_mode": session.session_mode,
        "session_type": session.session_type,
        "conversational_feedback": conversational_feedback, # Exposing this to App.jsx!
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