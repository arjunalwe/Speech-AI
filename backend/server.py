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

load_dotenv()
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

gemini.configure(api_key=GEMINI_API_KEY)

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
        return {"pass": False, "target_viseme": "", "observed_viseme": "", "geometric_flaw": "No data", "metrics": {}}
    if session_type == "articulation":
        if target_phoneme in ["r", "l"]:
            pucker = max(f.shapes.mouthPucker for f in frames)
            smile = max((f.shapes.mouthSmileLeft + f.shapes.mouthSmileRight) / 2 for f in frames)
            passed = smile > (pucker + 0.1)
            return {
                "pass": passed,
                "target_viseme": "liquid_r_shape",
                "observed_viseme": "liquid_r_shape" if passed else "labial_w_shape",
                "geometric_flaw": "None" if passed else "Lips too rounded",
                "metrics": {"pucker": round(pucker, 3), "smile": round(smile, 3)}
            }
        if target_phoneme in ["s", "z"]:
            jaw = max(f.shapes.jawOpen for f in frames)
            passed = jaw < 0.20
            return {
                "pass": passed,
                "target_viseme": "dental_s_shape",
                "observed_viseme": "dental_s_shape" if passed else "interdental_th_shape",
                "geometric_flaw": "None" if passed else "Jaw too open",
                "metrics": {"jaw": round(jaw, 3)}
            }
    if session_type == "pragmatics":
        base = sum(f.shapes.browInnerUp for f in frames[:5]) / 5 if len(frames) >= 5 else 0
        peak = max(f.shapes.browInnerUp for f in frames)
        passed = peak >= (base + 0.20)
        return {
            "pass": passed,
            "target_viseme": "engaged_brow",
            "observed_viseme": "engaged_brow" if passed else "flat_affect",
            "geometric_flaw": "None" if passed else "No brow raise",
            "metrics": {"peak": round(peak, 3)}
        }
    if session_type == "stuttering":
        var = max(f.shapes.jawOpen for f in frames) - min(f.shapes.jawOpen for f in frames)
        passed = var < 0.4
        return {
            "pass": passed,
            "target_viseme": "smooth_phonation",
            "observed_viseme": "smooth_phonation" if passed else "secondary_tremor",
            "geometric_flaw": "None" if passed else "Jaw tremor detected",
            "metrics": {"var": round(var, 3)}
        }
    return {"pass": False, "target_viseme": "", "observed_viseme": "", "geometric_flaw": "Unknown type", "metrics": {}}

async def generate_response(prompt: str) -> str:
    try:
        model = gemini.GenerativeModel('gemini-1.5-pro')
        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text.replace("*", "") 
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "You're doing great! Keep practicing that sound, paying close attention to your mouth shape!"

@app.post("/analyze")
async def analyze_attempt(session: Session):
    
    user_data = MOCK_DB.get(session.user_id, {"age": 5}) 
    child_age = user_data["age"]

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
    
    dialogue = None
    if session.session_mode == "practice":
        visual_flaw = vis_data.get("geometric_flaw", "No visual data")
        
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
2. Sound out the target word phonetically with dashes (e.g., "RRRR-abbit" or "ssss-nake") so they hear how to break it down.
3. Use the Visual Diagnostic to give ONE specific, anatomical piece of advice on how to move their lips, jaw, or tongue to fix the specific error they made. 
Make it empathetic and engaging! Do not use bullet points or markdown."""
        
        dialogue = await generate_response(prompt)

    return {
        "status": status,
        "session_mode": session.session_mode,
        "session_type": session.session_type,
        "feedback": dialogue,
        "animation_triggers": {
            "target_word": session.target_word,
            "failed_word": "" if status == "success" else session.target_word,
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
                "geometric_flaw": vis_data.get("geometric_flaw", ""),
                "metrics": vis_data.get("metrics", {})
            }
        }
    }