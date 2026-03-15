import base64
from fastapi import FastAPI, HTTPException
import imageio_ffmpeg
from pydantic import BaseModel, Field
from typing import List
import os
import tempfile
import subprocess
from fastapi.middleware.cors import CORSMiddleware
import azure.cognitiveservices.speech as speechsdk
import json

app = FastAPI(title="Speech Therapy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class FacialShapes(BaseModel):
    mouth_close: float = Field(alias="mouthClose")
    mouth_pucker: float = Field(alias="mouthPucker")
    mouth_funnel: float = Field(alias="mouthFunnel")
    mouth_roll_lower: float = Field(alias="mouthRollLower")
    mouth_upper_up: float = Field(alias="mouthUpperUp")
    jaw_open: float = Field(alias="jawOpen")
    brow_inner_up: float = Field(alias="browInnerUp")

class HeadPose(BaseModel):
    pitch: float
    yaw: float
    roll: float

class Frame(BaseModel):
    time_ms: int
    shapes: FacialShapes
    pose: HeadPose
    
class Session(BaseModel):
    session_type: str
    audio_base64: str
    frames: List[Frame]
    
    
async def convert_audio(base64_str: str) -> bytes:
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    base64_str += "=" * ((4 - len(base64_str) % 4) % 4)
    try:
        audio_bytes = base64.b64decode(base64_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 Decode Error: {str(e)}")
    
    if len(audio_bytes) < 100:
        return b"fake_audio_bytes_for_postman_testing"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_in:
        temp_in.write(audio_bytes)
        temp_in_path = temp_in.name
        
    temp_out_path = temp_in_path + "_out.wav"

    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        
        command = [
            ffmpeg_path,
            "-y",                  
            "-i", temp_in_path,    
            "-ar", "16000",       
            "-ac", "1",      
            "-f", "wav", 
            temp_out_path
        ]
        
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        with open(temp_out_path, "rb") as f:
            wav_bytes = f.read()
            
        return wav_bytes
        
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="FFmpeg core failed to process the audio.")
    finally:
        if os.path.exists(temp_in_path):
            os.remove(temp_in_path)
        if os.path.exists(temp_out_path):
            os.remove(temp_out_path)
    
    
def facial_evaluation(frames: List[Frame], session_type: str) -> dict[str: str]:
    if len(frames) < 5:
        return {"pass": False, "reason": "not enough data"}
    
    if session_type == "bilabial":
        baseline_closed = sum(f.shapes.mouth_close for f in frames[:5]) / 5
        
        max_vel = 0.0
        hold_duration = 0
        is_holding = False
        hold_start = 0
        
        for i in range(1, len(frames)):
            curr = frames[i]
            prev = frames[i - 1]
            
            dt = (curr.time_ms - prev.time_ms) / 1000.0
            if dt == 0:
                continue
            
            dy = curr.shapes.mouth_close - prev.shapes.mouth_close
            vel = dy/dt
            
            if vel > max_vel:
                max_vel = vel
            
            if curr.shapes.mouth_close > (baseline_closed + 0.4):
                if not is_holding:
                    is_holding = True
                    hold_start = curr.time_ms
                    
                else:
                    hold_duration = curr.time_ms - hold_start
                    
            else:
                is_holding = False
    
        VELOCITY_THRESHOLD = 2.0
        HOLD_THRESHOLD_MS = 40
        
        passed_velocity = max_vel > VELOCITY_THRESHOLD
        passed_hold = hold_duration > HOLD_THRESHOLD_MS
        
        return {
                "pass": passed_velocity and passed_hold,
                "metrics": {
                    "resting_baseline": round(baseline_closed, 3),
                    "peak_velocity_snap": round(max_vel, 2),
                    "hold_duration_ms": hold_duration
                },
                "insight": (
                    "Perfect explosive articulation." if (passed_velocity and passed_hold) 
                    else "Articulation failed: Lip closure was too sluggish." if not passed_velocity 
                    else "Articulation failed: Lips did not maintain closure long enough."
                )
            }
        
    elif session_type == "question_pragmatics":
        baseline_brow = sum(f.shapes.brow_inner_up for f in frames[:5]) / 5
        
        is_raised = False
        raise_duration_ms = 0
        raise_start_time = 0
        peak_brow = 0.0

        for current in frames:
            if current.shapes.brow_inner_up > peak_brow:
                peak_brow = current.shapes.brow_inner_up
                
            if current.shapes.brow_inner_up > (baseline_brow + 0.20):
                if not is_raised:
                    is_raised = True
                    raise_start_time = current.time_ms
                else:
                    raise_duration_ms = current.time_ms - raise_start_time
            else:
                is_raised = False 

        HOLD_THRESHOLD_MS = 200
        passed_hold = raise_duration_ms > HOLD_THRESHOLD_MS
        
        return {
            "pass": passed_hold,
            "metrics": {
                "resting_brow": round(baseline_brow, 3),
                "peak_brow_raise": round(peak_brow, 3),
                "raise_duration_ms": raise_duration_ms
            },
            "insight": (
                "Excellent non-verbal question marker." if passed_hold 
                else "Eyebrows were raised, but it was a twitch. Needs to be held longer to register as a social cue." if peak_brow > (baseline_brow + 0.2)
                else "Flat affect detected. No significant eyebrow movement."
            )
        }

    return {"pass": True, "reason": "Goal analyzed"}


@app.post("/analyze")
async def analyze_attempt(session: Session):
    wav_bytes = await convert_audio(session.audio_base64)
    result = facial_evaluation(session.frames, session.session_type)
    
    return {
        "status": "success",
        "session_context": {
            "session_type": session.session_type,
            "audio_length_seconds": round(len(wav_bytes) / 32000, 2)
        },
        "multimodal_fusion_conclusion": result
    }