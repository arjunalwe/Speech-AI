import {
    FaceLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32";

const video = document.getElementById("video-feed");
const statusTextAssess = document.getElementById("assessment-status");
const statusTextPractice = document.getElementById("practice-status");
const feedbackBubble = document.getElementById("lumi-speech-bubble");
const visemeAvatar = document.getElementById("viseme-avatar");

const currentProfile = { userId: "user_123", focusArea: "articulation" };
let mediaStream = null;
let mediaRecorder = null;
let audioChunks = [];
let faceLandmarker = null;
let visualTelemetry = [];
let recordingStartTime = 0;
let animationFrameId = null;
let isTracking = false;

let audioContext, analyser, microphone, audioTrackingId;
let maxVolumeSpike = 0, currentSilenceStart = 0, longestSilenceMs = 0;
const SILENCE_THRESHOLD = 0.05;

const BACKEND_URL = "http://localhost:8000/analyze";

function switchView(viewId) {
    document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));
    document.getElementById(viewId).classList.add('active');
}

document.getElementById("btn-go-register").addEventListener("click", () => switchView("view-register"));
document.getElementById("btn-go-quiz").addEventListener("click", () => switchView("view-quiz"));
document.getElementById("btn-back-register").addEventListener("click", () => switchView("view-register"));
document.getElementById("btn-go-assessment").addEventListener("click", () => switchView("view-assessment"));

document.getElementById("btn-save-quiz").addEventListener("click", () => {
    const selected = document.querySelector('input[name="disorder"]:checked');
    if (selected) {
        currentProfile.focusArea = selected.value;
        document.getElementById("focus-area").innerText = selected.value;
        switchView("view-dashboard");
    } else {
        alert("Please select an area of focus.");
    }
});

function launchGame(type) {
    currentProfile.focusArea = type;
    document.getElementById("practice-title").innerText = `Playing: ${type}`;
    switchView("view-practice");
}

document.getElementById("btn-play-articulation").addEventListener("click", () => launchGame("articulation"));
document.getElementById("btn-play-stuttering").addEventListener("click", () => launchGame("stuttering"));
document.getElementById("btn-end-session").addEventListener("click", () => switchView("view-dashboard"));

async function createFaceLandmarker() {
    if (faceLandmarker) return faceLandmarker;
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm");
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" },
        runningMode: "VIDEO", numFaces: 1, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true
    });
    return faceLandmarker;
}

function getBlendshapeScore(result, name) {
    if (!result.faceBlendshapes || result.faceBlendshapes.length === 0) return 0.0;
    const match = result.faceBlendshapes[0].categories.find(c => c.categoryName === name);
    return match ? Number(match.score.toFixed(4)) : 0.0;
}

function trackFaceLoop() {
    if (!isTracking || !faceLandmarker) return;
    const now = performance.now();
    if (video.readyState >= 2) {
        const result = faceLandmarker.detectForVideo(video, now);
        if (result.faceLandmarks?.length > 0) {
            visualTelemetry.push({
                time_ms: Math.round(now - recordingStartTime),
                shapes: {
                    mouthClose: getBlendshapeScore(result, "mouthClose"),
                    mouthPucker: getBlendshapeScore(result, "mouthPucker"),
                    mouthSmileLeft: getBlendshapeScore(result, "mouthSmileLeft"),
                    mouthSmileRight: getBlendshapeScore(result, "mouthSmileRight"),
                    jawOpen: getBlendshapeScore(result, "jawOpen"),
                    browInnerUp: getBlendshapeScore(result, "browInnerUp")
                },
                pose: { pitch: 0, yaw: 0, roll: 0 }
            });
        }
    }
    animationFrameId = requestAnimationFrame(trackFaceLoop);
}

function trackAudioMetrics() {
    if (!isTracking) return;
    const dataArray = new Float32Array(analyser.frequencyBinCount);
    analyser.getFloatTimeDomainData(dataArray);
    let maxInFrame = 0;
    for (let i = 0; i < dataArray.length; i++) {
        let abs = Math.abs(dataArray[i]);
        if (abs > maxInFrame) maxInFrame = abs;
    }
    if (maxInFrame > maxVolumeSpike) maxVolumeSpike = maxInFrame;
    let now = performance.now();
    if (maxInFrame < SILENCE_THRESHOLD) {
        longestSilenceMs = Math.max(longestSilenceMs, now - currentSilenceStart);
    } else {
        currentSilenceStart = now;
    }
    audioTrackingId = requestAnimationFrame(trackAudioMetrics);
}

async function startSession(mode) {
    try {
        const statusLabel = mode === "assessment" ? statusTextAssess : statusTextPractice;
        statusLabel.textContent = "Initializing...";
        
        await createFaceLandmarker();
        mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        video.srcObject = mediaStream;
        await video.play();

        visualTelemetry = []; audioChunks = [];
        maxVolumeSpike = 0; longestSilenceMs = 0;
        currentSilenceStart = performance.now();

        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        microphone = audioContext.createMediaStreamSource(mediaStream);
        microphone.connect(analyser);

        mediaRecorder = new MediaRecorder(mediaStream);
        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
        mediaRecorder.onstop = async () => {
            const audioBase64 = await new Promise(resolve => {
                const reader = new FileReader();
                reader.onloadend = () => resolve(reader.result.split(",")[1]);
                reader.readAsDataURL(new Blob(audioChunks, { type: "audio/webm" }));
            });

            const payload = {
                user_id: currentProfile.userId,
                session_mode: mode,
                session_type: currentProfile.focusArea,
                target_word: document.getElementById(mode === "assessment" ? "assessment-text" : "practice-prompt").innerText,
                target_phoneme: "r", 
                audio_base64: audioBase64,
                frames: visualTelemetry,
                audio_metrics: { max_volume_spike: maxVolumeSpike, longest_silence_ms: Math.round(longestSilenceMs) }
            };

            const response = await fetch(BACKEND_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            
            if (mode === "practice") {
                feedbackBubble.innerText = data.nova_feedback || "Great try!";
                visemeAvatar.innerText = data.status === "success" ? "🌟" : "👄";
            } else {
                currentProfile.focusArea = data.session_type;
                document.getElementById("focus-area").innerText = data.session_type;
                switchView("view-dashboard");
            }
        };

        mediaRecorder.start();
        recordingStartTime = performance.now();
        isTracking = true;
        trackFaceLoop();
        trackAudioMetrics();

        document.getElementById(`btn-start-${mode}`).disabled = true;
        document.getElementById(`btn-stop-${mode}`).disabled = false;
        statusLabel.textContent = "Recording...";
    } catch (err) {
        console.error(err);
    }
}

function stopSession(mode) {
    isTracking = false;
    cancelAnimationFrame(animationFrameId);
    cancelAnimationFrame(audioTrackingId);
    if (audioContext) audioContext.close();
    if (mediaRecorder) mediaRecorder.stop();
    if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    
    document.getElementById(`btn-start-${mode}`).disabled = false;
    document.getElementById(`btn-stop-${mode}`).disabled = true;
}

document.getElementById("btn-start-assessment").addEventListener("click", () => startSession("assessment"));
document.getElementById("btn-stop-assessment").addEventListener("click", () => stopSession("assessment"));
document.getElementById("btn-start-practice").addEventListener("click", () => startSession("practice"));
document.getElementById("btn-stop-practice").addEventListener("click", () => stopSession("practice"));