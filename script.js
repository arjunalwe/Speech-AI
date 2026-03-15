import {
FaceLandmarker,
FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32";

const startButton=document.getElementById("startButton");
const stopButton=document.getElementById("stopButton");
const statusText=document.getElementById("status");
const video=document.getElementById("video");
const audioPlayback=document.getElementById("audioPlayback");
const jsonOutput=document.getElementById("jsonOutput");

let mediaStream=null;
let mediaRecorder=null;
let audioChunks=[];

let faceLandmarker=null;
let visualTelemetry=[];
let recordingStartTime=0;
let animationFrameId=null;
let isTracking=false;

const BACKEND_URL="http://localhost:8000/analyze";

async function createFaceLandmarker(){

if(faceLandmarker) return faceLandmarker;

const vision=await FilesetResolver.forVisionTasks(
"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm"
);

faceLandmarker=await FaceLandmarker.createFromOptions(vision,{
baseOptions:{
modelAssetPath:
"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
},
runningMode:"VIDEO",
numFaces:1,
outputFaceBlendshapes:true,
outputFacialTransformationMatrixes:true
});

return faceLandmarker;
}

function blobToBase64(blob){

return new Promise((resolve,reject)=>{

const reader=new FileReader();

reader.onloadend=()=>{
const result=reader.result;
resolve(result.split(",")[1]);
};

reader.onerror=reject;

reader.readAsDataURL(blob);

});
}

function getBlendshapeScore(result,name){

if(!result.faceBlendshapes||result.faceBlendshapes.length===0){
return null;
}

const categories=result.faceBlendshapes[0].categories;
const match=categories.find(c=>c.categoryName===name);

return match?Number(match.score.toFixed(4)):null;

}

function estimateHeadPose(result){

let pitch=null;
let yaw=null;
let roll=null;

const matrices=result.facialTransformationMatrixes;

if(!matrices||matrices.length===0){
return{pitch,yaw,roll};
}

const m=matrices[0].data;

if(!m||m.length<16){
return{pitch,yaw,roll};
}

const r00=m[0];
const r10=m[4];
const r21=m[9];
const r22=m[10];
const r20=m[8];

pitch=Math.atan2(-r21,r22)*(180/Math.PI);
roll=Math.atan2(r10,r00)*(180/Math.PI);
yaw=Math.atan2(-r20,Math.sqrt(r21*r21+r22*r22))*(180/Math.PI);

return{
pitch:Number(pitch.toFixed(2)),
yaw:Number(yaw.toFixed(2)),
roll:Number(roll.toFixed(2))
};
}

function buildTelemetryFrame(result,elapsedMs){

const pose=estimateHeadPose(result);

return{
time_ms:Math.round(elapsedMs),
shapes:{
mouthClose:getBlendshapeScore(result,"mouthClose"),
mouthPucker:getBlendshapeScore(result,"mouthPucker"),
mouthFunnel:getBlendshapeScore(result,"mouthFunnel"),
mouthRollLower:getBlendshapeScore(result,"mouthRollLower"),
mouthUpperUp:getBlendshapeScore(result,"mouthUpperUp"),
jawOpen:getBlendshapeScore(result,"jawOpen"),
browInnerUp:getBlendshapeScore(result,"browInnerUp")
},
pose:{
pitch:pose.pitch,
yaw:pose.yaw,
roll:pose.roll
}
};
}

function trackFaceLoop(){

if(!isTracking||!faceLandmarker) return;

const now=performance.now();
const elapsedMs=now-recordingStartTime;

if(video.readyState>=2){

const result=faceLandmarker.detectForVideo(video,now);

if(result.faceLandmarks&&result.faceLandmarks.length>0){

const frame=buildTelemetryFrame(result,elapsedMs);
visualTelemetry.push(frame);

}
}

animationFrameId=requestAnimationFrame(trackFaceLoop);

}

function getCurrentTimestampUtc(){
return new Date().toISOString();
}

async function startRecording(){

try{

statusText.textContent="Loading MediaPipe...";
await createFaceLandmarker();

statusText.textContent="Requesting camera and microphone...";

mediaStream=await navigator.mediaDevices.getUserMedia({
video:true,
audio:true
});

video.srcObject=mediaStream;
await video.play();

audioChunks=[];
visualTelemetry=[];

mediaRecorder=new MediaRecorder(mediaStream);

mediaRecorder.ondataavailable=e=>{
if(e.data.size>0) audioChunks.push(e.data);
};

mediaRecorder.onstop=async()=>{

const audioBlob=new Blob(audioChunks,{type:"audio/webm"});

audioPlayback.src=URL.createObjectURL(audioBlob);

const audioBase64=await blobToBase64(audioBlob);

const payload={
metadata:{
target_goal:"question_pragmatics",
timestamp_utc:getCurrentTimestampUtc()
},
audio_base64:audioBase64,
visual_telemetry:visualTelemetry
};

jsonOutput.textContent=JSON.stringify(payload,null,2);

try{

const response=await fetch(BACKEND_URL,{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify(payload)
});

const data=await response.json();

console.log("Backend response:",data);

statusText.textContent="Recording stopped. Payload sent to backend.";

}catch(err){

console.error(err);

statusText.textContent="Recording stopped. Backend request failed.";

}

};

mediaRecorder.start();

recordingStartTime=performance.now();
isTracking=true;

trackFaceLoop();

startButton.disabled=true;
stopButton.disabled=false;

statusText.textContent="Recording + tracking...";

}catch(error){

console.error(error);

statusText.textContent="Failed to start recording.";

}

}

function stopRecording(){

isTracking=false;

if(animationFrameId){
cancelAnimationFrame(animationFrameId);
}

if(mediaRecorder&&mediaRecorder.state!=="inactive"){
mediaRecorder.stop();
}

if(mediaStream){
mediaStream.getTracks().forEach(track=>track.stop());
video.srcObject=null;
}

startButton.disabled=false;
stopButton.disabled=true;

}

startButton.addEventListener("click",startRecording);
stopButton.addEventListener("click",stopRecording);