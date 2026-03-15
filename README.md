# Speech Therapy

## Inspiration

Speech disorders such as stuttering and articulation difficulties can make everyday communication challenging, especially for children and people with developmental language disorders. Traditional speech therapy often requires frequent in-person sessions with trained specialists, which can be expensive and inaccessible for many families.

We were inspired to explore whether AI and computer vision could make speech therapy more accessible. Instead of analyzing only what someone says, our system also evaluates how speech is physically produced by tracking mouth shape, jaw movement, and facial articulation through a webcam.

Our goal was to create a platform that combines speech recognition, facial analysis, and AI feedback to help children practice speech in a more engaging and supportive environment.

## What it does

Lumi is a multimodal AI speech therapy platform that analyzes both how someone speaks and how their mouth moves while speaking.

The system guides a child through short speech exercises and records their attempt using their microphone and webcam. While the child speaks, the system tracks facial movements such as lip shape, jaw opening, and eyebrow motion to capture articulation patterns.

The system then analyzes:
- Audio pronunciation accuracy
- Facial articulation patterns
- Speech rhythm and pauses

These signals are combined to determine whether the target sound or speech behavior was produced correctly.

For example, if a child tries to say the word “rabbit,” the system can detect whether the phoneme /r/ was pronounced correctly and whether the mouth shape matched the correct articulation pattern.

Based on this analysis, Lumi generates personalized feedback that helps the child understand what they did correctly and how to improve their pronunciation.

## How we built it

**Frontend**

The interface was built using React and Vite, with Tailwind CSS for styling. During practice sessions, the browser accesses the user’s webcam and microphone to record speech attempts.

**Facial tracking**

We integrated MediaPipe’s Face Landmarker to track facial landmarks and blendshape signals in real time. This allows the system to capture articulation cues such as jaw opening, lip rounding, and eyebrow movement, producing a timeline of facial telemetry while the user speaks.

**Audio capture**

Audio is recorded in the browser using the MediaRecorder API. The recording is converted to Base64 and packaged with the facial telemetry into a structured JSON payload representing the speech attempt.

**Backend analysis**

The backend, built with Python and FastAPI, receives the synchronized audio and visual data. It converts the audio using FFmpeg, sends it to Azure’s Pronunciation Assessment API for phoneme analysis, and evaluates facial articulation patterns using rule-based geometric checks.

**AI feedback**

Finally, Google Gemini generates clear, supportive feedback explaining how the user can improve their pronunciation.

## Challenges we ran into

One challenge was getting the text-to-speech system to generate audio with the appropriate tone. Since the feedback is meant to guide users in a supportive way, we needed the generated voice to sound natural and clear rather than robotic, which required some experimentation with the API.

Another challenge was ensuring reliable data transfer between the frontend and backend. Our platform sends synchronized audio and facial telemetry data from the browser as a structured JSON payload, so we had to carefully design and debug the data format to make sure the backend could process it correctly.

We also had to narrow the scope of the project to something achievable within the hackathon timeframe. Initially we planned to incorporate hardware such as a Raspberry Pi, but after feedback from mentors we decided to focus on a web-based platform so we could prioritize building the core functionality.

## Accomplishments that we're proud of

We built a multimodal speech analysis system that evaluates both how a user sounds and how their mouth moves while speaking. By combining Microsoft Azure’s phonetic audio analysis with facial landmark data from Google MediaPipe Blendshapes, we developed a scoring system that verifies whether a child’s mouth is physically positioned correctly for specific sounds.

We also developed a React-based frontend capable of managing high-frequency data streams, capturing 60fps facial telemetry and real-time audio signals simultaneously while maintaining a smooth user interface.

Finally, we implemented a multi-agent AI system using Railtracks that dynamically routes users to specialized expert agents for articulation, pragmatics, or fluency-related speech disorders. These agents use age-adaptive heuristics to adjust linguistic complexity and therapeutic tone while focusing on the user’s developmental stage and speech challenges.

## What we learned

One of our biggest takeaways from this project was the importance of communication. Since the system involved multiple components across the frontend, backend, and AI services, clear coordination within the team was essential.

We also learned how combining LLMs with physical data and traditional machine learning processing can help create more robust AI systems. By grounding AI responses in structured audio and visual data, we were able to reduce hallucinations while still benefiting from the flexibility of large language models.

## What's next for Lumi

There are many ways this project could be expanded.

Future improvements could include:
- expanding support for more phonemes and languages
- adding animated mouth visualizations that demonstrate correct articulation
- building adaptive practice plans based on a child’s progress
- improving real-time feedback during speech attempts

Our long-term vision is to create an accessible AI-powered speech practice platform that helps children build confidence and communication skills through engaging, personalized training.
