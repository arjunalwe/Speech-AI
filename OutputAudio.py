import json
import os
from elevenlabs.client import ElevenLabs

# Put your ElevenLabs API key here
client = ElevenLabs(api_key="fc9e9dba45a1bd4a027239fb9def8a779e24cf945238919ec8a16f6f750c6fef")


def generate_audio_from_feedback(json_input):

    # Convert JSON string to dictionary if needed
    if isinstance(json_input, str):
        if not json_input.strip():
            raise ValueError("JSON input is empty. Check test.json")
        data = json.loads(json_input)
    else:
        data = json_input

    # Extract values
    feedback_text = data["feedback_payload"]["feedback"]
    session_id = data["session_id"]

    print("Generating audio for:", feedback_text)

    # Generate audio from ElevenLabs
    audio = client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB",
        model_id="eleven_multilingual_v2",
        text=feedback_text
    )

    # Output file
    output_filename = f"feedback_{session_id}.mp3"

    # Save audio file
    with open(output_filename, "wb") as f:
        if isinstance(audio, bytes):
            f.write(audio)
        else:
            for chunk in audio:
                if chunk:
                    f.write(chunk)

    print("Saved file:", output_filename)
    print("File size:", os.path.getsize(output_filename), "bytes")

    return output_filename


# --- TEST SCRIPT ---
if __name__ == "__main__":

    try:
        # Read JSON file
        with open("test.json", "r", encoding="utf-8") as file:
            file_contents = file.read()

        print("Loaded JSON:", file_contents)

        generated_file_path = generate_audio_from_feedback(file_contents)

        print("Done! Opening:", generated_file_path)

        # Open audio file
        os.startfile(generated_file_path)

    except Exception as e:
        print("ERROR:", e)