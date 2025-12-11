import openai
from IPython.display import Audio
import io
import os
from dotenv import load_dotenv
from storytellor import story
load_dotenv() 
openai_api_key = os.getenv("OPENAI_API_KEY")

# Make sure your OpenAI API key is set


# The text you want to convert to speech
text_to_speak = story  # your generated story

# Generate speech using OpenAI TTS
audio_response = openai.audio.speech.create(
    model="gpt-4o-mini-tts",  # OpenAI TTS model
    voice="alloy",            # optional: choose a voice
    input=text_to_speak
)

# Convert the response to bytes
audio_bytes = io.BytesIO(audio_response.read())
audio_bytes.seek(0)

# Play audio in the notebook
Audio(audio_bytes.read(), autoplay=True)

# Save as MP3 file
# audio_bytes.save("generated_story.mp3")
