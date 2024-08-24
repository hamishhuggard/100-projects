import boto3

# Initialize a session using Amazon Polly
polly = boto3.client('polly', region_name='us-east-1')  # Replace with your desired region

# Text to synthesize
text = "Hello, how are you today?"

# Synthesize speech and get MP3 audio
response_audio = polly.synthesize_speech(
    Text=text,
    OutputFormat='mp3',
    VoiceId='Joanna'
)

# Save the MP3 audio to a file
with open('speech.mp3', 'wb') as audio_file:
    audio_file.write(response_audio['AudioStream'].read())

# Synthesize speech and get Speech Marks in JSON format
response_speech_marks = polly.synthesize_speech(
    Text=text,
    OutputFormat='json',
    VoiceId='Joanna',
    SpeechMarkTypes=['word']
)

# Save the Speech Marks to a file
with open('speech_marks.json', 'wb') as marks_file:
    marks_file.write(response_speech_marks['AudioStream'].read())

print("MP3 audio saved as 'speech.mp3' and speech marks saved as 'speech_marks.json'")
