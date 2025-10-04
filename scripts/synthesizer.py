from gtts import gTTS
import subprocess
import os
import time

# --- Extended realistic doctor-patient script with casual conversation ---
script = [
    ("Doctor", "Hi! Come on in, make yourself comfortable. How’s your week been so far?"),
    ("Patient", "Hi Doctor… it’s been hectic. Work’s been crazy, deadlines everywhere."),
    ("Doctor", "I hear you. It seems everyone’s been feeling that way lately. Any chance you got a bit of rest over the weekend?"),
    ("Patient", "A little, but honestly I spent half of it catching up on emails."),
    ("Doctor", "Well, at least you managed some downtime. Now, tell me, what brought you in today?"),
    ("Patient", "Um… I’ve been having these headaches for the past week. They come and go, but sometimes they get pretty bad."),
    ("Doctor", "I see. Can you describe them? Are they throbbing, sharp, or more like pressure?"),
    ("Patient", "Mostly pressure… like a band around my head. But occasionally they spike and feel sharp."),
    ("Doctor", "Got it. And when do they usually occur? Morning, afternoon, night?"),
    ("Patient", "Mostly in the afternoon. After I’ve been at the computer for a while."),
    ("Doctor", "Sounds like tension headaches. Do you notice any nausea, sensitivity to light or sound?"),
    ("Patient", "Yeah, light bothers me sometimes… and I feel a little nauseous when it’s bad."),
    ("Doctor", "Okay, that helps. Any other symptoms, like dizziness or blurred vision?"),
    ("Patient", "A bit dizzy if I stand up too quickly. Nothing else really."),
    ("Doctor", "Understood. Have you had migraines in the past?"),
    ("Patient", "Yes, but only occasionally. Haven’t had a big episode in a couple of years."),
    ("Doctor", "Alright. Let’s also talk about lifestyle a bit. How’s your sleep?"),
    ("Patient", "Not great… I’ve been staying up late finishing work."),
    ("Doctor", "That can contribute. And diet? Are you eating regularly?"),
    ("Patient", "Honestly, I skip breakfast often. Lunch is usually whatever’s quick at the office."),
    ("Doctor", "That’s probably not helping the headaches. Hydration too—are you drinking enough water?"),
    ("Patient", "I try, but coffee takes over most of the day."),
    ("Doctor", "Coffee’s fine in moderation, but make sure to balance with water. How about exercise?"),
    ("Patient", "Hah, exercise… barely. Maybe a short walk if I remember."),
    ("Doctor", "Alright, we’ll come back to that. Let’s check your blood pressure and do a basic exam first."),
    ("Patient", "Okay, sounds good."),
    ("Doctor", "By the way, did you manage to take that trip you were planning last month?"),
    ("Patient", "Oh, no… had to cancel. Work deadlines again."),
    ("Doctor", "I understand. Hopefully you can plan a short break soon—it helps reduce stress too."),
    ("Patient", "Yeah, I hope so."),
    ("Doctor", "Okay, your vitals look fine. Blood pressure is 120 over 78. Heart rate is normal."),
    ("Patient", "That’s a relief."),
    ("Doctor", "Now, regarding the headaches, I think they’re mostly tension-related. But I want you to track them: time, severity, triggers."),
    ("Patient", "Alright, I can do that."),
    ("Doctor", "In the meantime, acetaminophen or ibuprofen can help when they get intense. Make sure not to exceed recommended doses."),
    ("Patient", "Got it."),
    ("Doctor", "Also, try short breaks from screens, stretching, and some light walking."),
    ("Patient", "I’ll try that. It’s just hard with all the deadlines."),
    ("Doctor", "I get it. Even 5–10 minutes can make a difference. And try to prioritize sleep where you can."),
    ("Patient", "Yeah… easier said than done, but I’ll try."),
    ("Doctor", "Anything else on your mind? Stress, work, or anything personal bothering you? Sometimes that shows up physically."),
    ("Patient", "Well… I guess I’ve been feeling anxious about a project at work. That probably doesn’t help the headaches."),
    ("Doctor", "That’s completely normal. Stress can trigger physical symptoms. Mindfulness, short walks, or talking about it can help."),
    ("Patient", "I’ll try to keep that in mind."),
    ("Doctor", "Good. Let’s schedule a follow-up in a week or sooner if the headaches get worse."),
    ("Patient", "Okay. Thank you for listening and for the advice."),
    ("Doctor", "Of course! Remember, self-care matters, even with busy schedules."),
    ("Patient", "I’ll try. Thanks again, Doctor."),
    ("Doctor", "Take care, and don’t hesitate to call if anything changes."),
]


# Differentiate voices by language accent (top-level domain)
voice_profiles = {
    "Doctor": {"tld": "ca", "slow": False},   # Canadian English - professional, clear
    "Patient": {"tld": "co.uk", "slow": True}  # British English - slightly slower, more deliberate
}

final_output_file = "doctor_patient_conversation.wav"

# --- Function to generate TTS audio for a line ---
def generate_audio_for_line(text, filename, **kwargs):
    try:
        tts = gTTS(text=text, lang='en', **kwargs)
        tts.save(filename)
        print(f"✓ Generated: {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to generate {filename}: {e}")
        return False

# --- Function to convert MP3 to proper WAV ---
def convert_to_wav(input_file, output_file):
    """Convert MP3 to proper WAV format with correct encoding"""
    try:
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-acodec', 'pcm_s16le',  # Proper WAV codec
            '-ar', '16000',          # 16kHz sample rate
            '-ac', '1',              # Mono
            '-f', 'wav',             # Force WAV format
            output_file,
            '-y'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Verify the conversion worked
        if os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
            print(f"✓ Converted to WAV: {output_file}")
            return True
        else:
            print(f"✗ Conversion failed: {output_file} is too small")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ FFmpeg conversion failed: {e}")
        print(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ Conversion error: {e}")
        return False

# --- Main Execution ---
print("Creating synthetic doctor-patient conversation...")
print(f"Total lines: {len(script)}")
print("-" * 50)

# Create a list to hold all the WAV file names
audio_files = []
temp_dir = "temp_lines"
os.makedirs(temp_dir, exist_ok=True)

# 1. Generate audio for each line with progress tracking
successful_generations = 0
for i, (speaker, line_text) in enumerate(script):
    print(f"[{i+1:02d}/{len(script):02d}] {speaker}: {line_text[:50]}...")
    
    # Generate as MP3 first (gTTS limitation), then convert to proper WAV
    temp_mp3 = f"{temp_dir}/line_{i:02d}.mp3"
    temp_wav = f"{temp_dir}/line_{i:02d}.wav"
    
    if generate_audio_for_line(line_text, temp_mp3, **voice_profiles[speaker]):
        # Convert MP3 to proper WAV format
        if convert_to_wav(temp_mp3, temp_wav):
            audio_files.append(temp_wav)
            successful_generations += 1
            # Remove the temporary MP3 file
            os.remove(temp_mp3)
        else:
            print(f"✗ Failed to convert line {i} to WAV")
    
    # Add small delay to avoid rate limiting
    time.sleep(1)

print(f"\nSuccessfully generated {successful_generations}/{len(script)} lines")

if successful_generations == 0:
    print("No audio files were generated. Exiting.")
    exit(1)

# 2. Concatenate all WAV files
try:
    print("\nCombining WAV files...")
    
    # Create file list for concatenation
    concat_list_file = f"{temp_dir}/file_list.txt"
    with open(concat_list_file, 'w') as f:
        for audio_file in audio_files:
            f.write(f"file '{os.path.abspath(audio_file)}'\n")
    
    # Concatenate using ffmpeg
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_list_file,
        '-c', 'copy',  # Copy without re-encoding
        final_output_file,
        '-y'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Success! Final WAV file created: {final_output_file}")
        
        # Verify final file
        final_size = os.path.getsize(final_output_file)
        print(f"📊 Final file size: {final_size} bytes")
        
        # Test base64 encoding
        base64_test = subprocess.run(['base64', '-w', '0', final_output_file], capture_output=True, text=True)
        if len(base64_test.stdout) > 100:
            print(f"✓ Base64 encoding test passed: {len(base64_test.stdout)} characters")
        else:
            print("✗ Base64 encoding test failed - string too short")
            
    else:
        print(f"❌ Concatenation failed: {result.stderr}")

except Exception as e:
    print(f"❌ An error occurred: {e}")

finally:
    # Cleanup temporary files
    print("\n🧹 Cleaning up temporary files...")
    for file in audio_files:
        try:
            os.remove(file)
        except:
            pass
    try:
        os.remove(concat_list_file)
        os.rmdir(temp_dir)
    except:
        pass

print("🎉 Process completed!")