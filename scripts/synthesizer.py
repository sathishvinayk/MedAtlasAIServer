from gtts import gTTS
import subprocess
import os
import time

# --- Configuration ---
script = [
    ("Doctor", "Come in, please have a seat."),
    ("Patient", "Thank you, Doctor. It's good to see you."),
    ("Doctor", "You're welcome. So, how have you been feeling since we started you on the Lisinopril two weeks ago?"),
    ("Patient", "Well, overall I think it's helping. My blood pressure feels more stable, but I've been experiencing some side effects."),
    ("Doctor", "I see. What kind of side effects have you noticed?"),
    ("Patient", "The main thing is this persistent dry cough, especially at night. It's been keeping me up, and I've also been feeling more tired during the day."),
    ("Doctor", "I'm sorry to hear that. The dry cough is actually a known side effect of ACE inhibitors like Lisinopril. Have you noticed any dizziness or lightheadedness?"),
    ("Patient", "Yes, actually. I get dizzy sometimes when I stand up too quickly, and I've had a couple of headaches this week."),
    ("Doctor", "Okay, let me make note of that. Any swelling in your ankles or feet? Shortness of breath?"),
    ("Patient", "No swelling that I've noticed, but I do get short of breath going up stairs, though that's not entirely new for me."),
    ("Doctor", "Understood. Let's check your vitals today. Your blood pressure is 138 over 82, which is much better than your previous reading of 160 over 95."),
    ("Patient", "That's good news at least. But what about these side effects? Should I stop taking the medication?"),
    ("Doctor", "I don't recommend stopping abruptly. The cough and dizziness are common with this class of medication. We have a few options - we could reduce the dosage, add another medication to counteract the side effects, or switch you to a different type of blood pressure medication altogether."),
    ("Patient", "What would you recommend, Doctor?"),
    ("Doctor", "Given that you're responding well to the blood pressure control but having significant side effects, I'd suggest we switch you to an ARB medication like Losartan. It works similarly but typically doesn't cause the cough side effect."),
    ("Patient", "That sounds reasonable. Will I need any tests before switching?"),
    ("Doctor", "We should check your kidney function with a blood test before starting the new medication. I'll also want to see you back in 4 weeks to check your blood pressure on the new medication."),
    ("Patient", "Okay, that makes sense. Should I continue with the Lisinopril until then?"),
    ("Doctor", "Let's have you stop the Lisinopril today, and we'll start the Losartan tomorrow morning. I'll send the prescription to your pharmacy."),
    ("Patient", "Thank you, Doctor. I appreciate you taking the time to explain everything."),
    ("Doctor", "Of course. Remember to monitor how you're feeling, and don't hesitate to call if the symptoms worsen or if you have any concerns. Let's schedule your follow-up for four weeks from today."),
    ("Patient", "Will do. Thank you again."),
    ("Doctor", "Take care, and I'll see you next month.")
]

# Differentiate voices by language accent (top-level domain)
voice_profiles = {
    "Doctor": {"tld": "ca", "slow": False},   # Canadian English - professional, clear
    "Patient": {"tld": "co.uk", "slow": True}  # British English - slightly slower, more deliberate
}

final_output_file = "doctor_patient_conversation.mp3"

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

# --- Function to add pause between lines ---
def add_pause_between_lines(input_file, output_file, pause_duration=1.0):
    """Add silence between lines for more natural conversation flow"""
    try:
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-af', f'apad=pad_dur={pause_duration}',
            output_file,
            '-y'
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"Failed to add pause: {e}")
        return False

# --- Main Execution ---
print("Creating synthetic doctor-patient conversation...")
print(f"Total lines: {len(script)}")
print("-" * 50)

# Create a list to hold all the audio file names
audio_files = []
temp_dir = "temp_lines"
os.makedirs(temp_dir, exist_ok=True)

# 1. Generate audio for each line with progress tracking
successful_generations = 0
for i, (speaker, line_text) in enumerate(script):
    print(f"[{i+1:02d}/{len(script):02d}] {speaker}: {line_text[:50]}...")
    
    temp_filename = f"{temp_dir}/line_{i:02d}.mp3"
    temp_with_pause = f"{temp_dir}/line_{i:02d}_paused.mp3"
    
    if generate_audio_for_line(line_text, temp_filename, **voice_profiles[speaker]):
        # Add pause after each line
        if add_pause_between_lines(temp_filename, temp_with_pause, pause_duration=1.5):
            audio_files.append(temp_with_pause)
            successful_generations += 1
        else:
            audio_files.append(temp_filename)  # Fallback to original
            successful_generations += 1
    
    # Add small delay to avoid rate limiting
    time.sleep(0.5)

print(f"\nSuccessfully generated {successful_generations}/{len(script)} lines")

if successful_generations == 0:
    print("No audio files were generated. Exiting.")
    exit(1)

# 2. Create a text file listing all audio files for ffmpeg
concat_list_file = f"{temp_dir}/concat_list.txt"
with open(concat_list_file, 'w') as f:
    for audio_file in audio_files:
        f.write(f"file '{os.path.abspath(audio_file)}'\n")

# 3. Use ffmpeg to concatenate all files with proper audio normalization
try:
    print("\nCombining audio files with ffmpeg...")
    
    # Use filter complex for better audio quality and normalization
    input_files = "".join([f"-i {file} " for file in audio_files])
    filter_complex = f"concat=n={len(audio_files)}:v=0:a=1 [a]"
    
    cmd = f'ffmpeg {input_files} -filter_complex "{filter_complex}" -map "[a]" -af "loudnorm=I=-16:TP=-1.5:LRA=11" -c:a libmp3lame -q:a 2 {final_output_file} -y'
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error with filter complex: {result.stderr}")
        # Fallback to simple concat
        print("Trying simple concat method...")
        cmd = [
            'ffmpeg', 
            '-f', 'concat', 
            '-safe', '0', 
            '-i', concat_list_file,
            '-c', 'copy',
            final_output_file,
            '-y'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg failed: {result.stderr}")
    
    # Get final file duration
    duration_cmd = ['ffmpeg', '-i', final_output_file, '2>&1']
    duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
    
    print(f"✅ Success! Audio file created: {final_output_file}")
    print(f"📊 Conversation duration: Approximately {len(script) * 3} seconds")
    
except FileNotFoundError:
    print("❌ Error: ffmpeg not found. Please install ffmpeg:")
    print("macOS: brew install ffmpeg")
    print("Windows: Download from https://ffmpeg.org/download.html")
    print("Linux: sudo apt install ffmpeg")

except Exception as e:
    print(f"❌ An error occurred: {e}")

finally:
    # 4. Cleanup temporary files
    print("\n🧹 Cleaning up temporary files...")
    for file in audio_files:
        try:
            os.remove(file)
        except:
            pass
    # Remove original files without pauses
    for i in range(len(script)):
        try:
            os.remove(f"{temp_dir}/line_{i:02d}.mp3")
        except:
            pass
    try:
        os.remove(concat_list_file)
        os.rmdir(temp_dir)
    except:
        pass

print("🎉 Process completed successfully!")
print(f"🎧 Your conversation is ready: {final_output_file}")