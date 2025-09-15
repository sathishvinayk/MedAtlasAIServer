from gtts import gTTS
import subprocess
import os

# --- Configuration ---
script = [
    ("Doctor", "Come in!"),
    ("Patient", "Thank you, Doctor."),
    ("Doctor", "So, how have you been since your last visit? We were going to try that new medication, the Lisinopril."),
    ("Patient", "Well, uh, mostly okay. The dizziness is much better, thank goodness."),
    ("Doctor", "Good, I'm glad to hear that. Any other side effects? Dry cough? Anything like that?"),
    ("Patient", "No cough, no. But I have been feeling a bit more tired than usual in the evenings."),
    ("Doctor", "That's not uncommon. We'll keep an eye on it."),
]

# Differentiate voices by language accent (top-level domain)
voice_profiles = {
    "Doctor": {"tld": "ca"},   # Canadian English
    "Patient": {"tld": "co.uk"} # British English
}

final_output_file = "doctor_patient_conversation.mp3"

# --- Function to generate TTS audio for a line ---
def generate_audio_for_line(text, filename, **kwargs):
    tts = gTTS(text=text, lang='en', **kwargs)
    tts.save(filename)

# --- Main Execution ---
print("Creating synthetic doctor-patient conversation...")

# Create a list to hold all the audio file names
audio_files = []
temp_dir = "temp_lines"
os.makedirs(temp_dir, exist_ok=True)

# 1. Generate audio for each line
for i, (speaker, line_text) in enumerate(script):
    print(f"Generating line {i+1}: {speaker}")
    
    temp_filename = f"{temp_dir}/line_{i:02d}.mp3"  # Pad with zeros for proper sorting
    generate_audio_for_line(line_text, temp_filename, **voice_profiles[speaker])
    audio_files.append(temp_filename)

# 2. Create a text file listing all audio files for ffmpeg
concat_list_file = f"{temp_dir}/concat_list.txt"
with open(concat_list_file, 'w') as f:
    for audio_file in audio_files:
        f.write(f"file '{os.path.abspath(audio_file)}'\n")
        f.write(f"duration 0.5\n")  # Add 0.5 seconds of silence after each file

# 3. Use ffmpeg to concatenate all files
try:
    # Method 1: Using concat protocol (more reliable)
    cmd = [
        'ffmpeg', 
        '-f', 'concat', 
        '-safe', '0', 
        '-i', concat_list_file,
        '-c', 'copy',
        final_output_file,
        '-y'  # Overwrite output file if it exists
    ]
    
    print("Combining audio files with ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error with concat method: {result.stderr}")
        # Fallback: Try alternative method
        print("Trying alternative method...")
        
        # Create a filter complex string
        filter_complex = "".join([f"[{i}:a]" for i in range(len(audio_files))])
        filter_complex += f"concat=n={len(audio_files)}:v=0:a=1[out]"
        
        # Build alternative command
        cmd = ['ffmpeg']
        for file in audio_files:
            cmd.extend(['-i', file])
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '[out]',
            final_output_file,
            '-y'
        ])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Alternative method also failed: {result.stderr}")
            raise Exception("FFmpeg failed to combine audio files")
    
    print(f"Success! Audio file created: {final_output_file}")

except FileNotFoundError:
    print("Error: ffmpeg not found. Please install ffmpeg and make sure it's in your system PATH.")
    print("Download from: https://ffmpeg.org/download.html")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # 4. Cleanup temporary files
    print("Cleaning up temporary files...")
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

print("Process completed.")