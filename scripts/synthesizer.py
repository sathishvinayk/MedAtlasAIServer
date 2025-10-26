from gtts import gTTS
import subprocess
import os
import time

# --- Extended realistic doctor-patient script with casual conversation ---
script = [
    ("Doctor", "I see. What kind of side effects have you noticed?"),
    ("Patient", "The main thing is this persistent dry cough, especially at night. It's been keeping me up, and I've also been feeling more tired during the day."),
    ("Doctor", "I'm sorry to hear that. The dry cough is actually a known side effect of ACE inhibitors like Lisinopril. Have you noticed any dizziness or lightheadedness?"),
    ("Patient", "Yes, actually. I get dizzy sometimes when I stand up too quickly, and I've had a couple of headaches this week."),
    ("Doctor", "Okay, let me make note of that. Any swelling in your ankles or feet? Shortness of breath?")
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