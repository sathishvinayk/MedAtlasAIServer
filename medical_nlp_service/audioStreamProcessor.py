import struct
class AudioStreamProcessor:
    """Process audio streams in real-time without temp files"""
    
    def __init__(self, sample_rate=16000, channels=1, bits_per_sample=16):
        self.sample_rate = sample_rate
        self.channels = channels
        self.bits_per_sample = bits_per_sample
        self.bytes_per_sample = bits_per_sample // 8
        
    def add_audio_to_buffer(self, buffer: bytearray, audio_data: bytes):
        """Add audio data to buffer for continuous processing"""
        buffer.extend(audio_data)
        # Keep buffer manageable (max 30 seconds of audio)
        max_buffer_size = 30 * self.sample_rate * self.channels * self.bytes_per_sample
        if len(buffer) > max_buffer_size:
            # Keep the most recent 20 seconds
            keep_size = 20 * self.sample_rate * self.channels * self.bytes_per_sample
            buffer = buffer[-keep_size:]
    
    def create_wav_from_buffer(self, buffer: bytearray) -> bytes:
        """Create valid WAV file from audio buffer"""
        if len(buffer) < 1000:  # Minimum audio size
            return b""
        
        # Calculate sizes
        data_size = len(buffer)
        file_size = data_size + 36  # WAV header size minus 8 bytes
        
        # Create WAV header
        byte_rate = self.sample_rate * self.channels * self.bytes_per_sample
        block_align = self.channels * self.bytes_per_sample
        
        header = b'RIFF'
        header += struct.pack('<I', file_size)
        header += b'WAVE'
        header += b'fmt '
        header += struct.pack('<I', 16)
        header += struct.pack('<H', 1)  # PCM
        header += struct.pack('<H', self.channels)
        header += struct.pack('<I', self.sample_rate)
        header += struct.pack('<I', byte_rate)
        header += struct.pack('<H', block_align)
        header += struct.pack('<H', self.bits_per_sample)
        header += b'data'
        header += struct.pack('<I', data_size)
        
        return header + bytes(buffer)