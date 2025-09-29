import asyncio
import websockets
import json
import sys
import os

class SimpleDeepScribeClient:
    """
    Simple DeepScribe-like client without audio dependencies
    """
    
    def __init__(self, server_url="ws://localhost:8000/ws/realtime-audio"):
        self.server_url = server_url
        self.websocket = None
        self.is_connected = False
    
    async def connect(self):
        """Connect to WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.is_connected = True
            print("✅ Connected to DeepScribe server")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from server"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self.is_connected = False
        print("🔌 Disconnected")
    
    async def send_audio_file(self, file_path: str, chunk_size=4096):
        """Stream audio file with better final processing handling"""
        if not await self.connect():
            return False
        
        try:
            # Start listening for results
            listener_task = asyncio.create_task(self._listen_for_results())
            
            # Read and stream audio file
            with open(file_path, 'rb') as f:
                file_size = os.path.getsize(file_path)
                print(f"📁 Streaming {file_path} ({file_size} bytes)...")
                
                # Skip WAV header if present (44 bytes)
                header = f.read(44)
                if header.startswith(b'RIFF'):
                    print("🎵 Detected WAV file, skipping header")
                    audio_data = f.read()
                else:
                    # Not a WAV file, read everything
                    f.seek(0)
                    audio_data = f.read()
                
                total_chunks = len(audio_data) // chunk_size + (1 if len(audio_data) % chunk_size else 0)
                
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    await self.websocket.send(chunk)
                    
                    # Show progress
                    if i % (chunk_size * 10) == 0:  # Every 10 chunks
                        progress = (i / len(audio_data)) * 100
                        print(f"📊 Progress: {progress:.1f}%")
                    
                    # Small delay to simulate real-time
                    await asyncio.sleep(0.01)
            
            print("✅ File streaming complete")
            
            # CRITICAL: Wait much longer for final processing
            print("⏳ Waiting for final processing (this may take 30-60 seconds)...")
            
            # Wait for final results with progress updates
            for i in range(120):  # Wait up to 60 seconds
                await asyncio.sleep(1.0)
                if i % 10 == 0:  # Print progress every 10 seconds
                    print(f"⏰ Waiting... {i}s elapsed")
            
            print("ℹ️  Final processing should be complete. If no results, check server logs.")
            
            # Cancel listener after waiting
            listener_task.cancel()
            
            return True
            
        except Exception as e:
            print(f"❌ File streaming error: {e}")
            return False
        finally:
            await self.disconnect()
    
    async def _listen_for_results(self):
        """Listen for real-time results from server"""
        try:
            async for message in self.websocket:
                await self._handle_result(message)
        except Exception as e:
            print(f"❌ Result listening error: {e}")
    
    async def _handle_result(self, message: str):
        """Handle server results with progress tracking"""
        try:
            result = json.loads(message)
            result_type = result.get('type', 'unknown')
            data = result.get('data', {})
            
            if result_type == "transcript":
                transcript = data.get('text', '')
                if transcript:
                    prefix = "⏳" if result.get('is_partial', True) else "✅"
                    print(f"{prefix} Transcript: {transcript}")
            
            elif result_type == "entities":
                entities = data.get('entities', [])
                if entities:
                    entity_list = [f"{e.get('entity', 'unknown')}: {e.get('text', '')}" for e in entities]
                    print(f"🏥 Entities: {', '.join(entity_list)}")
            
            elif result_type == "progress":
                print(f"📈 {data.get('message', 'Processing...')}")
            
            elif result_type == "soap_note_complete":
                soap_note = data.get('content', '')
                duration = data.get('session_duration', 0)
                print("\n" + "="*60)
                print("🎉 PROCESSING COMPLETE!")
                print("="*60)
                print(f"Session Duration: {duration:.2f}s")
                print(f"Transcript Length: {data.get('transcript_length', 0)} chars")
                print(f"Entities Found: {data.get('entities_found', 0)}")
                print(f"Model Used: {data.get('model_used', 'unknown')}")
                print(f"Final SOAP Note:\n{soap_note}")
                print("="*60)
            
            elif result_type == "error":
                print(f"❌ Server Error: {data.get('message', 'Unknown error')}")
            
            elif result_type == "keepalive":
                # Silently handle keepalive messages
                pass
                
        except json.JSONDecodeError:
            print(f"📥 Raw message: {message}")
        except Exception as e:
            print(f"❌ Error handling result: {e}")
    
    async def test_connection(self):
        """Test server connection"""
        if await self.connect():
            print("✅ Server is responsive!")
            await self.disconnect()
            return True
        return False

async def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("DeepScribe-like Medical Transcription Client")
        print("Usage:")
        print("  python simple_client.py test                    # Test server connection")
        print("  python simple_client.py file <audio_file>      # Stream audio file")
        print("  python simple_client.py listen                 # Just listen to server")
        return
    
    client = SimpleDeepScribeClient()
    command = sys.argv[1]
    
    if command == "test":
        await client.test_connection()
    
    elif command == "file" and len(sys.argv) > 2:
        file_path = sys.argv[2]
        if os.path.exists(file_path):
            await client.send_audio_file(file_path)
        else:
            print(f"❌ File not found: {file_path}")
    
    elif command == "listen":
        print("👂 Listening for server messages...")
        if await client.connect():
            try:
                # Just listen indefinitely
                await asyncio.Future()  # Run forever
            except KeyboardInterrupt:
                print("\n🛑 Stopped listening")
            finally:
                await client.disconnect()
    
    else:
        print("❌ Invalid command")

if __name__ == "__main__":
    # Check if websockets is installed
    try:
        import websockets
    except ImportError:
        print("❌ websockets module not installed")
        print("Install it with: pip install websockets")
        sys.exit(1)
    
    asyncio.run(main())