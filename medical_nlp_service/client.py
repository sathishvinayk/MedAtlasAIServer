import asyncio
import websockets
import json
import sys
import os
import base64  # ADDED
import time    # ADDED

class SimpleDeepScribeClient:
    """
    Simple DeepScribe-like client without audio dependencies
    """
    
    def __init__(self, server_url="ws://localhost:8000/ws/realtime-audio"):
        self.server_url = server_url
        self.websocket = None
        self.is_connected = False
    
    async def _listen_for_results(self):
        """Enhanced listener with timeout reset on any message"""
        last_message_time = time.time()
        timeout = 60.0  # Windows timeout threshold
        
        try:
            async for message in self.websocket:
                last_message_time = time.time()  # Reset timeout on ANY message
                status = await self._handle_result(message)
                if status in ["COMPLETE", "ERROR"]:
                    break
                    
                # Check if we're approaching timeout
                if time.time() - last_message_time > timeout - 5:  # 5 seconds before timeout
                    print("⚠️  No recent messages - connection may timeout")
                    
        except Exception as e:
            print(f"❌ Result listening error: {e}")
    
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
        """Stream audio file with proper final processing handling"""
        if not await self.connect():
            return False
        
        try:
            # Start listening for results
            listener_task = asyncio.create_task(self._listen_for_results())
            
            if sys.platform == "win32":
                ping_task = asyncio.create_task(self._send_periodic_pings())
            
            # Read and stream audio file
            with open(file_path, 'rb') as f:
                file_size = os.path.getsize(file_path)
                print(f"📁 Streaming {file_path} ({file_size} bytes)...")
                
                audio_data = f.read()
                total_chunks = len(audio_data) // chunk_size + (1 if len(audio_data) % chunk_size else 0)
                
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                    await self.websocket.send(chunk_b64)
                    
                    # Show progress
                    if i % (chunk_size * 10) == 0:
                        progress = (i / len(audio_data)) * 100
                        print(f"📊 Progress: {progress:.1f}% ({i}/{len(audio_data)} bytes)")
                    
                    await asyncio.sleep(0.01)
            
            print("✅ File streaming complete")
            
            # Clean up
            if sys.platform == "win32":
                ping_task.cancel()
            
            # Send end stream signal
            await self.websocket.send("END_STREAM")
            print("📤 Sent END_STREAM signal")
            
            # Wait for listener to complete naturally
            print("⏳ Waiting for final processing...")
            try:
                await asyncio.wait_for(listener_task, timeout=560.0)  # 60 second timeout
                print("✅ Processing completed successfully")
            except asyncio.TimeoutError:
                print("❌ Processing timeout - taking too long")
                listener_task.cancel()
            
            return True
            
        except Exception as e:
            print(f"❌ File streaming error: {e}")
            return False
        finally:
            await self.disconnect()
    
    async def _send_periodic_pings(self):
        """Send periodic pings to reset Windows timeout"""
        try:
            while True:
                await asyncio.sleep(5.0)  # Every 10 seconds
                # WebSocket protocol ping (not application-level)
                if hasattr(self.websocket, 'ping'):
                    await self.websocket.ping()
                    print("🏓 Sent protocol ping")
        except:
            pass  # Connection closed
            
    async def _listen_for_results(self):
        """Listen for real-time results from server"""
        try:
            async for message in self.websocket:
                await self._handle_result(message)
        except Exception as e:
            print(f"❌ Result listening error: {e}")
    
    async def _handle_result(self, message: str):
        try:
            result = json.loads(message)
            result_type = result.get('type', 'unknown')
            
            if result_type == "soap_update":
                soap_note = result.get('data', {}).get('soap_note', '')
                if soap_note:
                    print(f"📝 SOAP Update: {soap_note[:100]}...")
            
            elif result_type == "soap_note_complete":
                soap_note = result.get('data', {}).get('content', '')
                print(f"\n🎉 FINAL SOAP NOTE RECEIVED!")
                print("=" * 60)
                print(soap_note)
                print("=" * 60)
                return "COMPLETE"
            elif result_type == "medical_alert":
                severity = result.get('severity', 'moderate')
                message = result.get('message', '')
                alert_type = result.get('alert_type', 'safety_alert')
                
                # Color code by severity
                if severity == "urgent":
                    icon = "🚨"
                    color = "RED"
                elif severity == "high":
                    icon = "⚠️" 
                    color = "YELLOW"
                else:
                    icon = "ℹ️"
                    color = "BLUE"
                
                print(f"\n{icon} MEDICAL ALERT ({severity.upper()}): {message}")
                print(f"   Type: {alert_type}")
                print(f"   Recommendation: {result.get('recommendation', 'Please review')}")
                
        except Exception as e:
            print(f"Error handling result: {e}")

    async def _listen_for_results(self):
        """Listen for real-time results from server"""
        try:
            async for message in self.websocket:
                status = await self._handle_result(message)
                if status in ["COMPLETE", "ERROR"]:
                    break  # Stop listening when processing is complete
        except Exception as e:
            print(f"❌ Result listening error: {e}")
    
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
    asyncio.run(main())