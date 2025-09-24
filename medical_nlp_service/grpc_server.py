# import grpc
# from concurrent import futures
# import audio_processor_pb2
# import audio_processor_pb2_grpc
# import torch
# import torchaudio
# from pyannote.audio import Pipeline
# from transformers import pipeline
# import numpy as np
# import asyncio
# from typing import AsyncIterator

# class AudioProcessorService(audio_processor_pb2_grpc.AudioProcessorService):
#     def __init__(self):
#         self.sessions = {}
#         self.whisper_model = None
#         self.diarization_pipeline = None
#         self.medical_ner = None
#         self.medical_llm = None

#     async def _initialize_session(self, session_id: str):
#         self.sessions[session_id] = {
#             'audio_buffer': [],
#             'transcript_buffer': [],
#             'speakers': {},
#             'entities': [],
#             'start_time': None
#         }
    
#     async def ProcessAudioStream(self, request_iterator: AsyncIterator, content):
#         try:
#             async for chunk in request_iterator:
#                 session_id = chunk.session_id

#                 if session_id not in self.sessions:
#                     await self._initialize_session(session_id)

#                 results = await self._process_chunk(chunk)

#     async def _process_chunk(self, chunk) -> list:
#         results = []
#         transcript = await self._transcribe_chunk(chunk)