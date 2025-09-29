# shared_utils.py
import asyncio
from typing import List
import logging

logger = logging.getLogger("shared-utils")

class AudioChunkGenerator:
    """Generator to convert HTTP chunks to async iterator"""
    def __init__(self):
        self.chunks = asyncio.Queue()
        self.final_chunk_sent = False
    
    async def put_chunk(self, chunk: bytes, is_final: bool = False):
        await self.chunks.put((chunk, is_final))
        if is_final:
            self.final_chunk_sent = True
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.final_chunk_sent and self.chunks.empty():
            raise StopAsyncIteration
        chunk, is_final = await self.chunks.get()
        return chunk