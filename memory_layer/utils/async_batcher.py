import asyncio
from typing import Sequence, Callable, List, Any

class AsyncBatcher:
    def __init__(self, batch_size: int, concurrency: int):
        self.batch_size = batch_size
        self.sem = asyncio.Semaphore(concurrency)

    async def map(self, items: Sequence[Any], fn: Callable[[Sequence[Any]], Any]) -> List[Any]:
        async def _run(chunk):
            async with self.sem:
                return await fn(chunk)

        tasks = []
        for i in range(0, len(items), self.batch_size):
            tasks.append(asyncio.create_task(_run(items[i:i+self.batch_size])))
        out = []
        for t in tasks:
            out.extend(await t)
        return out