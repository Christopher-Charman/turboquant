"""Chunked prefill and decode preserve KV token order (CPU-only regression)."""
import importlib.util
import random
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tq_capture_order", ROOT / "turboquant" / "capture.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class RecordingStore:
    def __init__(self):
        self.num_kv_heads = 1
        self.head_dim = 4
        self.device = torch.device("cpu")
        self.seen = []

    @property
    def num_tokens(self):
        return len(self.seen)

    def append_chunk(self, keys, values):
        self.seen.extend((int(k), int(v)) for k, v in
                         zip(keys[:, 0, 0].tolist(), values[:, 0, 0].tolist()))

    def reset(self):
        self.seen.clear()


class ChunkedPrefillOrderTests(unittest.TestCase):
    def check_sequence(self, chunks, capacity=8, mix_decode=False):
        store = RecordingStore()
        engine = capture.KVCaptureEngine(
            store, ring_capacity=capacity, device=torch.device("cpu"),
            dtype=torch.float32,
        )
        expected = []
        pos = 0
        for i, size in enumerate(chunks):
            keys = torch.arange(pos, pos + size, dtype=torch.float32).reshape(size, 1, 1).expand(size, 1, 4)
            values = -keys
            if mix_decode and i % 2:
                engine.ingest_decode(keys, values, size)
            else:
                engine.ingest_prefill(keys, values, size)
            expected.extend((j, -j) for j in range(pos, pos + size))
            pos += size
            self.assertEqual(engine.total_tokens, pos, (chunks, capacity))
        engine.flush()
        self.assertEqual(store.seen, expected, (chunks, capacity, mix_decode))
        self.assertEqual(engine.total_tokens, len(expected))
        engine.reset()
        self.assertEqual(engine.total_tokens, 0)

    def test_known_regressions(self):
        for chunks in ([12,12], [4,12], [16,16], [2,20,2], [12,4], [6,6,6], [8,8,8], [1]*20):
            with self.subTest(chunks=chunks):
                self.check_sequence(chunks)

    def test_randomized_prefill(self):
        rng = random.Random(20261016)
        for _ in range(120):
            cap = rng.choice([2,4,8,16])
            chunks = [rng.randint(1,24) for _ in range(rng.randint(1,9))]
            with self.subTest(cap=cap, chunks=chunks):
                self.check_sequence(chunks, cap)

    def test_mixed_prefill_decode(self):
        for chunks in ([12,2,20,2], [4,8,12,4], [2,20,2,16], [16,1,16,1]):
            with self.subTest(chunks=chunks):
                self.check_sequence(chunks, mix_decode=True)

if __name__ == "__main__":
    unittest.main()
