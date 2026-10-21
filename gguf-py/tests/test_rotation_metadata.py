import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gguf import GGUFReader, GGUFWriter, Keys
from gguf.rotation import ROTATION_PREFIX, LEGACY_ROTATION_PREFIX, rotation_metadata
from gguf.scripts.gguf_migrate_rotation import migrate


class RotationMetadataTests(unittest.TestCase):
    def test_namespaces_and_incomplete_contracts(self):
        suffixes = {"version": 2, "tied_output": True, "block_size": 128,
                    "transform": "normalized-smooth-discrete-hartley",
                    "inverse_weight_names": ["token_embd.weight"],
                    "weight_names": ["blk.0.ffn_gate.weight"],
                    "diagonal_weight_names": ["blk.0.ffn_gate.weight"]}
        canonical = {ROTATION_PREFIX + key: value for key, value in suffixes.items()}
        legacy = {LEGACY_ROTATION_PREFIX + key: value for key, value in suffixes.items()}
        self.assertEqual(rotation_metadata(canonical), canonical)
        self.assertEqual(rotation_metadata(legacy), canonical)
        for invalid in [dict(canonical, **legacy), {ROTATION_PREFIX + "block_size": 128},
                        {LEGACY_ROTATION_PREFIX + "transform": "normalized-discrete-hartley"}]:
            with self.assertRaises(ValueError):
                rotation_metadata(invalid)

    def test_writer_and_lossless_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "old.gguf", Path(tmp) / "new.gguf"
            writer = GGUFWriter(source, "llama")
            writer.add_uint32(LEGACY_ROTATION_PREFIX + "version", 1)
            writer.add_string(LEGACY_ROTATION_PREFIX + "transform", "normalized-discrete-hartley")
            writer.add_array(LEGACY_ROTATION_PREFIX + "weight_names", ["test.weight"])
            writer.add_tensor("test.weight", np.arange(256, dtype=np.float32).reshape(2, 128))
            writer.add_tensor("test.weight.input_diag", np.ones(128, dtype=np.float16))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            record = migrate(source, output)
            old, new = GGUFReader(source), GGUFReader(output)
            try:
                self.assertEqual(record["renamed_keys"], 3)
                self.assertEqual(rotation_metadata(old.fields).keys(), rotation_metadata(new.fields).keys())
                self.assertFalse(any(name.startswith(LEGACY_ROTATION_PREFIX) for name in new.fields))
                for before, after in zip(old.tensors, new.tensors):
                    self.assertEqual(before.name, after.name)
                    self.assertEqual(before.tensor_type, after.tensor_type)
                    np.testing.assert_array_equal(before.data, after.data)
                before = source.read_bytes()[old.data_offset:]
                after = output.read_bytes()[new.data_offset:]
                self.assertEqual(before, after)
                self.assertEqual(record["payload_sha256"], hashlib.sha256(after).hexdigest())
                with self.assertRaises(ValueError):
                    migrate(source, output)
                with self.assertRaises(ValueError):
                    migrate(source, source)
                new_output = Path(tmp) / "again.gguf"
                with self.assertRaises(ValueError):
                    migrate(output, new_output)
                self.assertFalse(new_output.exists())
            finally:
                old.data._mmap.close()
                new.data._mmap.close()
            writer = GGUFWriter(Path(tmp) / "writer.gguf", "llama")
            writer.add_rotation_version(1)
            writer.add_rotation_diagonal_weight_names(["test.weight"])
            self.assertIn(Keys.Rotation.VERSION, writer.kv_data[0])
            self.assertIn(Keys.Rotation.DIAGONAL_WEIGHT_NAMES, writer.kv_data[0])
            self.assertFalse(any(name.startswith(LEGACY_ROTATION_PREFIX) for name in writer.kv_data[0]))


if __name__ == "__main__":
    unittest.main()
