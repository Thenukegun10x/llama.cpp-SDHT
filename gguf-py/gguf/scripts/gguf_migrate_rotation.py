"""Rename legacy input-rotation keys while copying all model payload bytes."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from gguf import GGUFEndian, GGUFReader
from gguf.rotation import LEGACY_ROTATION_PREFIX, rotation_key, rotation_metadata


def migrate(source: Path, output: Path) -> dict[str, object]:
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise ValueError("output must be a new file, different from the input")
    reader = GGUFReader(source)
    created = False
    try:
        rotation_metadata(reader.fields)
        fields = [field for name, field in reader.fields.items() if not name.startswith("GGUF.")]
        renamed = [field.name for field in fields if field.name.startswith(LEGACY_ROTATION_PREFIX)]
        if not renamed:
            raise ValueError("input has no legacy rotation keys to rename")
        sizes = {field.name: sum(part.nbytes for part in field.parts) for field in fields}
        metadata_end = max((field.offset + sizes[field.name] for field in fields), default=24)
        info_end = max((tensor.field.offset + sum(part.nbytes for part in tensor.field.parts)
                        for tensor in reader.tensors), default=metadata_end)
        endian = "little" if reader.endianess == GGUFEndian.LITTLE else "big"
        digest = hashlib.sha256()
        payload_bytes = 0
        with source.open("rb") as src, output.open("xb") as dst:
            created = True
            dst.write(src.read(24))
            for field in fields:
                key = rotation_key(field.name).encode("utf-8")
                old_key_bytes = field.parts[1].nbytes
                src.seek(field.offset + 8 + old_key_bytes)
                value = src.read(sizes[field.name] - 8 - old_key_bytes)
                dst.write(len(key).to_bytes(8, endian))
                dst.write(key)
                dst.write(value)
            src.seek(metadata_end)
            dst.write(src.read(info_end - metadata_end))
            dst.write(b"\0" * (-dst.tell() % reader.alignment))
            src.seek(reader.data_offset)
            while chunk := src.read(8 * 1024 * 1024):
                dst.write(chunk)
                digest.update(chunk)
                payload_bytes += len(chunk)
        return {"renamed_keys": len(renamed), "payload_bytes": payload_bytes,
                "payload_sha256": digest.hexdigest(), "output": str(output)}
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        raise
    finally:
        reader.data._mmap.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import json
    print(json.dumps(migrate(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
