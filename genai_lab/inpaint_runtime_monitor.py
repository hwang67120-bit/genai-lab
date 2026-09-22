"""Replay child runtime JSONL to the parent's PowerShell without waiting for exit."""

import json
import sys
from pathlib import Path


class RuntimeLogTail:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0

    def poll(self):
        records = []
        try:
            with self.path.open('rb') as stream:
                stream.seek(self.offset)
                for _ in range(512):
                    raw = stream.readline()
                    if not raw or not raw.endswith(b'\n'):
                        break
                    self.offset = stream.tell()
                    try:
                        item = json.loads(raw)
                        if isinstance(item, dict):
                            records.append(item)
                    except (ValueError, UnicodeError):
                        continue
        except OSError:
            pass
        for item in records:
            try:
                resources = item.get('resources', {})
                active = item.get('active', [])
                stage = ' > '.join(f"{x['stage']}({x['active_seconds']}s)" for x in active) or item.get('stage')
                print(
                    f"[복원 계측][{resources.get('pid', '?')}][{stage}] {item.get('event')} "
                    f"call={item.get('call', '-')} duration={item.get('duration', '-')}s "
                    f"RAM_RSS={resources.get('rss_mib', '?')}MiB "
                    f"RAM_Private={resources.get('private_mib', '?')}MiB "
                    f"RAM_Free={resources.get('system_available_mib', '?')}MiB "
                    f"VRAM_alloc={resources.get('cuda_allocated_mib', '?')}MiB "
                    f"VRAM_reserved={resources.get('cuda_reserved_mib', '?')}MiB "
                    f"error={item.get('error_type', '-')}",
                    file=sys.stderr, flush=True,
                )
            except Exception:
                pass
        return records
