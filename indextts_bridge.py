#!/usr/bin/env python3
"""
IndexTTS2 Bridge — long-lived subprocess for voice cloning inference.
Runs inside the index-tts uv environment.
Communicates with the parent process via stdin/stdout JSON lines.

Protocol (one JSON object per line):
  → {"cmd": "init", "cfg_path": "...", "model_dir": "...", "use_fp16": false, ...}
  ← {"status": "ok"} | {"status": "error", "message": "..."}

  → {"cmd": "infer", "spk_audio_prompt": "...", "text": "...", "output_path": "...", ...}
  ← {"status": "ok", "output_path": "..."} | {"status": "error", "message": "..."}

  → {"cmd": "quit"}
  ← {"status": "ok"}
"""

import json
import os
import sys

INDEXTTS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, INDEXTTS_PATH)

_tts = None

try:
    from indextts.infer_v2 import IndexTTS2
except ImportError as e:
    print(json.dumps({"status": "error", "message": f"Import failed: {e}. "
                     "Ensure you are running inside the index-tts uv environment. "
                     "Run: uv sync --all-extras && uv run python indextts_bridge.py"}))
    sys.stdout.flush()
    sys.exit(1)


def main():
    global _tts

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"status": "error", "message": f"Invalid JSON: {e}"}))
            sys.stdout.flush()
            continue

        cmd = req.get("cmd", "")

        if cmd == "init":
            try:
                # Suppress model loading noise to keep JSON protocol clean
                import io
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    _tts = IndexTTS2(
                        cfg_path=req["cfg_path"],
                        model_dir=req.get("model_dir", "checkpoints"),
                        use_fp16=req.get("use_fp16", False),
                        use_cuda_kernel=req.get("use_cuda_kernel", False),
                        use_deepspeed=req.get("use_deepspeed", False),
                    )
                finally:
                    sys.stdout = old_stdout
                print(json.dumps({"status": "ok"}))
            except Exception as e:
                sys.stdout = old_stdout
                print(json.dumps({"status": "error", "message": str(e)}))

        elif cmd == "infer":
            if _tts is None:
                print(json.dumps({"status": "error", "message": "Model not initialized"}))
                sys.stdout.flush()
                continue

            try:
                infer_kwargs = {
                    "spk_audio_prompt": req["spk_audio_prompt"],
                    "text": req["text"],
                    "output_path": req.get("output_path", "gen.wav"),
                    "verbose": req.get("verbose", False),
                }

                for opt in ("emo_audio_prompt", "emo_vector", "emo_alpha",
                            "emo_text", "use_emo_text", "use_random"):
                    if opt in req:
                        infer_kwargs[opt] = req[opt]

                import io
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    _tts.infer(**infer_kwargs)
                finally:
                    sys.stdout = old_stdout

                print(json.dumps({
                    "status": "ok",
                    "output_path": infer_kwargs["output_path"],
                }))
            except Exception as e:
                sys.stdout = old_stdout
                print(json.dumps({"status": "error", "message": str(e)}))

        elif cmd == "quit":
            print(json.dumps({"status": "ok"}))
            sys.stdout.flush()
            break

        else:
            print(json.dumps({"status": "error", "message": f"Unknown command: {cmd}"}))

        sys.stdout.flush()


if __name__ == "__main__":
    main()
