#!/usr/bin/env python3
"""Generate a simple development click track WAV for local testing."""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output WAV path.")
    parser.add_argument("--duration", type=float, default=38.4, help="Duration in seconds.")
    parser.add_argument("--bpm", type=float, default=100.0, help="Tempo in BPM.")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_rate = args.sample_rate
    total_frames = int(args.duration * sample_rate)
    beat_spacing_frames = max(1, int((60.0 / args.bpm) * sample_rate))
    click_frames = int(0.045 * sample_rate)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for frame_index in range(total_frames):
            beat_offset = frame_index % beat_spacing_frames
            sample = 0.0
            if beat_offset < click_frames:
                frequency = 1320.0 if (frame_index // beat_spacing_frames) % 4 == 0 else 880.0
                t = beat_offset / sample_rate
                envelope = 1.0 - (beat_offset / max(1, click_frames))
                sample = math.sin(2.0 * math.pi * frequency * t) * envelope * 0.4

            clamped = max(-1.0, min(1.0, sample))
            int_sample = int(clamped * 32767)
            frames.extend(int_sample.to_bytes(2, byteorder="little", signed=True))

        wav_file.writeframes(frames)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
