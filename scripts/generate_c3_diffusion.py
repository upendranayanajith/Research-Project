"""
scripts/generate_c3_diffusion.py
=================================
[T3.1] Diffusion-based synthetic clock data generator.

Uses the Gemini Imagen API (or an alternative diffusion backend) to generate
diverse synthetic clock face images for augmenting the C3 training dataset.

Motivation:
  The C3 model currently trains on real clock crops extracted by C2 from
  existing annotated images. This limits diversity to available clock styles.
  Synthetic data from diffusion models can:
    1. Cover rare times (e.g. 3:17, 11:53) with few real examples
    2. Generate diverse clock styles (antique, minimalist, Roman numerals)
    3. Augment training set without additional manual labelling

Pipeline:
  1. Configure time slots and styles to generate
  2. Call Gemini Imagen API with a clock-specific prompt
  3. Save generated images with angle-encoded filenames (matching C3 dataset format)
  4. Optionally run C2 pipeline to verify keypoints are detectable

Usage:
    python scripts/generate_c3_diffusion.py \\
        --times_per_hour 5 \\
        --styles all \\
        --output_dir data/c3_synthetic \\
        --n_per_slot 3

    # Preview prompts without generating (dry run):
    python scripts/generate_c3_diffusion.py --dry-run

Requirements:
    pip install google-genai Pillow
    GEMINI_API_KEY must be set in .env or environment.
"""

import os
import sys
import argparse
import math
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────
STYLE_PROMPTS = {
    "modern":      "modern minimalist analog clock, clean white dial, simple black hands, no numerals",
    "classic":     "classic wall clock, traditional design, bold black Arabic numerals, elegant hands",
    "antique":     "antique ornate clock, Roman numerals, decorative hands, aged brass frame, vintage dial",
    "digital_analog": "hybrid clock, digital-style analog, thin hands, dot markers instead of numerals",
    "pocket":      "antique pocket watch face, intricate filigree, small seconds subdial",
}

BACKGROUND_VARIANTS = [
    "white background",
    "light grey background",
    "on a wooden table",
    "studio lighting, product photo",
]


def time_to_hand_angles(h: int, m: int) -> tuple:
    """
    Compute hour and minute hand angles from 12 o'clock (clockwise).
    Returns (hour_angle_deg, minute_angle_deg).
    """
    minute_angle = m * 6.0          # 360° / 60 min = 6°/min
    hour_angle   = (h % 12) * 30.0 + m * 0.5   # 30°/hr + 0.5°/min
    return round(hour_angle, 2), round(minute_angle, 2)


def build_prompt(h: int, m: int, style: str, background: str = "white background") -> str:
    """Build an Imagen-optimised prompt for a clock showing time H:MM."""
    style_desc = STYLE_PROMPTS.get(style, STYLE_PROMPTS["modern"])
    time_str   = f"{h}:{m:02d}"
    return (
        f"Photorealistic {style_desc}, showing exact time {time_str}, "
        f"{background}, sharp focus, high resolution, straight-on front view, "
        f"no text overlay, no decorative frame cropping hands, "
        f"clock hands clearly visible and unambiguous"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Time slot generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_time_slots(times_per_hour: int = 4) -> list:
    """
    Generate a list of (h, m) tuples to cover all hours.

    Args:
        times_per_hour: How many time slots per hour (evenly spaced).
    Returns:
        List of (hour, minute) tuples covering 12 hours.
    """
    slots = []
    step  = 60 // times_per_hour
    for h in range(1, 13):
        for m in range(0, 60, step):
            slots.append((h, m))
    return slots


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Imagen generator
# ─────────────────────────────────────────────────────────────────────────────
class GeminiImagenGenerator:
    """
    Wraps the Gemini Imagen 3 API to generate clock images.

    Falls back to a placeholder PNG if the API is unavailable.
    """

    MODEL_ID = "imagen-3.0-generate-001"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.available = False

        if self.api_key:
            try:
                from google import genai
                self.client    = genai.Client(api_key=self.api_key)
                self.available = True
                print("✅ Gemini Imagen API connected")
            except ImportError:
                print("⚠️  google-genai not installed. Run: pip install google-genai")
            except Exception as e:
                print(f"⚠️  Gemini Imagen init failed: {e}")
        else:
            print("⚠️  GEMINI_API_KEY not set — dry-run mode only")

    def generate(self, prompt: str, n: int = 1) -> list:
        """
        Generate n images from a prompt.

        Returns:
            List of PIL.Image objects, or empty list on failure.
        """
        if not self.available:
            return []

        try:
            from google.genai import types as genai_types
            response = self.client.models.generate_images(
                model=self.MODEL_ID,
                prompt=prompt,
                config=genai_types.GenerateImagesConfig(
                    number_of_images=n,
                    aspect_ratio="1:1",
                    safety_filter_level="BLOCK_ONLY_HIGH",
                ),
            )
            images = []
            for img in response.generated_images:
                from PIL import Image
                import io
                pil = Image.open(io.BytesIO(img.image.image_bytes))
                images.append(pil)
            return images
        except Exception as e:
            print(f"  ⚠️ Generation failed: {e}")
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder fallback (when API not available)
# ─────────────────────────────────────────────────────────────────────────────
def create_placeholder_clock(h: int, m: int, size: int = 224):
    """
    Draw a simple clock face programmatically as a PIL Image.
    Used as fallback when Imagen API is unavailable.
    """
    try:
        from PIL import Image, ImageDraw
        import math as _math

        img  = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(img)
        cx, cy, r = size // 2, size // 2, size // 2 - 10

        # Clock face
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="black", width=3)

        # Hour markers
        for i in range(12):
            angle = _math.radians(i * 30 - 90)
            x1 = cx + (r - 12) * _math.cos(angle)
            y1 = cy + (r - 12) * _math.sin(angle)
            x2 = cx + r * _math.cos(angle)
            y2 = cy + r * _math.sin(angle)
            draw.line([x1, y1, x2, y2], fill="black", width=2)

        # Hour hand
        h_angle, m_angle = time_to_hand_angles(h, m)
        for angle_deg, length, color, width in [
            (h_angle,  r * 0.55, "black", 4),   # hour
            (m_angle,  r * 0.80, "black", 2),   # minute
        ]:
            rad = _math.radians(angle_deg - 90)
            x2  = cx + length * _math.cos(rad)
            y2  = cy + length * _math.sin(rad)
            draw.line([cx, cy, x2, y2], fill=color, width=width)

        draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill="black")
        return img

    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────────────────────────────────────
def generate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator  = GeminiImagenGenerator()
    time_slots = generate_time_slots(args.times_per_hour)
    styles     = list(STYLE_PROMPTS.keys()) if args.styles == "all" else [args.styles]
    bgs        = BACKGROUND_VARIANTS

    manifest   = []
    n_generated = 0
    n_skipped   = 0

    print(f"\n[T3.1] Diffusion Data Generation")
    print(f"  Time slots: {len(time_slots)} | Styles: {styles} | N per slot: {args.n_per_slot}")
    print(f"  Output: {output_dir}\n")

    for h, m in time_slots:
        h_angle, m_angle = time_to_hand_angles(h, m)

        for style in styles:
            bg = bgs[(h * len(styles) + styles.index(style)) % len(bgs)]
            prompt = build_prompt(h, m, style, bg)

            if args.dry_run:
                print(f"  [DRY] {h}:{m:02d} [{style}] → H={h_angle}°, M={m_angle}°")
                print(f"        Prompt: {prompt[:90]}...")
                continue

            print(f"  Generating {h}:{m:02d} [{style}] ({args.n_per_slot} images)...", end=" ", flush=True)

            # Try Imagen API first, then fallback to programmatic
            images = generator.generate(prompt, n=args.n_per_slot)
            if not images:
                images = [create_placeholder_clock(h, m) for _ in range(args.n_per_slot)]
                if images[0] is None:
                    print("SKIP (Pillow not available)")
                    n_skipped += args.n_per_slot
                    continue
                print("(placeholder)", end=" ")

            for i, img in enumerate(images):
                if img is None:
                    continue
                # Filename convention: "{hour_angle:.1f}_{minute_angle:.1f}_{style}_{idx:03d}.jpg"
                fname = f"{h_angle:.1f}_{m_angle:.1f}_{style}_{i:03d}.jpg"
                fpath = output_dir / fname
                img.resize((128, 128)).save(fpath, "JPEG", quality=95)
                n_generated += 1
                manifest.append({
                    "file": fname, "hour": h, "minute": m,
                    "hour_angle": h_angle, "minute_angle": m_angle,
                    "style": style, "source": "imagen" if generator.available else "placeholder"
                })

            print(f"OK ({len(images)} saved)")
            time.sleep(0.5)   # Rate limiting

    if not args.dry_run:
        # Write manifest JSON
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"\n  Generated: {n_generated} | Skipped: {n_skipped}")
        print(f"  Manifest:  {manifest_path}")
        print(f"\n  ✅ Done! Add {output_dir} to your C3 training data directory.")
    else:
        print(f"\n  [DRY RUN] Would generate {len(time_slots) * len(styles) * args.n_per_slot} images")
        print(f"  Re-run without --dry-run to generate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="[T3.1] Generate synthetic clock images using Gemini Imagen")
    parser.add_argument("--times_per_hour", type=int, default=4,        help="Time slots per hour (4 = every 15 min)")
    parser.add_argument("--styles",         default="all",               help="Style key or 'all'")
    parser.add_argument("--output_dir",     default="data/c3_synthetic", help="Output directory")
    parser.add_argument("--n_per_slot",     type=int, default=2,        help="Images per (time, style) slot")
    parser.add_argument("--dry-run",        action="store_true",         help="Print prompts without generating")
    args = parser.parse_args()
    generate(args)
