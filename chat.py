"""Simple offline chat with your downloaded MLX vision models.

Works on airplane mode — no internet needed after first download.

Usage:
    python chat.py
    python chat.py --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit
    python chat.py --model mlx-community/pixtral-12b-2409-4bit

Commands inside chat:
    /image path/to/photo.png   — attach an image for the next question
    /clear                     — clear image and start fresh
    /model                     — show current model name
    /quit                      — exit
"""

import argparse
import sys

try:
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_image
except ImportError:
    sys.exit("pip install mlx-vlm")

DEFAULT_MODEL = "vikhyatk/moondream2"


def main():
    parser = argparse.ArgumentParser(description="Offline MLX Chat")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, processor = load(args.model)
    print("Ready! Type your message. /image to attach a photo, /quit to exit.\n")

    image = None
    image_path = None

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user:
            continue

        if user == "/quit":
            print("Bye!")
            break

        if user == "/clear":
            image = None
            image_path = None
            print("  Image cleared.\n")
            continue

        if user == "/model":
            print(f"  {args.model}\n")
            continue

        if user.startswith("/image"):
            path = user[6:].strip()
            if not path:
                print("  Usage: /image path/to/photo.png\n")
                continue
            try:
                image = load_image(path)
                image_path = path
                print(f"  Image loaded: {path}")
                print("  Now type your question about it.\n")
            except Exception as e:
                print(f"  Failed to load image: {e}\n")
            continue

        prompt = user
        if image is not None:
            prompt = "<image>\n" + user

        try:
            result = generate(
                model, processor, prompt, image,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                verbose=False,
            )
            text = result.text if hasattr(result, "text") else str(result)
            print(f"\nBot: {text}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
