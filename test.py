'''
Question 9: Batch pipeline: Make a Python script (not Notebook!) that takes a text file with
image file names as input, a “model” file as another input. It processes images
one-by-one keeping the results, then loads the model, updates it with all
results, and saves it back
'''
import sys
import os
import json
import io
import base64
from PIL import Image
import ollama

def process_image(image_path, model_name="embeddinggemma"):
    """
    Process a single image and return its embedding vector.
    Converts the image to base64 and sends it to the Ollama embed API.
    """
    try:
        # Convert image to base64
        with Image.open(image_path) as img:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Send base64 image to ollama.embed
        response = ollama.embed(
            model=model_name,
            input=img_base64
        )

        # Extract embeddings
        embeddings = response.get("embeddings")
        if not embeddings or not isinstance(embeddings, list):
            raise ValueError("No embeddings received from Ollama.")

        # If a list of lists, take first (single item)
        vector = embeddings[0] if isinstance(embeddings[0], list) else embeddings
        return vector

    except Exception as e:
        print(f"[ERROR] Processing {image_path}: {e}")
        return None


def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_update_model.py <image_list.txt> <model_file.json>")
        sys.exit(1)

    image_list_file = sys.argv[1]
    model_file = sys.argv[2]

    if not os.path.exists(image_list_file):
        print(f"[ERROR] Image list file does not exist: {image_list_file}")
        sys.exit(1)

    # Load or create JSON model
    if os.path.exists(model_file):
        try:
            with open(model_file, "r") as f:
                model = json.load(f)
                if not isinstance(model, dict):
                    print(f"[WARNING] Model file is not a dict — reinitializing.")
                    model = {}
        except Exception as e:
            print(f"[WARNING] Failed to read model JSON, starting fresh: {e}")
            model = {}
    else:
        model = {}

    # Read image paths
    with open(image_list_file, "r") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    print(f"[INFO] Found {len(image_paths)} images to process.")

    results = {}
    for i, img_path in enumerate(image_paths, 1):
        embedding = process_image(img_path)
        if embedding:
            results[img_path] = embedding
        print(f"[INFO] Processed {i}/{len(image_paths)}: {img_path}")

    print(f"[INFO] Successfully processed {len(results)} embeddings.")

    model.update(results)

    with open(model_file, "w") as f:
        json.dump(model, f)

    print(f"[INFO] Saved updated model to {model_file}")


if __name__ == "__main__":
    main()
