#!/usr/bin/env python3

# supress SSL warning from cluttering logs
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
warnings.filterwarnings("ignore", module="urllib3")

import sys
import json
import os
from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util


# Register the HEIF opener so Pillow can handle .heic/.heif/.hif
register_heif_opener()

debug = False

# Add this after the debug = False line

def dprint(*args, **kwargs):
    """
    Print debug information only if debug mode is enabled.
    Usage: dprint("Some debug info", var1, var2)
    """
    if debug:
        print("[DEBUG]", *args, **kwargs)

# ----------------------------------------
# 1) SETUP: BLIP for captioning + Sentence-BERT for similarity
# ----------------------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else "cpu"

dprint(f"Using device: {device}")

# BLIP model + processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Sentence-BERT model for text similarity
similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# Your 10 photography prompts
# tag_prompts = [
#     "A close-up portrait emphasizing the subject’s expression and personality.",
#     "A dramatic cityscape showcasing skyscrapers, bridges, and urban design.",
#     "A sweeping natural vista with mountains, forests, or coastlines.",
#     "A candid street scene capturing real-life moments and everyday urban life.",
#     "A moody nighttime shot featuring light trails or illuminated city streets.",
#     "An extreme close-up revealing fine details, textures, or abstract patterns.",
#     "A high-energy moment freezing intense motion in a sporting event.",
#     "A stylish editorial scene focusing on clothing, models, and trendy settings.",
#     "A surreal or experimental composition playing with light, form, or symbolism.",
#     "A vibrant travel scene featuring cultural landmarks, bustling markets, or scenic wonders."
# ]
categories = [
    (
        "Landscape",
        "A photo of a landscape. There could be mountains, forests, or coastlines. All taken at a distance."
    ),
    (
        "Night",
        "A photo at night, outside, illuminated with city lights"
    ),
    (
        "Winter Snow",
        "A photo of the outdoors in the winter. Outside, snow covering the ground or environment."
    ),
    (
        "Urban Downtown",
        "A photo of urban downtown life, with skyscrapers or bustling streets."
    ),
    (
        "Silhouette",
        "A silhouette photo, where the subject is dark against a bright background."
    ),
    (
        "Portrait",
        "A portrait photo focusing on a person’s face or upper body."
    ),
    (
        "Closeup Nature",
        "A photo of close-up natural elements, like leaves, insects, or small details."
    ),
    (
        "Patterns/Detail",
        "A photo emphasizing detail, patterns, or repetition in textures."
    ),
    (
        "Fall Colors",
        "A photo showing autumn foliage with red, orange, or yellow leaves."
    ),
    (
        "Garden/Flowers",
        "A photo of a garden or flowers in bloom."
    ),
    (
        "Bodies of Water",
        "A photo including lakes, rivers, or other bodies of water."
    ),
    (
        "Oscar",
        "A photo of a housecat"
    ),
    (
        "Suburbs",
        "A photo highlighting the suburbs. The photo is predominantly small residential houses."
    ),
    (
        "Lights & Shadow",
        "A photo that focuses predominantly on lighting and harsh shadows"
    ),
]

# Separate out the prompts for embedding
tag_names = [c[0] for c in categories]
tag_prompts = [c[1] for c in categories]

# Pre-encode the category prompts for faster matching
with torch.no_grad():
    category_embeddings = similarity_model.encode(tag_prompts, convert_to_tensor=True)

# ----------------------------------------
# 2) FUNCTION: Generate BLIP caption
# ----------------------------------------
def generate_caption(image_path: str) -> str:
    """
    Generates a caption using BLIP. Returns a string.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        # caption_ids = blip_model.generate(**inputs)
        # caption_ids = blip_model.generate(
        #     **inputs,
        #     num_beams=5,          # beam search with 5 beams
        #     max_length=100,        # allow a longer caption
        #     min_length=30,        # force at least 10 tokens
        #     repetition_penalty=2.0  # discourage repeated phrases
        # )
        caption_ids = blip_model.generate(
            **inputs,
            num_beams=7,
            max_length=100,
            min_length=30,
            repetition_penalty=2.5,
            no_repeat_ngram_size=2,
            # temperature=0.7
        )
        # caption_ids = blip_model.generate(
        #     **inputs,
        #     do_sample=True,
        #     num_beams=1,           # or keep beams, but do_sample means "beam search + sampling"
        #     temperature=0.7,
        #     max_length=60,
        #     min_length=15,
        #     repetition_penalty=2.5,
        #     no_repeat_ngram_size=2
        # )

    caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)

    dprint(f"At Path: {image_path}, Generated caption: {caption}")

    return caption

# ----------------------------------------
# 3) FUNCTION: Match caption to best category
# ----------------------------------------
def match_caption_to_category(caption: str):
    """
    Uses Sentence-BERT to find the single best matching category.
    Returns (best_category, similarity_score).
    """
    # Encode the caption
    caption_embedding = similarity_model.encode([caption], convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.cos_sim(caption_embedding, category_embeddings)  # shape: [1, len(tag_prompts)]
    similarities = similarities.squeeze(0)  # shape: [len(tag_prompts)]

    best_idx = int(torch.argmax(similarities))
    best_score = float(similarities[best_idx])
    best_tag_name = tag_names[best_idx]  # We return the short label, not the full prompt

    return best_tag_name, best_score

# ----------------------------------------
# 4) FUNCTION: Classify a single image (BLIP + match)
# ----------------------------------------
def classify_image(image_path: str):
    """
    Returns a single best matching category in the format:
    [ (category, score) ]
    or empty [] if we skip the file.
    """
    # Skip .raf
    if image_path.lower().endswith(".raf"):
        return []

    # Attempt to open and caption
    try:
        caption = generate_caption(image_path)
    except UnidentifiedImageError:
        # If PIL can't open it, skip
        return []
    except Exception:
        # Any other error, skip
        return []

    # Match caption to the best category
    best_category, score = match_caption_to_category(caption)

    dprint(f"Best category: {best_category}, Score: {score}")
    dprint()

    # Return a list with one (category, score) tuple
    return [(best_category, score)]

# ----------------------------------------
# 5) MAIN: JSON input -> JSON output
# ----------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python blip_multilabel.py /path/to/photo_paths.json")
        sys.exit(1)

    json_file = sys.argv[1]

    # Load the list of image paths from JSON
    with open(json_file, "r") as f:
        image_paths = json.load(f)

    results = []
    for path in image_paths:
        tags = classify_image(path)
        results.append({"image_path": path, "tags": tags})

    # Output one JSON array with all results
    print(json.dumps(results))
    # del similarity_model
    # del blip_model
    # import gc
    # gc.collect()
    # sys.exit(0)
    # os._exit(0)

if __name__ == "__main__":
    main()
