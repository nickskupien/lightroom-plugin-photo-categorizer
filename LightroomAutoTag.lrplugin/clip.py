import torch
import clip
from PIL import Image, UnidentifiedImageError
import sys
import json

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

pool_of_tags = ["A close-up portrait emphasizing the subject’s expression and personality.", "A dramatic cityscape showcasing skyscrapers, bridges, and urban design.", "A sweeping natural vista with mountains, forests, or coastlines.", "A candid street scene capturing real-life moments and everyday urban life.", "A moody nighttime shot featuring light trails or illuminated city streets.", "An extreme close-up revealing fine details, textures, or abstract patterns.", "A high-energy moment freezing intense motion in a sporting event.", "A stylish editorial scene focusing on clothing, models, and trendy settings.", "A surreal or experimental composition playing with light, form, or symbolism.", "A vibrant travel scene featuring cultural landmarks, bustling markets, or scenic wonders."]

# 3) Precompute text embeddings for the 100 tags
# ----------------------------
text_tokens = clip.tokenize(pool_of_tags).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    # Normalize text features to unit length
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

def classify_image(image_path, top_k=10, threshold=0.25):
    """Classifies an image using CLIP and returns the best matching categories."""

    # ----------------------------
    # Check if the image is a RAW file, and skip
    # ----------------------------
    if image_path.lower().endswith(".raf"):
            return []

    # ----------------------------
    # 4) Load and preprocess the image
    # ----------------------------
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        # Can't open this file with Pillow, so skip or return empty
        return []

    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode image and normalize
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarities to each text embedding
        similarities = (image_features @ text_features.T).squeeze(0)

        # Get top_k matches
        best_matches = similarities.topk(top_k)
        top_indices = best_matches.indices.tolist()
        top_scores = best_matches.values.tolist()

    # Map indices to tag names and scores
    predicted_tags = []
    for idx, score in zip(top_indices, top_scores):
        predicted_tags.append((pool_of_tags[idx], float(score)))

    # ----------------------------
    # 5) Filter tags by threshold (optional)
    # ----------------------------
    filtered_tags = [(tag, s) for (tag, s) in predicted_tags if s > threshold]

    return filtered_tags

def main():
    if len(sys.argv) < 2:
        print("Usage: python clip_multilabel.py /path/to/photo_paths.json [top_k=10] [threshold=0.25]")
        sys.exit(1)

    json_file = sys.argv[1]
    top_k = 10
    threshold = 0.25

    if len(sys.argv) >= 3:
        top_k = int(sys.argv[2])
    if len(sys.argv) >= 4:
        threshold = float(sys.argv[3])

    # Load the list of image paths from JSON
    with open(json_file, "r") as f:
        image_paths = json.load(f)

    results = []
    for path in image_paths:
        tags = classify_image(path, top_k=top_k, threshold=threshold)
        results.append({"image_path": path, "tags": tags})

    # Output one JSON array with all results
    print(json.dumps(results))

if __name__ == "__main__":
    main()