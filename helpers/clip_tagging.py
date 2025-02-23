import torch
import clip
from PIL import Image
import sys
import json

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Predefined categories for tagging
# categories = ['Portrait', 'Landscape', 'Cityscape', 'Night Photography', 'Aerial Photography', 'Macro Photography', 'Street Photography', 'Documentary Photography', 'Fine Art Photography', 'Black and White Photography', 'Wildlife', 'Birds', 'Underwater Photography', 'Forest', 'Desert', 'Mountains', 'Beach', 'Waterfalls', 'Flowers', 'Sunsets & Sunrises', 'Fashion Photography', 'Sports Photography', 'Wedding Photography', 'Concert Photography', 'Candid Photography', 'Group Photos', 'Selfies', 'Dance & Performance', 'Fitness Photography', 'Lifestyle Photography', 'Architecture', 'Interiors', 'Bridges', 'Skyscrapers', 'Abandoned Places', 'Street Art & Graffiti', 'Transportation', 'Markets & Street Vendors', 'Industrial Photography', 'Historical Landmarks', 'Minimalist Photography', 'Abstract Photography', 'Silhouettes', 'Shadows & Light Play', 'Reflections', 'Motion Blur', 'Textures & Patterns', 'Long Exposure Photography', 'Still Life Photography', 'Surreal Photography']
# categories = ['Home', 'Nature', 'Gay', 'Man', 'Day to Day Mundane', 'Architecture', 'Minimalism', 'Motion Blur and Shutter Drag', 'Nature', 'Landscapes', 'Birds and Wildlife', 'Snow', 'Winter', 'Spring', 'Fall', 'Monochrome', 'Multicolor', 'Cityscape', 'Strangers', 'Portrait']

# Creating an array with all 50 CLIP-optimized scene descriptions
categories = [
    "portrait", "selfie", "group photo", "wedding", "party",
    "landscape", "mountain", "desert", "forest", "beach",
    "cityscape", "skyscraper", "historic building", "bridge", "road",
    "street art", "graffiti", "cars", "bicycle", "train station",
    "night photography", "sunset", "sunrise", "overcast day", "rainy day",
    "sports", "basketball", "soccer", "running", "fitness",
    "wildlife", "bird", "dog", "cat", "horse",
    "macro", "flowers", "insect", "snowflake", "texture",
    "abstract", "minimalist", "black and white", "fine art", "surreal",
    "underwater", "aerial", "drone shot", "interiors", "architecture",
    "concert", "festival", "crowd", "market", "people walking",
    "food", "coffee", "dessert", "fruit", "vegetables",
    "technology", "computer", "smartphone", "electronics", "robotics",
    "fashion", "clothing", "shoes", "runway", "model",
    "night sky", "stars", "milky way", "moon", "light trails",
    "street photography", "documentary", "protest", "musician", "performance",
    "family", "children", "couple", "elderly", "candid",
    "lifestyle", "home interior", "kitchen", "living room", "office space",
    "sculpture", "painting", "museum", "gallery", "exhibition",
    "historic monument", "castle", "palace", "temple", "church",
    "futuristic", "cyberpunk", "neon lights", "motion blur", "long exposure"
]


def classify_image(image_path):
    """Classifies an image using CLIP and returns the best matching categories."""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Convert categories to CLIP input format
    text_inputs = clip.tokenize(categories).to(device)

    # Run image and text through CLIP model
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

    # Compute cosine similarity between image and category descriptions
    similarities = (image_features @ text_features.T).squeeze(0)
    best_matches = similarities.topk(3)  # Get top 5 matching categories

    return [categories[i] for i in best_matches.indices]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    tags = classify_image(image_path)
    
    print(json.dumps({"tags": tags}))
