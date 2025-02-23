import torch
import clip
from PIL import Image, UnidentifiedImageError
import sys
import json

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Predefined categories for tagging
# categories = ['Portrait', 'Landscape', 'Cityscape', 'Night Photography', 'Aerial Photography', 'Macro Photography', 'Street Photography', 'Documentary Photography', 'Fine Art Photography', 'Black and White Photography', 'Wildlife', 'Birds', 'Underwater Photography', 'Forest', 'Desert', 'Mountains', 'Beach', 'Waterfalls', 'Flowers', 'Sunsets & Sunrises', 'Fashion Photography', 'Sports Photography', 'Wedding Photography', 'Concert Photography', 'Candid Photography', 'Group Photos', 'Selfies', 'Dance & Performance', 'Fitness Photography', 'Lifestyle Photography', 'Architecture', 'Interiors', 'Bridges', 'Skyscrapers', 'Abandoned Places', 'Street Art & Graffiti', 'Transportation', 'Markets & Street Vendors', 'Industrial Photography', 'Historical Landmarks', 'Minimalist Photography', 'Abstract Photography', 'Silhouettes', 'Shadows & Light Play', 'Reflections', 'Motion Blur', 'Textures & Patterns', 'Long Exposure Photography', 'Still Life Photography', 'Surreal Photography']
# categories = ['Home', 'Nature', 'Gay', 'Man', 'Day to Day Mundane', 'Architecture', 'Minimalism', 'Motion Blur and Shutter Drag', 'Nature', 'Landscapes', 'Birds and Wildlife', 'Snow', 'Winter', 'Spring', 'Fall', 'Monochrome', 'Multicolor', 'Cityscape', 'Strangers', 'Portrait']

# Creating an array with all 50 CLIP-optimized scene descriptions
# pool_of_tags = [
#     "portrait", "selfie", "group photo", "wedding", "party",
#     "landscape", "mountain", "desert", "forest", "beach",
#     "cityscape", "skyscraper", "historic building", "bridge", "road",
#     "street art", "graffiti", "cars", "bicycle", "train station",
#     "night photography", "sunset", "sunrise", "overcast day", "rainy day",
#     "sports", "basketball", "soccer", "running", "fitness",
#     "wildlife", "bird", "dog", "cat", "horse",
#     "macro", "flowers", "insect", "snowflake", "texture",
#     "abstract", "minimalist", "black and white", "fine art", "surreal",
#     "underwater", "aerial", "drone shot", "interiors", "architecture",
#     "concert", "festival", "crowd", "market", "people walking",
#     "food", "coffee", "dessert", "fruit", "vegetables",
#     "technology", "computer", "smartphone", "electronics", "robotics",
#     "fashion", "clothing", "shoes", "runway", "model",
#     "night sky", "stars", "milky way", "moon", "light trails",
#     "street photography", "documentary", "protest", "musician", "performance",
#     "family", "children", "couple", "elderly", "candid",
#     "lifestyle", "home interior", "kitchen", "living room", "office space",
#     "sculpture", "painting", "museum", "gallery", "exhibition",
#     "historic monument", "castle", "palace", "temple", "church",
#     "futuristic", "cyberpunk", "neon lights", "motion blur", "long exposure"
# ]

pool_of_tags = ["Portrait", "Selfie", "Group Photo", "Drone Shot", "Wedding", "Party", "Family", "Children", "Couple", "Fashion", "Model", "Street Photography", "Candid", "Lifestyle", "Concert", "Festival", "Nightlife", "Sports", "Fitness", "Action", "Macro", "Flowers", "Insect", "Snowflake", "Texture", "Abstract", "Minimalist", "Still Life", "Product", "Food", "Cuisine", "Dessert", "Coffee", "Drink", "Technology", "Computer", "Smartphone", "Robotics", "Electronics", "Architecture", "Interior", "Furniture", "Design", "Office", "Vehicle", "Car", "Motorcycle", "Bicycle", "Aircraft", "Ship", "Train", "Drone", "Underwater", "Scuba", "Snorkeling", "Landscape", "Mountain", "Desert", "Forest", "Beach", "Cityscape", "Skyscraper", "Bridge", "Historic Building", "Street Art", "Graffiti", "Alleyway", "Market", "Night Photography", "Sunrise", "Sunset", "Storm", "Fog", "Snow", "Rain", "Rainbow", "Lightning", "Astrophotography", "Milky Way", "Star Trails", "Moon", "Wildlife", "Bird", "Dog", "Cat", "Butterfly", "Bees", "Farm", "Agriculture", "Camping", "Hiking", "Backpacking", "Travel", "Tourism", "Waterfall", "Island", "Canyon", "Cliff", "Glacier", "Volcano", "Cave", "Lake", "River", "Coast", "Prairie", "Jungle", "Meadow", "Patterns", "Silhouette", "Reflection", "Bokeh", "High Key", "Low Key", "Black and White", "Vintage", "Retro", "Film", "Experimental", "Infrared", "Long Exposure", "Motion Blur", "Light Trails", "Light Painting", "Double Exposure", "Panorama", "Collage", "Diptych", "Triptych", "Self-Portrait", "Surreal", "Fantasy", "Cosplay", "Boudoir", "Editorial", "Magazine", "Advertising", "Branding", "Social Media", "Influencer", "Event", "Skateboarding", "Surfing", "Snowboarding", "Skiing", "Sculpture", "Museum", "Gallery", "Exhibition", "Home Interior", "Kitchen", "Living Room", "Street Scene", "Photojournalism", "Fine Art", "Street Style", "Glamour", "Headshot", "Documentary", "Golden Hour", "Blue Hour", "Overcast", "Protest", "Market Stall", "Busker", "Festival Crowd", "Fireworks", "Reflection Pool", "Urban Skyline", "Traffic Jam", "Painted Mural", "Farmers Market", "Hot Air Balloon", "Vineyard", "Lighthouse"]

# ----------------------------
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