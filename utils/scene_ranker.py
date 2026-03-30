import json

def load_scene_scores(path):
    with open(path, "r") as f:
        return json.load(f)
    
def rank_scenes(scene_data):
    return sorted(
        scene_data,
        key = lambda x : x["final"],
        reverse= True
    )
    
def select_top_scenes(ranked_scenes, top_n=5):
    return ranked_scenes[:top_n]
    
def select_by_threshold(ranked_scenes, threshold = 0.5):
    return [scene for scene in ranked_scenes if scene["final"] >= threshold]

def restore_timeline_order(selected_scenes):
    return sorted(selected_scenes, key=lambda x: x["scene_id"])

def get_ranked_scenes(scene_data, threshold=0.5):

    ranked = rank_scenes(scene_data)
    selected = select_by_threshold(ranked, threshold)
    ordered = restore_timeline_order(selected)
    return ordered

def save_selected_scenes(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def extract_scene_ids(selected_scenes):
    return [int(scene["scene_id"]) for scene in selected_scenes]


def main():
    scene_data = load_scene_scores("data/fused_scores.json")

    selected_scenes = get_ranked_scenes(scene_data, threshold=0.3)
    selected_scene_ids = extract_scene_ids(selected_scenes)

    save_selected_scenes(selected_scenes, "data/selected_scenes.json")
    save_selected_scenes(selected_scene_ids, "data/ranked_scene_ids.json")

    print("Selected Scenes:")

    for scene in selected_scenes:
        print(scene)

    print("Saved selected scenes to data/selected_scenes.json")
    print("Saved ranked scene IDs to data/ranked_scene_ids.json")

if __name__ == "__main__":
    main()