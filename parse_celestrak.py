import os
import json
import re

content_path = r"C:\Users\Hp\.gemini\antigravity-ide\brain\84c5050e-7584-4d18-8239-639dfd4031d0\.system_generated\steps\56\content.md"

if os.path.exists(content_path):
    with open(content_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Extract JSON part starting after '---'
    json_start = text.find("[{")
    if json_start != -1:
        json_str = text[json_start:]
        data = json.loads(json_str)
        
        output_file = "celestrak_active.json"
        with open(output_file, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2)
            
        print(f"Successfully processed {len(data)} objects.")
        print(f"Saved to '{output_file}'.")
        print("\n--- Sample NORAD Data ---")
        for obj in data[:5]:
            print(f"NORAD ID: {obj.get('NORAD_CAT_ID')} | Name: {obj.get('OBJECT_NAME')} | Inclination: {obj.get('INCLINATION')}° | Period: {1440/obj.get('MEAN_MOTION', 1):.2f} mins")
    else:
        print("Could not find JSON array in cached content.")
else:
    print(f"File not found: {content_path}")
