import modelSelection as ext
import json
import re
from json_repair import repair_json

responses, models = ext.get_json_ld()
for i in range(len(models)):
    json_str_match = re.search(r"\{.*\}", responses[i], re.DOTALL)

    if json_str_match:
        json_str = json_str_match.group(0)
        json_str = repair_json(json_str)

        try:
            # Parse the JSON string to ensure its valid JSON
            json_data = json.loads(json_str)

            # Define the file path where you want to save the JSON data
            file_path = 'json/KG_'+models[i].replace("/", "")+'.json'

            # Write the JSON data to a file
            with open(file_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=4)

            print(f"JSON data has been successfully written to {file_path}", models[i])
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e, models[i])
            print("----------------------")
    else:
        print("Could not find a JSON structure in the input text.", models[i])
