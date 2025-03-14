
import os
import json

# for subfolder_name in os.listdir('.'):

for root, _, files in os.walk('.'):

    for filename in files:
        
        # read and resave
        if filename.endswith('_score.json'):

            full_filepath = root + '/' + filename

            with open(full_filepath, 'r') as f:
                data = json.load(f)

            if 'details' in data:
                data['details'] = data['details'][:20]

            
            with open(full_filepath, 'w', encoding='utf-8', ) as f:
                json.dump(data, f, indent=4, ensure_ascii=False)


            
            print(full_filepath, '-----done')

