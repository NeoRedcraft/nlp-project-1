import json
import os

notebook_path = 'Final.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for cell in data['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Replace the model name
            if 'embedding_model_name' in line and ('gte-large-en' in line or 'gte-large-en-v1.5' in line):
                line = line.replace('Alibaba-NLP/gte-large-en-v1.5', 'BAAI/bge-large-en-v1.5')
                line = line.replace('Alibaba-NLP/gte-large-en', 'BAAI/bge-large-en-v1.5')
            
            # Remove the trust_remote_code=True since it's not needed and can cause issues
            if 'model_kwargs' in line and 'trust_remote_code' in line:
                continue # Skip this line
            
            # If the previous line ended with a comma and the next line is removed, we might need to strip trailing commas
            # In python, `model_name=embedding_model_name,\n` is fine even if it's the last argument before a closing parenthesis in some cases (though better to be clean). 
            # In the original, it was:
            # "        model_name=embedding_model_name,\n",
            # "        model_kwargs={'trust_remote_code': True}\n",
            # So if we skip the model_kwargs, `model_name=embedding_model_name,\n` becomes `model_name=embedding_model_name\n`
            if 'model_name=embedding_model_name' in line:
                line = line.replace(',', '')
                
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("Final.ipynb updated successfully.")
