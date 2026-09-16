import re

with open("models/Text.py", "r") as f:
    content = f.read()

with open("prompts_output.txt", "r") as f:
    new_prompts = f.read()

# Find the start of prompt_ensemble_emotic_26 and replace the whole loop
pattern = r"# Simple prompt ensemble for EMOTIC\nprompt_ensemble_emotic_26 = \[\]\nfor cls in class_names_emotic:\n    prompt_ensemble_emotic_26\.append\(\[\n(?:        f\"[^\"]*\",\n)+    \]\)\n"
replacement = f"# Rich prompt ensemble for EMOTIC v5\n{new_prompts}"

new_content, count = re.subn(pattern, replacement, content)

if count > 0:
    with open("models/Text.py", "w") as f:
        f.write(new_content)
    print("Successfully replaced prompt_ensemble_emotic_26")
else:
    print("Failed to match the regex pattern.")
