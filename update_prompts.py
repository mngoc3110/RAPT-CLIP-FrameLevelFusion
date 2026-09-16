import re

exact_defs = [
    "fond feelings, love, or tenderness",
    "intense displeasure or rage, furious, or resentful",
    "bothered by something or someone, irritated, impatient, or frustrated",
    "state of looking forward, hoping on or getting prepared for possible future events",
    "feeling disgust, dislike, repulsion, or hate",
    "feeling of being certain, conviction that an outcome will be favorable, encouraged, or proud",
    "feeling that something is wrong or reprehensible, contempt, or hostile",
    "feeling not interested in the main event of the surrounding, indifferent, bored, or distracted",
    "nervous, worried, upset, anxious, tense, pressured, or alarmed",
    "difficulty to understand or decide, or thinking about different options",
    "feeling ashamed or guilty",
    "paying attention to something, absorbed into something, curious, or interested",
    "feelings of favorable opinion or judgment, respect, admiration, or gratefulness",
    "feeling enthusiasm, stimulated, or energetic",
    "weariness, tiredness, or sleepy",
    "feeling suspicious or afraid of danger, threat, evil or pain, or horror",
    "feeling delighted, or feeling enjoyment or amusement",
    "physical suffering",
    "well being and relaxed, no worry, having positive thoughts or sensations, or satisfied",
    "feeling of delight in the senses",
    "feeling unhappy, sorrow, disappointed, or discouraged",
    "feeling of being physically or emotionally wounded, or feeling delicate or vulnerable",
    "psychological or emotional pain, distressed, or anguished",
    "sudden discovery of something unexpected",
    "state of sharing others emotions, goals or troubles, supportive, or compassionate",
    "strong desire to have something, jealous, envious, or lust"
]

class_names = [
    'affection', 'anger', 'annoyance', 'anticipation', 'aversion', 'confidence', 'disapproval', 
    'disconnection', 'disquietment', 'doubt/confusion', 'embarrassment', 'engagement', 
    'esteem', 'excitement', 'fatigue', 'fear', 'happiness', 'pain', 'peace', 'pleasure', 
    'sadness', 'sensitivity', 'suffering', 'surprise', 'sympathy', 'yearning'
]

templates = [
    "a photo of a person feeling {cls}: {desc}.",
    "a video of a person feeling {cls}: {desc}.",
    "a portrait of a person feeling {cls}: {desc}.",
    "an image showing a person feeling {cls}: {desc}.",
    "a close-up of a person feeling {cls}: {desc}.",
    "a face of a person feeling {cls}: {desc}.",
    "a picture of a person feeling {cls}: {desc}.",
    "a shot of a person feeling {cls}: {desc}.",
]

ensemble = []
for i in range(26):
    prompts = [t.format(cls=class_names[i], desc=exact_defs[i]) for t in templates]
    ensemble.append(prompts)

with open('/Users/macbook/Downloads/RAPT-CLIP/models/Text.py', 'r') as f:
    content = f.read()

# Replace the ensemble part
new_ensemble_str = "prompt_ensemble_emotic_26 = [\n"
for prompts in ensemble:
    new_ensemble_str += "    [\n"
    for p in prompts:
        new_ensemble_str += f'        "{p}",\n'
    new_ensemble_str += "    ],\n"
new_ensemble_str += "]\n"

import re
content = re.sub(r'prompt_ensemble_emotic_26\s*=\s*\[(.*?)\]\n(?!\s*\])', new_ensemble_str + '\n---MARKER---', content, flags=re.DOTALL)
# It's safer to just replace the lines manually since regex with nested lists is tricky.
