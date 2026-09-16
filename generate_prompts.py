class_names_emotic = [
    'Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 
    'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 
    'Esteem', 'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace', 'Pleasure', 
    'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning'
]

hard_prompts = {
    'Yearning': "looking into the distance with a wistful expression, longing",
    'Embarrassment': "averting gaze, blushing or covering face in shame",
    'Fear': "with wide eyes, open mouth in terror, tense posture",
    'Surprise': "with raised eyebrows, gasping, widened eyes",
    'Aversion': "turning head away in disgust, wrinkling nose",
    'Affection': "showing fond attachment, devotion, or love",
    'Anger': "expressing strong displeasure or hostility, frowning",
    'Annoyance': "showing mild anger, irritation, or nuisance",
    'Anticipation': "looking forward to an upcoming event eagerly",
    'Confidence': "appearing self-assured and trusting their abilities",
    'Disapproval': "expressing condemnation or a negative opinion",
    'Disconnection': "seeming detached, isolated, looking away or unresponsive",
    'Disquietment': "appearing restless, uneasy, or anxious",
    'Doubt/Confusion': "appearing uncertain, puzzled, or lacking conviction",
    'Engagement': "paying close attention, interacting or being deeply involved",
    'Esteem': "showing respect, admiration, or a favorable opinion",
    'Excitement': "appearing thrilled, eager, or highly enthusiastic",
    'Fatigue': "showing physical or mental exhaustion and weariness, slouching",
    'Happiness': "expressing joy, contentment, or pleasure, smiling",
    'Pain': "showing physical suffering or distress, grimacing",
    'Peace': "appearing calm, tranquil, relaxed and free from disturbance",
    'Pleasure': "showing delight, enjoyment, or satisfaction",
    'Sadness': "expressing sorrow, unhappiness, or grief, crying or frowning",
    'Sensitivity': "showing a delicate or acute emotional response",
    'Suffering': "enduring severe physical or mental pain, crying or holding face",
    'Sympathy': "showing compassion or understanding for others"
}

prefixes = [
    "a photo of a person",
    "a video of a person",
    "a portrait of a person",
    "an image showing a person",
    "a close-up of a person",
    "a face of a person",
    "a picture of a person",
    "a shot of a person"
]

print("prompt_ensemble_emotic_26 = [")
for cls in class_names_emotic:
    desc = hard_prompts[cls]
    print("    [")
    for pref in prefixes:
        prompt = f"{pref} {desc}, feeling {cls.lower()}."
        print(f'        "{prompt}",')
    print("    ],")
print("]")
