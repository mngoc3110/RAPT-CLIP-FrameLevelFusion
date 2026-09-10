class_names_5 = [
    "Neutral (student in class).",
    "Enjoyment (student in class).",
    "Confusion (student in class).",
    "Fatigue (student in class).",
    "Distraction (student in class)."
]

class_names_with_context_5 = [
    "A student shows a neutral learning state in a classroom.",
    "A student shows enjoyment while learning in a classroom.",
    "A student shows confusion during learning in a classroom.",
    "A student shows fatigue during learning in a classroom.",
    "A student shows distraction and is not focused in a classroom."
]

class_descriptor_5_only_face = [
    "A student has a neutral face with relaxed mouth, open eyes, and calm eyebrows.",
    "A student looks happy with a slight smile, bright eyes, and relaxed eyebrows.",
    "A student looks confused with furrowed eyebrows, a puzzled look, and slightly open mouth.",
    "A student looks tired with drooping eyelids, frequent yawning, and a sleepy face.",
    "A student looks distracted with unfocused eyes and a wandering gaze away from the lesson."
]

class_descriptor_5_only_body = [
    "A student sits still with an upright posture and hands on the desk, showing a neutral learning state.",
    "A student leans slightly forward with an open, engaged posture, showing enjoyment in learning.",
    "A student tilts the head and leans in, hand on chin, showing confusion while trying to understand.",
    "A student slouches with shoulders dropped and head lowered, showing fatigue during class.",
    "A student shifts around, turns away from the desk, or looks sideways, showing distraction and low focus."
]

class_descriptor_5 = [
    "A student looks neutral and calm in class, with a relaxed face and steady gaze, quietly watching the lecture or reading notes.",
    "A student shows enjoyment while learning, with a gentle smile and bright eyes, appearing engaged and interested in the lesson.",
    "A student looks confused in class, with furrowed eyebrows and a puzzled expression, focusing on the material as if trying to understand.",
    "A student appears fatigued in class, with drooping eyelids and yawning, head slightly lowered, showing low energy.",
    "A student is distracted in class, frequently looking away from the lesson, scanning around, and not paying attention to learning materials."
]

# Prompt Ensemble for RAER (5 classes)
# Each inner list contains multiple descriptions for a single class.
prompt_ensemble_5 = [
    [   # Neutral
        "A photo of a student being alert and looking straight ahead.",
        "A photo of a student with a calm and steady gaze.",
        "A photo of a student paying attention with a neutral expression."
    ],
    [   # Enjoyment
        "A photo of a student smiling and looking happy.",
        "A photo of a student showing joy and enthusiasm.",
        "A photo of a student appearing pleased and engaged."
    ],
    [   # Confusion
        "A photo of a student frowning with a puzzled expression.",
        "A photo of a student scratching their head or looking confused.",
        "A photo of a student trying hard to understand but failing."
    ],
    [   # Fatigue
        "A photo of a student yawning or falling asleep.",
        "A photo of a student with heavy drooping eyelids.",
        "A photo of a student resting their head, looking very tired."
    ],
    [   # Distraction
        "A photo of a student looking away from the screen.",
        "A photo of a student turning their head to the side.",
        "A photo of a student engaging in other activities, not studying."
    ]
]

class_descriptor_5_au = [
    "A student with AU0 neutral face and AU43 eyes normally open, showing neutrality.",
    "A student with AU6 cheek raiser and AU12 lip corner puller, showing enjoyment.",
    "A student with AU4 brow lowerer and AU7 lid tightener, showing confusion.",
    "A student with AU43 eyes closed and AU46 drooping eyelids, showing fatigue.",
    "A student with AU51 head turn left or AU52 head turn right, showing distraction."
]

class_descriptor_8 = [
    'A person who is feeling neutral.',
    'A person who is feeling happy.',
    'A person who is feeling sad.',
    'A person who is feeling surprise.',
    'A person who is feeling fear.',
    'A person who is feeling disgust.',
    'A person who is feeling anger.',
    'A person who is feeling contempt.'
]

class_names_8 = [
    'Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt'
]

class_names_7 = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']

class_descriptor_7 = [
    'A person who is feeling neutral.',
    'A person who is feeling happy.',
    'A person who is feeling sad.',
    'A person who is feeling surprise.',
    'A person who is feeling fear.',
    'A person who is feeling disgust.',
    'A person who is feeling anger.'
]

class_names_with_context_7 = [
    'A person shows neutral emotion.',
    'A person shows happy emotion.',
    'A person shows sad emotion.',
    'A person shows surprise emotion.',
    'A person shows fear emotion.',
    'A person shows disgust emotion.',
    'A person shows anger emotion.'
]

class_descriptor_7_only_face = [
    'The face of a person who is feeling neutral.',
    'The face of a person who is feeling happy.',
    'The face of a person who is feeling sad.',
    'The face of a person who is feeling surprise.',
    'The face of a person who is feeling fear.',
    'The face of a person who is feeling disgust.',
    'The face of a person who is feeling anger.'
]

class_descriptor_7_only_body = [
    'The body of a person who is feeling neutral.',
    'The body of a person who is feeling happy.',
    'The body of a person who is feeling sad.',
    'The body of a person who is feeling surprise.',
    'The body of a person who is feeling fear.',
    'The body of a person who is feeling disgust.',
    'The body of a person who is feeling anger.'
]

class_names_with_context_8 = [
    'A person shows neutral emotion.',
    'A person shows happy emotion.',
    'A person shows sad emotion.',
    'A person shows surprise emotion.',
    'A person shows fear emotion.',
    'A person shows disgust emotion.',
    'A person shows anger emotion.',
    'A person shows contempt emotion.'
]

class_descriptor_8_only_face = [
    'The face of a person who is feeling neutral.',
    'The face of a person who is feeling happy.',
    'The face of a person who is feeling sad.',
    'The face of a person who is feeling surprise.',
    'The face of a person who is feeling fear.',
    'The face of a person who is feeling disgust.',
    'The face of a person who is feeling anger.',
    'The face of a person who is feeling contempt.'
]

class_descriptor_8_only_body = [
    'The body of a person who is feeling neutral.',
    'The body of a person who is feeling happy.',
    'The body of a person who is feeling sad.',
    'The body of a person who is feeling surprise.',
    'The body of a person who is feeling fear.',
    'The body of a person who is feeling disgust.',
    'The body of a person who is feeling anger.',
    'The body of a person who is feeling contempt.'
]

# CK+ Classes (Alphabetical Order: Anger, Contempt, Disgust, Fear, Happy, Sadness, Surprise)
class_names_ckplus = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']

class_names_with_context_ckplus = [
    "A person shows anger.",
    "A person shows contempt.",
    "A person shows disgust.",
    "A person shows fear.",
    "A person shows happiness.",
    "A person shows sadness.",
    "A person shows surprise."
]

class_descriptor_ckplus = [
    "A person with an angry expression, furrowed brows and tightened lips.",
    "A person with a contemptuous expression, one corner of the lip raised.",
    "A person with a disgusted expression, nose wrinkled and upper lip raised.",
    "A person with a fearful expression, eyes wide open and eyebrows raised.",
    "A person with a happy expression, smiling with cheeks raised.",
    "A person with a sad expression, corners of the lips turned down and drooping eyelids.",
    "A person with a surprised expression, mouth open and eyes widened."
]

prompt_ensemble_ckplus = [
    [ # Anger
        "A photo of a person showing anger.",
        "A face with furrowed brows and a glare.",
        "An angry facial expression."
    ],
    [ # Contempt
        "A photo of a person showing contempt.",
        "A face with a smirk or sneer.",
        "A contemptuous facial expression."
    ],
    [ # Disgust
        "A photo of a person showing disgust.",
        "A face with a wrinkled nose.",
        "A disgusted facial expression."
    ],
    [ # Fear
        "A photo of a person showing fear.",
        "A face with wide eyes and a terrified look.",
        "A fearful facial expression."
    ],
    [ # Happy
        "A photo of a person showing happiness.",
        "A smiling face with joy.",
        "A happy facial expression."
    ],
    [ # Sadness
        "A photo of a person showing sadness.",
        "A face with a frown and sorrowful eyes.",
        "A sad facial expression."
    ],
    [ # Surprise
        "A photo of a person showing surprise.",
        "A face with an open mouth and wide eyes.",
        "A surprised facial expression."
    ]
]

# SFER Classes (Alphabetical: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
class_names_sfer = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

class_names_with_context_sfer = [
    "A student shows anger.",
    "A student shows disgust.",
    "A student shows fear.",
    "A student shows happiness.",
    "A student shows neutrality.",
    "A student shows sadness.",
    "A student shows surprise."
]

class_descriptor_sfer = [
    "A student with an angry expression, furrowed brows and tightened lips.",
    "A student with a disgusted expression, nose wrinkled and upper lip raised.",
    "A student with a fearful expression, eyes wide open and eyebrows raised.",
    "A student with a happy expression, smiling with cheeks raised.",
    "A student with a neutral expression, relaxed face and calm gaze.",
    "A student with a sad expression, corners of the lips turned down and drooping eyelids.",
    "A student with a surprised expression, mouth open and eyes widened."
]

prompt_ensemble_sfer = [
    [ # Anger
        "A photo of a student showing anger.",
        "A face with furrowed brows and a glare.",
        "An angry facial expression."
    ],
    [ # Disgust
        "A photo of a student showing disgust.",
        "A face with a wrinkled nose.",
        "A disgusted facial expression."
    ],
    [ # Fear
        "A photo of a student showing fear.",
        "A face with wide eyes and a terrified look.",
        "A fearful facial expression."
    ],
    [ # Happy
        "A photo of a student showing happiness.",
        "A smiling face with joy.",
        "A happy facial expression."
    ],
    [ # Neutral
        "A photo of a student showing a neutral expression.",
        "A calm face with no strong emotion.",
        "A neutral facial expression."
    ],
    [ # Sad
        "A photo of a student showing sadness.",
        "A face with a frown and sorrowful eyes.",
        "A sad facial expression."
    ],
    [ # Surprise
        "A photo of a student showing surprise.",
        "A face with an open mouth and wide eyes.",
        "A surprised facial expression."
    ]
]

# CAER Classes (Alphabetical: Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise)
class_names_caer = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

class_names_with_context_caer = [
    "A person shows anger.",
    "A person shows disgust.",
    "A person shows fear.",
    "A person shows happiness.",
    "A person shows neutrality.",
    "A person shows sadness.",
    "A person shows surprise."
]

class_descriptor_caer = [
    "A person with an angry expression, furrowed brows and tightened lips.",
    "A person with a disgusted expression, nose wrinkled and upper lip raised.",
    "A person with a fearful expression, eyes wide open and eyebrows raised.",
    "A person with a happy expression, smiling with cheeks raised.",
    "A person with a neutral expression, relaxed face and calm gaze.",
    "A person with a sad expression, corners of the lips turned down and drooping eyelids.",
    "A person with a surprised expression, mouth open and eyes widened."
]

prompt_ensemble_caer = [
    [ # Anger
        "A photo of a person showing anger.",
        "A face with furrowed brows and a glare.",
        "An angry facial expression."
    ],
    [ # Disgust
        "A photo of a person showing disgust.",
        "A face with a wrinkled nose.",
        "A disgusted facial expression."
    ],
    [ # Fear
        "A photo of a person showing fear.",
        "A face with wide eyes and a terrified look.",
        "A fearful facial expression."
    ],
    [ # Happy
        "A photo of a person showing happiness.",
        "A smiling face with joy.",
        "A happy facial expression."
    ],
    [ # Neutral
        "A photo of a person showing a neutral expression.",
        "A calm face with no strong emotion.",
        "A neutral facial expression."
    ],
    [ # Sad
        "A photo of a person showing sadness.",
        "A face with a frown and sorrowful eyes.",
        "A sad facial expression."
    ],
    [ # Surprise
        "A photo of a person showing surprise.",
        "A face with an open mouth and wide eyes.",
        "A surprised facial expression."
    ]
]


class_names_daisee = ['Very Low', 'Low', 'High', 'Very High']

class_names_with_context_daisee = [
    "A student shows very low engagement.",
    "A student shows low engagement.",
    "A student shows high engagement.",
    "A student shows very high engagement."
]

class_descriptor_daisee = [
    "A student is completely disengaged, looking away, sleeping, or doing something else entirely.",
    "A student is distracted, frequently looking around, yawning, or showing little interest.",
    "A student is paying attention, looking at the screen, and following the lesson.",
    "A student is highly focused, leaning forward, taking notes, and reacting to the content."
]

prompt_ensemble_daisee = [
    [ # Very Low (0)
        "A video of a student with very low engagement.",
        "A student looking away or sleeping.",
        "A completely disengaged student."
    ],
    [ # Low (1)
        "A video of a student with low engagement.",
        "A student looking distracted or bored.",
        "A student showing little interest in the lesson."
    ],
    [ # High (2)
        "A video of a student with high engagement.",
        "A student looking at the screen attentively.",
        "A student following the lecture."
    ],
    [ # Very High (3)
        "A video of a highly engaged student.",
        "A student leaning forward and taking notes.",
        "A student completely absorbed in learning."
    ]
]

# EMOTIC 26 Classes
class_names_emotic = [
    'Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 
    'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 
    'Esteem', 'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace', 'Pleasure', 
    'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning'
]

class_descriptor_emotic_26 = [
    "A person feeling affection, showing fond attachment, devotion, or love.",
    "A person feeling anger, expressing strong displeasure or hostility.",
    "A person feeling annoyance, showing mild anger, irritation, or nuisance.",
    "A person feeling anticipation, looking forward to an upcoming event.",
    "A person feeling aversion, showing a strong feeling of dislike or opposition.",
    "A person feeling confidence, appearing self-assured and trusting their abilities.",
    "A person feeling disapproval, expressing condemnation or a negative opinion.",
    "A person feeling disconnection, seeming detached, isolated, or unresponsive.",
    "A person feeling disquietment, appearing restless, uneasy, or anxious.",
    "A person feeling doubt or confusion, appearing uncertain, puzzled, or lacking conviction.",
    "A person feeling embarrassment, showing shame, self-consciousness, or awkwardness.",
    "A person feeling engagement, paying close attention or being deeply involved.",
    "A person feeling esteem, showing respect, admiration, or a favorable opinion.",
    "A person feeling excitement, appearing thrilled, eager, or highly enthusiastic.",
    "A person feeling fatigue, showing physical or mental exhaustion and weariness.",
    "A person feeling fear, showing alarm or apprehension from perceived danger.",
    "A person feeling happiness, expressing joy, contentment, or pleasure.",
    "A person feeling pain, showing physical suffering or distress.",
    "A person feeling peace, appearing calm, tranquil, and free from disturbance.",
    "A person feeling pleasure, showing delight, enjoyment, or satisfaction.",
    "A person feeling sadness, expressing sorrow, unhappiness, or grief.",
    "A person feeling sensitivity, showing a delicate or acute emotional response.",
    "A person feeling suffering, enduring severe physical or mental pain.",
    "A person feeling surprise, showing astonishment or wonder at something unexpected.",
    "A person feeling sympathy, showing compassion or understanding for others.",
    "A person feeling yearning, showing a strong, longing desire for something."
]

class_names_with_context_emotic = [f"a photo of a person showing {cls}" for cls in class_names_emotic]

# Rich prompt ensemble for EMOTIC v5
prompt_ensemble_emotic_26 = [
    [
        "a photo of a person showing fond attachment, devotion, or love, feeling affection.",
        "a video of a person showing fond attachment, devotion, or love, feeling affection.",
        "a portrait of a person showing fond attachment, devotion, or love, feeling affection.",
        "an image showing a person showing fond attachment, devotion, or love, feeling affection.",
        "a close-up of a person showing fond attachment, devotion, or love, feeling affection.",
        "a face of a person showing fond attachment, devotion, or love, feeling affection.",
        "a picture of a person showing fond attachment, devotion, or love, feeling affection.",
        "a shot of a person showing fond attachment, devotion, or love, feeling affection.",
    ],
    [
        "a photo of a person expressing strong displeasure or hostility, frowning, feeling anger.",
        "a video of a person expressing strong displeasure or hostility, frowning, feeling anger.",
        "a portrait of a person expressing strong displeasure or hostility, frowning, feeling anger.",
        "an image showing a person expressing strong displeasure or hostility, frowning, feeling anger.",
        "a close-up of a person expressing strong displeasure or hostility, frowning, feeling anger.",
        "a face of a person expressing strong displeasure or hostility, frowning, feeling anger.",
        "a picture of a person expressing strong displeasure or hostility, frowning, feeling anger.",
        "a shot of a person expressing strong displeasure or hostility, frowning, feeling anger.",
    ],
    [
        "a photo of a person showing mild anger, irritation, or nuisance, feeling annoyance.",
        "a video of a person showing mild anger, irritation, or nuisance, feeling annoyance.",
        "a portrait of a person showing mild anger, irritation, or nuisance, feeling annoyance.",
        "an image showing a person showing mild anger, irritation, or nuisance, feeling annoyance.",
        "a close-up of a person showing mild anger, irritation, or nuisance, feeling annoyance.",
        "a face of a person showing mild anger, irritation, or nuisance, feeling annoyance.",
        "a picture of a person showing mild anger, irritation, or nuisance, feeling annoyance.",
        "a shot of a person showing mild anger, irritation, or nuisance, feeling annoyance.",
    ],
    [
        "a photo of a person looking forward to an upcoming event eagerly, feeling anticipation.",
        "a video of a person looking forward to an upcoming event eagerly, feeling anticipation.",
        "a portrait of a person looking forward to an upcoming event eagerly, feeling anticipation.",
        "an image showing a person looking forward to an upcoming event eagerly, feeling anticipation.",
        "a close-up of a person looking forward to an upcoming event eagerly, feeling anticipation.",
        "a face of a person looking forward to an upcoming event eagerly, feeling anticipation.",
        "a picture of a person looking forward to an upcoming event eagerly, feeling anticipation.",
        "a shot of a person looking forward to an upcoming event eagerly, feeling anticipation.",
    ],
    [
        "a photo of a person turning head away in disgust, wrinkling nose, feeling aversion.",
        "a video of a person turning head away in disgust, wrinkling nose, feeling aversion.",
        "a portrait of a person turning head away in disgust, wrinkling nose, feeling aversion.",
        "an image showing a person turning head away in disgust, wrinkling nose, feeling aversion.",
        "a close-up of a person turning head away in disgust, wrinkling nose, feeling aversion.",
        "a face of a person turning head away in disgust, wrinkling nose, feeling aversion.",
        "a picture of a person turning head away in disgust, wrinkling nose, feeling aversion.",
        "a shot of a person turning head away in disgust, wrinkling nose, feeling aversion.",
    ],
    [
        "a photo of a person appearing self-assured and trusting their abilities, feeling confidence.",
        "a video of a person appearing self-assured and trusting their abilities, feeling confidence.",
        "a portrait of a person appearing self-assured and trusting their abilities, feeling confidence.",
        "an image showing a person appearing self-assured and trusting their abilities, feeling confidence.",
        "a close-up of a person appearing self-assured and trusting their abilities, feeling confidence.",
        "a face of a person appearing self-assured and trusting their abilities, feeling confidence.",
        "a picture of a person appearing self-assured and trusting their abilities, feeling confidence.",
        "a shot of a person appearing self-assured and trusting their abilities, feeling confidence.",
    ],
    [
        "a photo of a person expressing condemnation or a negative opinion, feeling disapproval.",
        "a video of a person expressing condemnation or a negative opinion, feeling disapproval.",
        "a portrait of a person expressing condemnation or a negative opinion, feeling disapproval.",
        "an image showing a person expressing condemnation or a negative opinion, feeling disapproval.",
        "a close-up of a person expressing condemnation or a negative opinion, feeling disapproval.",
        "a face of a person expressing condemnation or a negative opinion, feeling disapproval.",
        "a picture of a person expressing condemnation or a negative opinion, feeling disapproval.",
        "a shot of a person expressing condemnation or a negative opinion, feeling disapproval.",
    ],
    [
        "a photo of a person seeming detached, isolated, looking away or unresponsive, feeling disconnection.",
        "a video of a person seeming detached, isolated, looking away or unresponsive, feeling disconnection.",
        "a portrait of a person seeming detached, isolated, looking away or unresponsive, feeling disconnection.",
        "an image showing a person seeming detached, isolated, looking away or unresponsive, feeling disconnection.",
        "a close-up of a person seeming detached, isolated, looking away or unresponsive, feeling disconnection.",
        "a face of a person seeming detached, isolated, looking away or unresponsive, feeling disconnection.",
        "a picture of a person seeming detached, isolated, looking away or unresponsive, feeling disconnection.",
        "a shot of a person seeming detached, isolated, looking away or unresponsive, feeling disconnection.",
    ],
    [
        "a photo of a person appearing restless, uneasy, or anxious, feeling disquietment.",
        "a video of a person appearing restless, uneasy, or anxious, feeling disquietment.",
        "a portrait of a person appearing restless, uneasy, or anxious, feeling disquietment.",
        "an image showing a person appearing restless, uneasy, or anxious, feeling disquietment.",
        "a close-up of a person appearing restless, uneasy, or anxious, feeling disquietment.",
        "a face of a person appearing restless, uneasy, or anxious, feeling disquietment.",
        "a picture of a person appearing restless, uneasy, or anxious, feeling disquietment.",
        "a shot of a person appearing restless, uneasy, or anxious, feeling disquietment.",
    ],
    [
        "a photo of a person appearing uncertain, puzzled, or lacking conviction, feeling doubt/confusion.",
        "a video of a person appearing uncertain, puzzled, or lacking conviction, feeling doubt/confusion.",
        "a portrait of a person appearing uncertain, puzzled, or lacking conviction, feeling doubt/confusion.",
        "an image showing a person appearing uncertain, puzzled, or lacking conviction, feeling doubt/confusion.",
        "a close-up of a person appearing uncertain, puzzled, or lacking conviction, feeling doubt/confusion.",
        "a face of a person appearing uncertain, puzzled, or lacking conviction, feeling doubt/confusion.",
        "a picture of a person appearing uncertain, puzzled, or lacking conviction, feeling doubt/confusion.",
        "a shot of a person appearing uncertain, puzzled, or lacking conviction, feeling doubt/confusion.",
    ],
    [
        "a photo of a person averting gaze, blushing or covering face in shame, feeling embarrassment.",
        "a video of a person averting gaze, blushing or covering face in shame, feeling embarrassment.",
        "a portrait of a person averting gaze, blushing or covering face in shame, feeling embarrassment.",
        "an image showing a person averting gaze, blushing or covering face in shame, feeling embarrassment.",
        "a close-up of a person averting gaze, blushing or covering face in shame, feeling embarrassment.",
        "a face of a person averting gaze, blushing or covering face in shame, feeling embarrassment.",
        "a picture of a person averting gaze, blushing or covering face in shame, feeling embarrassment.",
        "a shot of a person averting gaze, blushing or covering face in shame, feeling embarrassment.",
    ],
    [
        "a photo of a person paying close attention, interacting or being deeply involved, feeling engagement.",
        "a video of a person paying close attention, interacting or being deeply involved, feeling engagement.",
        "a portrait of a person paying close attention, interacting or being deeply involved, feeling engagement.",
        "an image showing a person paying close attention, interacting or being deeply involved, feeling engagement.",
        "a close-up of a person paying close attention, interacting or being deeply involved, feeling engagement.",
        "a face of a person paying close attention, interacting or being deeply involved, feeling engagement.",
        "a picture of a person paying close attention, interacting or being deeply involved, feeling engagement.",
        "a shot of a person paying close attention, interacting or being deeply involved, feeling engagement.",
    ],
    [
        "a photo of a person showing respect, admiration, or a favorable opinion, feeling esteem.",
        "a video of a person showing respect, admiration, or a favorable opinion, feeling esteem.",
        "a portrait of a person showing respect, admiration, or a favorable opinion, feeling esteem.",
        "an image showing a person showing respect, admiration, or a favorable opinion, feeling esteem.",
        "a close-up of a person showing respect, admiration, or a favorable opinion, feeling esteem.",
        "a face of a person showing respect, admiration, or a favorable opinion, feeling esteem.",
        "a picture of a person showing respect, admiration, or a favorable opinion, feeling esteem.",
        "a shot of a person showing respect, admiration, or a favorable opinion, feeling esteem.",
    ],
    [
        "a photo of a person appearing thrilled, eager, or highly enthusiastic, feeling excitement.",
        "a video of a person appearing thrilled, eager, or highly enthusiastic, feeling excitement.",
        "a portrait of a person appearing thrilled, eager, or highly enthusiastic, feeling excitement.",
        "an image showing a person appearing thrilled, eager, or highly enthusiastic, feeling excitement.",
        "a close-up of a person appearing thrilled, eager, or highly enthusiastic, feeling excitement.",
        "a face of a person appearing thrilled, eager, or highly enthusiastic, feeling excitement.",
        "a picture of a person appearing thrilled, eager, or highly enthusiastic, feeling excitement.",
        "a shot of a person appearing thrilled, eager, or highly enthusiastic, feeling excitement.",
    ],
    [
        "a photo of a person showing physical or mental exhaustion and weariness, slouching, feeling fatigue.",
        "a video of a person showing physical or mental exhaustion and weariness, slouching, feeling fatigue.",
        "a portrait of a person showing physical or mental exhaustion and weariness, slouching, feeling fatigue.",
        "an image showing a person showing physical or mental exhaustion and weariness, slouching, feeling fatigue.",
        "a close-up of a person showing physical or mental exhaustion and weariness, slouching, feeling fatigue.",
        "a face of a person showing physical or mental exhaustion and weariness, slouching, feeling fatigue.",
        "a picture of a person showing physical or mental exhaustion and weariness, slouching, feeling fatigue.",
        "a shot of a person showing physical or mental exhaustion and weariness, slouching, feeling fatigue.",
    ],
    [
        "a photo of a person with wide eyes, open mouth in terror, tense posture, feeling fear.",
        "a video of a person with wide eyes, open mouth in terror, tense posture, feeling fear.",
        "a portrait of a person with wide eyes, open mouth in terror, tense posture, feeling fear.",
        "an image showing a person with wide eyes, open mouth in terror, tense posture, feeling fear.",
        "a close-up of a person with wide eyes, open mouth in terror, tense posture, feeling fear.",
        "a face of a person with wide eyes, open mouth in terror, tense posture, feeling fear.",
        "a picture of a person with wide eyes, open mouth in terror, tense posture, feeling fear.",
        "a shot of a person with wide eyes, open mouth in terror, tense posture, feeling fear.",
    ],
    [
        "a photo of a person expressing joy, contentment, or pleasure, smiling, feeling happiness.",
        "a video of a person expressing joy, contentment, or pleasure, smiling, feeling happiness.",
        "a portrait of a person expressing joy, contentment, or pleasure, smiling, feeling happiness.",
        "an image showing a person expressing joy, contentment, or pleasure, smiling, feeling happiness.",
        "a close-up of a person expressing joy, contentment, or pleasure, smiling, feeling happiness.",
        "a face of a person expressing joy, contentment, or pleasure, smiling, feeling happiness.",
        "a picture of a person expressing joy, contentment, or pleasure, smiling, feeling happiness.",
        "a shot of a person expressing joy, contentment, or pleasure, smiling, feeling happiness.",
    ],
    [
        "a photo of a person showing physical suffering or distress, grimacing, feeling pain.",
        "a video of a person showing physical suffering or distress, grimacing, feeling pain.",
        "a portrait of a person showing physical suffering or distress, grimacing, feeling pain.",
        "an image showing a person showing physical suffering or distress, grimacing, feeling pain.",
        "a close-up of a person showing physical suffering or distress, grimacing, feeling pain.",
        "a face of a person showing physical suffering or distress, grimacing, feeling pain.",
        "a picture of a person showing physical suffering or distress, grimacing, feeling pain.",
        "a shot of a person showing physical suffering or distress, grimacing, feeling pain.",
    ],
    [
        "a photo of a person appearing calm, tranquil, relaxed and free from disturbance, feeling peace.",
        "a video of a person appearing calm, tranquil, relaxed and free from disturbance, feeling peace.",
        "a portrait of a person appearing calm, tranquil, relaxed and free from disturbance, feeling peace.",
        "an image showing a person appearing calm, tranquil, relaxed and free from disturbance, feeling peace.",
        "a close-up of a person appearing calm, tranquil, relaxed and free from disturbance, feeling peace.",
        "a face of a person appearing calm, tranquil, relaxed and free from disturbance, feeling peace.",
        "a picture of a person appearing calm, tranquil, relaxed and free from disturbance, feeling peace.",
        "a shot of a person appearing calm, tranquil, relaxed and free from disturbance, feeling peace.",
    ],
    [
        "a photo of a person showing delight, enjoyment, or satisfaction, feeling pleasure.",
        "a video of a person showing delight, enjoyment, or satisfaction, feeling pleasure.",
        "a portrait of a person showing delight, enjoyment, or satisfaction, feeling pleasure.",
        "an image showing a person showing delight, enjoyment, or satisfaction, feeling pleasure.",
        "a close-up of a person showing delight, enjoyment, or satisfaction, feeling pleasure.",
        "a face of a person showing delight, enjoyment, or satisfaction, feeling pleasure.",
        "a picture of a person showing delight, enjoyment, or satisfaction, feeling pleasure.",
        "a shot of a person showing delight, enjoyment, or satisfaction, feeling pleasure.",
    ],
    [
        "a photo of a person expressing sorrow, unhappiness, or grief, crying or frowning, feeling sadness.",
        "a video of a person expressing sorrow, unhappiness, or grief, crying or frowning, feeling sadness.",
        "a portrait of a person expressing sorrow, unhappiness, or grief, crying or frowning, feeling sadness.",
        "an image showing a person expressing sorrow, unhappiness, or grief, crying or frowning, feeling sadness.",
        "a close-up of a person expressing sorrow, unhappiness, or grief, crying or frowning, feeling sadness.",
        "a face of a person expressing sorrow, unhappiness, or grief, crying or frowning, feeling sadness.",
        "a picture of a person expressing sorrow, unhappiness, or grief, crying or frowning, feeling sadness.",
        "a shot of a person expressing sorrow, unhappiness, or grief, crying or frowning, feeling sadness.",
    ],
    [
        "a photo of a person showing a delicate or acute emotional response, feeling sensitivity.",
        "a video of a person showing a delicate or acute emotional response, feeling sensitivity.",
        "a portrait of a person showing a delicate or acute emotional response, feeling sensitivity.",
        "an image showing a person showing a delicate or acute emotional response, feeling sensitivity.",
        "a close-up of a person showing a delicate or acute emotional response, feeling sensitivity.",
        "a face of a person showing a delicate or acute emotional response, feeling sensitivity.",
        "a picture of a person showing a delicate or acute emotional response, feeling sensitivity.",
        "a shot of a person showing a delicate or acute emotional response, feeling sensitivity.",
    ],
    [
        "a photo of a person enduring severe physical or mental pain, crying or holding face, feeling suffering.",
        "a video of a person enduring severe physical or mental pain, crying or holding face, feeling suffering.",
        "a portrait of a person enduring severe physical or mental pain, crying or holding face, feeling suffering.",
        "an image showing a person enduring severe physical or mental pain, crying or holding face, feeling suffering.",
        "a close-up of a person enduring severe physical or mental pain, crying or holding face, feeling suffering.",
        "a face of a person enduring severe physical or mental pain, crying or holding face, feeling suffering.",
        "a picture of a person enduring severe physical or mental pain, crying or holding face, feeling suffering.",
        "a shot of a person enduring severe physical or mental pain, crying or holding face, feeling suffering.",
    ],
    [
        "a photo of a person with raised eyebrows, gasping, widened eyes, feeling surprise.",
        "a video of a person with raised eyebrows, gasping, widened eyes, feeling surprise.",
        "a portrait of a person with raised eyebrows, gasping, widened eyes, feeling surprise.",
        "an image showing a person with raised eyebrows, gasping, widened eyes, feeling surprise.",
        "a close-up of a person with raised eyebrows, gasping, widened eyes, feeling surprise.",
        "a face of a person with raised eyebrows, gasping, widened eyes, feeling surprise.",
        "a picture of a person with raised eyebrows, gasping, widened eyes, feeling surprise.",
        "a shot of a person with raised eyebrows, gasping, widened eyes, feeling surprise.",
    ],
    [
        "a photo of a person showing compassion or understanding for others, feeling sympathy.",
        "a video of a person showing compassion or understanding for others, feeling sympathy.",
        "a portrait of a person showing compassion or understanding for others, feeling sympathy.",
        "an image showing a person showing compassion or understanding for others, feeling sympathy.",
        "a close-up of a person showing compassion or understanding for others, feeling sympathy.",
        "a face of a person showing compassion or understanding for others, feeling sympathy.",
        "a picture of a person showing compassion or understanding for others, feeling sympathy.",
        "a shot of a person showing compassion or understanding for others, feeling sympathy.",
    ],
    [
        "a photo of a person looking into the distance with a wistful expression, longing, feeling yearning.",
        "a video of a person looking into the distance with a wistful expression, longing, feeling yearning.",
        "a portrait of a person looking into the distance with a wistful expression, longing, feeling yearning.",
        "an image showing a person looking into the distance with a wistful expression, longing, feeling yearning.",
        "a close-up of a person looking into the distance with a wistful expression, longing, feeling yearning.",
        "a face of a person looking into the distance with a wistful expression, longing, feeling yearning.",
        "a picture of a person looking into the distance with a wistful expression, longing, feeling yearning.",
        "a shot of a person looking into the distance with a wistful expression, longing, feeling yearning.",
    ],
]

class_names_daisee = ['Very Low Engagement', 'Low Engagement', 'High Engagement', 'Very High Engagement']
class_names_with_context_daisee = [
    'A student showing very low engagement in an online classroom.',
    'A student showing low engagement in an online classroom.',
    'A student showing high engagement in an online classroom.',
    'A student highly engaged and deeply focused on the online classroom.'
]
class_descriptor_daisee = [
    "A close-up of a student's face with eyes fully closed, appearing to be asleep during class.",
    "A student looking away from the screen, distracted and showing low engagement.",
    "A student's face with steady eye contact toward the camera and a neutral attentive expression.",
    "A close-up of a student's face with wide alert eyes showing intense focus and interest."
]
prompt_ensemble_daisee = [
    [
        "A close-up of a student's face with eyes fully closed, appearing to be asleep during class.",
        "A student showing very low engagement in an online classroom."
    ],
    [
        "A student looking away from the screen, distracted and showing low engagement.",
        "A student with unfocused eyes, looking bored or confused."
    ],
    [
        "A student's face with steady eye contact toward the camera and a neutral attentive expression.",
        "A student looking at the screen with high engagement."
    ],
    [
        "A close-up of a student's face with wide alert eyes showing intense focus and interest.",
        "A student highly engaged and deeply focused on the online classroom."
    ]
]
