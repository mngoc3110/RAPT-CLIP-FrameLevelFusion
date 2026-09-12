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
    "A person feeling affection: fond feelings, love, or tenderness.",
    "A person feeling anger: intense displeasure or rage, furious, or resentful.",
    "A person feeling annoyance: bothered by something or someone, irritated, impatient, or frustrated.",
    "A person feeling anticipation: state of looking forward, hoping on or getting prepared for possible future events.",
    "A person feeling aversion: feeling disgust, dislike, repulsion, or hate.",
    "A person feeling confidence: feeling of being certain, conviction that an outcome will be favorable, encouraged, or proud.",
    "A person feeling disapproval: feeling that something is wrong or reprehensible, contempt, or hostile.",
    "A person feeling disconnection: feeling not interested in the main event of the surrounding, indifferent, bored, or distracted.",
    "A person feeling disquietment: nervous, worried, upset, anxious, tense, pressured, or alarmed.",
    "A person feeling doubt or confusion: difficulty to understand or decide, or thinking about different options.",
    "A person feeling embarrassment: feeling ashamed or guilty.",
    "A person feeling engagement: paying attention to something, absorbed into something, curious, or interested.",
    "A person feeling esteem: feelings of favorable opinion or judgment, respect, admiration, or gratefulness.",
    "A person feeling excitement: feeling enthusiasm, stimulated, or energetic.",
    "A person feeling fatigue: weariness, tiredness, or sleepy.",
    "A person feeling fear: feeling suspicious or afraid of danger, threat, evil or pain, or horror.",
    "A person feeling happiness: feeling delighted, or feeling enjoyment or amusement.",
    "A person feeling pain: physical suffering.",
    "A person feeling peace: well being and relaxed, no worry, having positive thoughts or sensations, or satisfied.",
    "A person feeling pleasure: feeling of delight in the senses.",
    "A person feeling sadness: feeling unhappy, sorrow, disappointed, or discouraged.",
    "A person feeling sensitivity: feeling of being physically or emotionally wounded, or feeling delicate or vulnerable.",
    "A person feeling suffering: psychological or emotional pain, distressed, or anguished.",
    "A person feeling surprise: sudden discovery of something unexpected.",
    "A person feeling sympathy: state of sharing others emotions, goals or troubles, supportive, or compassionate.",
    "A person feeling yearning: strong desire to have something, jealous, envious, or lust."
]

class_names_with_context_emotic = [f"a photo of a person showing {cls}" for cls in class_names_emotic]

# Rich prompt ensemble for EMOTIC v5
prompt_ensemble_emotic_26 = [
    [
        "a photo of a person feeling affection: fond feelings, love, or tenderness.",
        "a video of a person feeling affection: fond feelings, love, or tenderness.",
        "a portrait of a person feeling affection: fond feelings, love, or tenderness.",
        "an image showing a person feeling affection: fond feelings, love, or tenderness.",
        "a close-up of a person feeling affection: fond feelings, love, or tenderness.",
        "a face of a person feeling affection: fond feelings, love, or tenderness.",
        "a picture of a person feeling affection: fond feelings, love, or tenderness.",
        "a shot of a person feeling affection: fond feelings, love, or tenderness.",
    ],
    [
        "a photo of a person feeling anger: intense displeasure or rage, furious, or resentful.",
        "a video of a person feeling anger: intense displeasure or rage, furious, or resentful.",
        "a portrait of a person feeling anger: intense displeasure or rage, furious, or resentful.",
        "an image showing a person feeling anger: intense displeasure or rage, furious, or resentful.",
        "a close-up of a person feeling anger: intense displeasure or rage, furious, or resentful.",
        "a face of a person feeling anger: intense displeasure or rage, furious, or resentful.",
        "a picture of a person feeling anger: intense displeasure or rage, furious, or resentful.",
        "a shot of a person feeling anger: intense displeasure or rage, furious, or resentful.",
    ],
    [
        "a photo of a person feeling annoyance: bothered by something or someone, irritated, impatient, or frustrated.",
        "a video of a person feeling annoyance: bothered by something or someone, irritated, impatient, or frustrated.",
        "a portrait of a person feeling annoyance: bothered by something or someone, irritated, impatient, or frustrated.",
        "an image showing a person feeling annoyance: bothered by something or someone, irritated, impatient, or frustrated.",
        "a close-up of a person feeling annoyance: bothered by something or someone, irritated, impatient, or frustrated.",
        "a face of a person feeling annoyance: bothered by something or someone, irritated, impatient, or frustrated.",
        "a picture of a person feeling annoyance: bothered by something or someone, irritated, impatient, or frustrated.",
        "a shot of a person feeling annoyance: bothered by something or someone, irritated, impatient, or frustrated.",
    ],
    [
        "a photo of a person feeling anticipation: state of looking forward, hoping on or getting prepared for possible future events.",
        "a video of a person feeling anticipation: state of looking forward, hoping on or getting prepared for possible future events.",
        "a portrait of a person feeling anticipation: state of looking forward, hoping on or getting prepared for possible future events.",
        "an image showing a person feeling anticipation: state of looking forward, hoping on or getting prepared for possible future events.",
        "a close-up of a person feeling anticipation: state of looking forward, hoping on or getting prepared for possible future events.",
        "a face of a person feeling anticipation: state of looking forward, hoping on or getting prepared for possible future events.",
        "a picture of a person feeling anticipation: state of looking forward, hoping on or getting prepared for possible future events.",
        "a shot of a person feeling anticipation: state of looking forward, hoping on or getting prepared for possible future events.",
    ],
    [
        "a photo of a person feeling aversion: feeling disgust, dislike, repulsion, or hate.",
        "a video of a person feeling aversion: feeling disgust, dislike, repulsion, or hate.",
        "a portrait of a person feeling aversion: feeling disgust, dislike, repulsion, or hate.",
        "an image showing a person feeling aversion: feeling disgust, dislike, repulsion, or hate.",
        "a close-up of a person feeling aversion: feeling disgust, dislike, repulsion, or hate.",
        "a face of a person feeling aversion: feeling disgust, dislike, repulsion, or hate.",
        "a picture of a person feeling aversion: feeling disgust, dislike, repulsion, or hate.",
        "a shot of a person feeling aversion: feeling disgust, dislike, repulsion, or hate.",
    ],
    [
        "a photo of a person feeling confidence: feeling of being certain, conviction that an outcome will be favorable, encouraged, or proud.",
        "a video of a person feeling confidence: feeling of being certain, conviction that an outcome will be favorable, encouraged, or proud.",
        "a portrait of a person feeling confidence: feeling of being certain, conviction that an outcome will be favorable, encouraged, or proud.",
        "an image showing a person feeling confidence: feeling of being certain, conviction that an outcome will be favorable, encouraged, or proud.",
        "a close-up of a person feeling confidence: feeling of being certain, conviction that an outcome will be favorable, encouraged, or proud.",
        "a face of a person feeling confidence: feeling of being certain, conviction that an outcome will be favorable, encouraged, or proud.",
        "a picture of a person feeling confidence: feeling of being certain, conviction that an outcome will be favorable, encouraged, or proud.",
        "a shot of a person feeling confidence: feeling of being certain, conviction that an outcome will be favorable, encouraged, or proud.",
    ],
    [
        "a photo of a person feeling disapproval: feeling that something is wrong or reprehensible, contempt, or hostile.",
        "a video of a person feeling disapproval: feeling that something is wrong or reprehensible, contempt, or hostile.",
        "a portrait of a person feeling disapproval: feeling that something is wrong or reprehensible, contempt, or hostile.",
        "an image showing a person feeling disapproval: feeling that something is wrong or reprehensible, contempt, or hostile.",
        "a close-up of a person feeling disapproval: feeling that something is wrong or reprehensible, contempt, or hostile.",
        "a face of a person feeling disapproval: feeling that something is wrong or reprehensible, contempt, or hostile.",
        "a picture of a person feeling disapproval: feeling that something is wrong or reprehensible, contempt, or hostile.",
        "a shot of a person feeling disapproval: feeling that something is wrong or reprehensible, contempt, or hostile.",
    ],
    [
        "a photo of a person feeling disconnection: feeling not interested in the main event of the surrounding, indifferent, bored, or distracted.",
        "a video of a person feeling disconnection: feeling not interested in the main event of the surrounding, indifferent, bored, or distracted.",
        "a portrait of a person feeling disconnection: feeling not interested in the main event of the surrounding, indifferent, bored, or distracted.",
        "an image showing a person feeling disconnection: feeling not interested in the main event of the surrounding, indifferent, bored, or distracted.",
        "a close-up of a person feeling disconnection: feeling not interested in the main event of the surrounding, indifferent, bored, or distracted.",
        "a face of a person feeling disconnection: feeling not interested in the main event of the surrounding, indifferent, bored, or distracted.",
        "a picture of a person feeling disconnection: feeling not interested in the main event of the surrounding, indifferent, bored, or distracted.",
        "a shot of a person feeling disconnection: feeling not interested in the main event of the surrounding, indifferent, bored, or distracted.",
    ],
    [
        "a photo of a person feeling disquietment: nervous, worried, upset, anxious, tense, pressured, or alarmed.",
        "a video of a person feeling disquietment: nervous, worried, upset, anxious, tense, pressured, or alarmed.",
        "a portrait of a person feeling disquietment: nervous, worried, upset, anxious, tense, pressured, or alarmed.",
        "an image showing a person feeling disquietment: nervous, worried, upset, anxious, tense, pressured, or alarmed.",
        "a close-up of a person feeling disquietment: nervous, worried, upset, anxious, tense, pressured, or alarmed.",
        "a face of a person feeling disquietment: nervous, worried, upset, anxious, tense, pressured, or alarmed.",
        "a picture of a person feeling disquietment: nervous, worried, upset, anxious, tense, pressured, or alarmed.",
        "a shot of a person feeling disquietment: nervous, worried, upset, anxious, tense, pressured, or alarmed.",
    ],
    [
        "a photo of a person feeling doubt/confusion: difficulty to understand or decide, or thinking about different options.",
        "a video of a person feeling doubt/confusion: difficulty to understand or decide, or thinking about different options.",
        "a portrait of a person feeling doubt/confusion: difficulty to understand or decide, or thinking about different options.",
        "an image showing a person feeling doubt/confusion: difficulty to understand or decide, or thinking about different options.",
        "a close-up of a person feeling doubt/confusion: difficulty to understand or decide, or thinking about different options.",
        "a face of a person feeling doubt/confusion: difficulty to understand or decide, or thinking about different options.",
        "a picture of a person feeling doubt/confusion: difficulty to understand or decide, or thinking about different options.",
        "a shot of a person feeling doubt/confusion: difficulty to understand or decide, or thinking about different options.",
    ],
    [
        "a photo of a person feeling embarrassment: feeling ashamed or guilty.",
        "a video of a person feeling embarrassment: feeling ashamed or guilty.",
        "a portrait of a person feeling embarrassment: feeling ashamed or guilty.",
        "an image showing a person feeling embarrassment: feeling ashamed or guilty.",
        "a close-up of a person feeling embarrassment: feeling ashamed or guilty.",
        "a face of a person feeling embarrassment: feeling ashamed or guilty.",
        "a picture of a person feeling embarrassment: feeling ashamed or guilty.",
        "a shot of a person feeling embarrassment: feeling ashamed or guilty.",
    ],
    [
        "a photo of a person feeling engagement: paying attention to something, absorbed into something, curious, or interested.",
        "a video of a person feeling engagement: paying attention to something, absorbed into something, curious, or interested.",
        "a portrait of a person feeling engagement: paying attention to something, absorbed into something, curious, or interested.",
        "an image showing a person feeling engagement: paying attention to something, absorbed into something, curious, or interested.",
        "a close-up of a person feeling engagement: paying attention to something, absorbed into something, curious, or interested.",
        "a face of a person feeling engagement: paying attention to something, absorbed into something, curious, or interested.",
        "a picture of a person feeling engagement: paying attention to something, absorbed into something, curious, or interested.",
        "a shot of a person feeling engagement: paying attention to something, absorbed into something, curious, or interested.",
    ],
    [
        "a photo of a person feeling esteem: feelings of favorable opinion or judgment, respect, admiration, or gratefulness.",
        "a video of a person feeling esteem: feelings of favorable opinion or judgment, respect, admiration, or gratefulness.",
        "a portrait of a person feeling esteem: feelings of favorable opinion or judgment, respect, admiration, or gratefulness.",
        "an image showing a person feeling esteem: feelings of favorable opinion or judgment, respect, admiration, or gratefulness.",
        "a close-up of a person feeling esteem: feelings of favorable opinion or judgment, respect, admiration, or gratefulness.",
        "a face of a person feeling esteem: feelings of favorable opinion or judgment, respect, admiration, or gratefulness.",
        "a picture of a person feeling esteem: feelings of favorable opinion or judgment, respect, admiration, or gratefulness.",
        "a shot of a person feeling esteem: feelings of favorable opinion or judgment, respect, admiration, or gratefulness.",
    ],
    [
        "a photo of a person feeling excitement: feeling enthusiasm, stimulated, or energetic.",
        "a video of a person feeling excitement: feeling enthusiasm, stimulated, or energetic.",
        "a portrait of a person feeling excitement: feeling enthusiasm, stimulated, or energetic.",
        "an image showing a person feeling excitement: feeling enthusiasm, stimulated, or energetic.",
        "a close-up of a person feeling excitement: feeling enthusiasm, stimulated, or energetic.",
        "a face of a person feeling excitement: feeling enthusiasm, stimulated, or energetic.",
        "a picture of a person feeling excitement: feeling enthusiasm, stimulated, or energetic.",
        "a shot of a person feeling excitement: feeling enthusiasm, stimulated, or energetic.",
    ],
    [
        "a photo of a person feeling fatigue: weariness, tiredness, or sleepy.",
        "a video of a person feeling fatigue: weariness, tiredness, or sleepy.",
        "a portrait of a person feeling fatigue: weariness, tiredness, or sleepy.",
        "an image showing a person feeling fatigue: weariness, tiredness, or sleepy.",
        "a close-up of a person feeling fatigue: weariness, tiredness, or sleepy.",
        "a face of a person feeling fatigue: weariness, tiredness, or sleepy.",
        "a picture of a person feeling fatigue: weariness, tiredness, or sleepy.",
        "a shot of a person feeling fatigue: weariness, tiredness, or sleepy.",
    ],
    [
        "a photo of a person feeling fear: feeling suspicious or afraid of danger, threat, evil or pain, or horror.",
        "a video of a person feeling fear: feeling suspicious or afraid of danger, threat, evil or pain, or horror.",
        "a portrait of a person feeling fear: feeling suspicious or afraid of danger, threat, evil or pain, or horror.",
        "an image showing a person feeling fear: feeling suspicious or afraid of danger, threat, evil or pain, or horror.",
        "a close-up of a person feeling fear: feeling suspicious or afraid of danger, threat, evil or pain, or horror.",
        "a face of a person feeling fear: feeling suspicious or afraid of danger, threat, evil or pain, or horror.",
        "a picture of a person feeling fear: feeling suspicious or afraid of danger, threat, evil or pain, or horror.",
        "a shot of a person feeling fear: feeling suspicious or afraid of danger, threat, evil or pain, or horror.",
    ],
    [
        "a photo of a person feeling happiness: feeling delighted, or feeling enjoyment or amusement.",
        "a video of a person feeling happiness: feeling delighted, or feeling enjoyment or amusement.",
        "a portrait of a person feeling happiness: feeling delighted, or feeling enjoyment or amusement.",
        "an image showing a person feeling happiness: feeling delighted, or feeling enjoyment or amusement.",
        "a close-up of a person feeling happiness: feeling delighted, or feeling enjoyment or amusement.",
        "a face of a person feeling happiness: feeling delighted, or feeling enjoyment or amusement.",
        "a picture of a person feeling happiness: feeling delighted, or feeling enjoyment or amusement.",
        "a shot of a person feeling happiness: feeling delighted, or feeling enjoyment or amusement.",
    ],
    [
        "a photo of a person feeling pain: physical suffering.",
        "a video of a person feeling pain: physical suffering.",
        "a portrait of a person feeling pain: physical suffering.",
        "an image showing a person feeling pain: physical suffering.",
        "a close-up of a person feeling pain: physical suffering.",
        "a face of a person feeling pain: physical suffering.",
        "a picture of a person feeling pain: physical suffering.",
        "a shot of a person feeling pain: physical suffering.",
    ],
    [
        "a photo of a person feeling peace: well being and relaxed, no worry, having positive thoughts or sensations, or satisfied.",
        "a video of a person feeling peace: well being and relaxed, no worry, having positive thoughts or sensations, or satisfied.",
        "a portrait of a person feeling peace: well being and relaxed, no worry, having positive thoughts or sensations, or satisfied.",
        "an image showing a person feeling peace: well being and relaxed, no worry, having positive thoughts or sensations, or satisfied.",
        "a close-up of a person feeling peace: well being and relaxed, no worry, having positive thoughts or sensations, or satisfied.",
        "a face of a person feeling peace: well being and relaxed, no worry, having positive thoughts or sensations, or satisfied.",
        "a picture of a person feeling peace: well being and relaxed, no worry, having positive thoughts or sensations, or satisfied.",
        "a shot of a person feeling peace: well being and relaxed, no worry, having positive thoughts or sensations, or satisfied.",
    ],
    [
        "a photo of a person feeling pleasure: feeling of delight in the senses.",
        "a video of a person feeling pleasure: feeling of delight in the senses.",
        "a portrait of a person feeling pleasure: feeling of delight in the senses.",
        "an image showing a person feeling pleasure: feeling of delight in the senses.",
        "a close-up of a person feeling pleasure: feeling of delight in the senses.",
        "a face of a person feeling pleasure: feeling of delight in the senses.",
        "a picture of a person feeling pleasure: feeling of delight in the senses.",
        "a shot of a person feeling pleasure: feeling of delight in the senses.",
    ],
    [
        "a photo of a person feeling sadness: feeling unhappy, sorrow, disappointed, or discouraged.",
        "a video of a person feeling sadness: feeling unhappy, sorrow, disappointed, or discouraged.",
        "a portrait of a person feeling sadness: feeling unhappy, sorrow, disappointed, or discouraged.",
        "an image showing a person feeling sadness: feeling unhappy, sorrow, disappointed, or discouraged.",
        "a close-up of a person feeling sadness: feeling unhappy, sorrow, disappointed, or discouraged.",
        "a face of a person feeling sadness: feeling unhappy, sorrow, disappointed, or discouraged.",
        "a picture of a person feeling sadness: feeling unhappy, sorrow, disappointed, or discouraged.",
        "a shot of a person feeling sadness: feeling unhappy, sorrow, disappointed, or discouraged.",
    ],
    [
        "a photo of a person feeling sensitivity: feeling of being physically or emotionally wounded, or feeling delicate or vulnerable.",
        "a video of a person feeling sensitivity: feeling of being physically or emotionally wounded, or feeling delicate or vulnerable.",
        "a portrait of a person feeling sensitivity: feeling of being physically or emotionally wounded, or feeling delicate or vulnerable.",
        "an image showing a person feeling sensitivity: feeling of being physically or emotionally wounded, or feeling delicate or vulnerable.",
        "a close-up of a person feeling sensitivity: feeling of being physically or emotionally wounded, or feeling delicate or vulnerable.",
        "a face of a person feeling sensitivity: feeling of being physically or emotionally wounded, or feeling delicate or vulnerable.",
        "a picture of a person feeling sensitivity: feeling of being physically or emotionally wounded, or feeling delicate or vulnerable.",
        "a shot of a person feeling sensitivity: feeling of being physically or emotionally wounded, or feeling delicate or vulnerable.",
    ],
    [
        "a photo of a person feeling suffering: psychological or emotional pain, distressed, or anguished.",
        "a video of a person feeling suffering: psychological or emotional pain, distressed, or anguished.",
        "a portrait of a person feeling suffering: psychological or emotional pain, distressed, or anguished.",
        "an image showing a person feeling suffering: psychological or emotional pain, distressed, or anguished.",
        "a close-up of a person feeling suffering: psychological or emotional pain, distressed, or anguished.",
        "a face of a person feeling suffering: psychological or emotional pain, distressed, or anguished.",
        "a picture of a person feeling suffering: psychological or emotional pain, distressed, or anguished.",
        "a shot of a person feeling suffering: psychological or emotional pain, distressed, or anguished.",
    ],
    [
        "a photo of a person feeling surprise: sudden discovery of something unexpected.",
        "a video of a person feeling surprise: sudden discovery of something unexpected.",
        "a portrait of a person feeling surprise: sudden discovery of something unexpected.",
        "an image showing a person feeling surprise: sudden discovery of something unexpected.",
        "a close-up of a person feeling surprise: sudden discovery of something unexpected.",
        "a face of a person feeling surprise: sudden discovery of something unexpected.",
        "a picture of a person feeling surprise: sudden discovery of something unexpected.",
        "a shot of a person feeling surprise: sudden discovery of something unexpected.",
    ],
    [
        "a photo of a person feeling sympathy: state of sharing others emotions, goals or troubles, supportive, or compassionate.",
        "a video of a person feeling sympathy: state of sharing others emotions, goals or troubles, supportive, or compassionate.",
        "a portrait of a person feeling sympathy: state of sharing others emotions, goals or troubles, supportive, or compassionate.",
        "an image showing a person feeling sympathy: state of sharing others emotions, goals or troubles, supportive, or compassionate.",
        "a close-up of a person feeling sympathy: state of sharing others emotions, goals or troubles, supportive, or compassionate.",
        "a face of a person feeling sympathy: state of sharing others emotions, goals or troubles, supportive, or compassionate.",
        "a picture of a person feeling sympathy: state of sharing others emotions, goals or troubles, supportive, or compassionate.",
        "a shot of a person feeling sympathy: state of sharing others emotions, goals or troubles, supportive, or compassionate.",
    ],
    [
        "a photo of a person feeling yearning: strong desire to have something, jealous, envious, or lust.",
        "a video of a person feeling yearning: strong desire to have something, jealous, envious, or lust.",
        "a portrait of a person feeling yearning: strong desire to have something, jealous, envious, or lust.",
        "an image showing a person feeling yearning: strong desire to have something, jealous, envious, or lust.",
        "a close-up of a person feeling yearning: strong desire to have something, jealous, envious, or lust.",
        "a face of a person feeling yearning: strong desire to have something, jealous, envious, or lust.",
        "a picture of a person feeling yearning: strong desire to have something, jealous, envious, or lust.",
        "a shot of a person feeling yearning: strong desire to have something, jealous, envious, or lust.",
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
