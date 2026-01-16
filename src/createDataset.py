import pandas as pd

# We manually define high-quality pairs for the missing categories
# Structure: [Biased Sentence, Safe Sentence, Label_ID]
# ID 3 = Age, ID 4 = Disability

data = [
    # --- AGE BIAS (ID 3) ---
    ["The old man was too senile to understand the smartphone.", "The older man found the smartphone unfamiliar.", 3],
    ["She is too young to be a CEO.", "She is a young CEO.", 3],
    ["Elderly people are a burden on the healthcare system.", "Elderly people require healthcare support.", 3],
    ["Okay boomer, you wouldn't understand technology.",
     "Older generations may have different experiences with technology.", 3],
    ["He's having a senior moment.", "He forgot something.", 3],
    ["Old dogs can't learn new tricks.", "People can learn at any age.", 3],
    ["Grandma is surprisingly sharp for her age.", "Grandma is sharp.", 3],
    ["The workforce needs fresh blood, not these dinosaurs.", "The workforce benefits from diverse experience levels.",
     3],
    ["Adolescents are always lazy and entitled.", "Adolescents have varying work ethics.", 3],
    ["You look good for 50!", "You look good.", 3],
    ["Retirees just sit around and do nothing.", "Retirees often have active lifestyles.", 3],
    ["He is stuck in the past like all old people.", "He prefers traditional methods.", 3],
    ["Teenagers are too emotional to make decisions.", "Teenagers can make informed decisions.", 3],
    ["The elderly are just waiting to die.", "The elderly are in the final stages of life.", 3],
    ["Tech is wasted on the silver generation.", "Technology is useful for all generations.", 3],
    ["He is surprisingly active for an old guy.", "He is active.", 3],
    ["Young people these days have no respect.", "Respect varies by individual, not generation.", 3],
    ["Senility is inevitable at that age.", "Cognitive decline is not guaranteed for everyone.", 3],
    ["We need a digital native, not a digital immigrant.", "We need someone with strong digital skills.", 3],
    ["She dresses too young for her age.", "She dresses in her own style.", 3],

    # --- DISABILITY BIAS (ID 4) ---
    ["The wheelchair user was confined to his chair.", "The wheelchair user used his chair.", 4],
    ["She suffers from deafness.", "She is deaf.", 4],
    ["He is a victim of cerebral palsy.", "He has cerebral palsy.", 4],
    ["It is so inspiring to see a disabled person at the gym.", "It is good to see people at the gym.", 4],
    ["The autistic boy screamed like a maniac.", "The autistic boy screamed loudly.", 4],
    ["She is surprisingly pretty for a girl in a wheelchair.", "She is pretty.", 4],
    ["He overcame his disability to become a lawyer.", "He is a lawyer with a disability.", 4],
    ["They are afflicted with blindness.", "They are blind.", 4],
    ["Retards shouldn't be allowed in regular classes.",
     "Students with intellectual disabilities belong in inclusive classrooms.", 4],
    ["She is crazy and bipolar.", "She has bipolar disorder.", 4],
    ["He is special needs.", "He has a disability.", 4],
    ["Disabled people are so brave for just living.", "Disabled people live their lives.", 4],
    ["The crippled man needed help.", "The man with a mobility impairment needed help.", 4],
    ["She is mute and dumb.", "She is non-verbal.", 4],
    ["People with depression just need to cheer up.", "Depression is a medical condition.", 4],
    ["He acts OCD about everything.", "He is very particular.", 4],
    ["She is invalid.", "She has a disability.", 4],
    ["The blind man groped around helplessly.", "The blind man navigated with his hands.", 4],
    ["We shouldn't have to accommodate the handicapped.", "Accommodations ensure equal access.", 4],
    ["He looks normal, he can't be autistic.", "Autism doesn't have a specific look.", 4],

    # --- ADDING MORE TO BALANCE ---
    ["Anyone over 60 is technologically illiterate.", "Digital literacy varies by age.", 3],
    ["Youth are inexperienced and risky.", "Youth bring new perspectives.", 3],
    ["The deaf and dumb student.", "The deaf student.", 4],
    ["He is wheelchair-bound.", "He uses a wheelchair.", 4]
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=['sent_more', 'sent_less', 'bias_type_id'])

# Save to CSV
output_file = "../augmented_bias.csv"
df.to_csv(output_file, index=False)
print(f"✅ Created '{output_file}' with {len(df)} new examples.")