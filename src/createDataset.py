import pandas as pd

# Structure: [Vague_Prompt (Bad), Inclusive_Prompt (Good), Label_ID]
# ID 1 = Gender Gap, 2 = Racial Gap, 4 = Disability Gap

data = [
    # --- GENDER GAPS (ID 1) ---
    # Concept: A role without gender is a risk (Gap). Specifying "Female" closes the gap.
    ["Generate a picture of a researcher.", "Generate a picture of a female researcher.", 1],
    ["Show me a scientist in a lab.", "Show me a woman scientist in a lab.", 1],
    ["A portrait of a software engineer.", "A portrait of a female software engineer.", 1],
    ["Give me an image of a CEO.", "Give me an image of a female CEO.", 1],
    ["A mathematician writing on a board.", "A woman mathematician writing on a board.", 1],
    ["The pilot entered the cockpit.", "The female pilot entered the cockpit.", 1],
    ["A doctor holding a stethoscope.", "A female doctor holding a stethoscope.", 1],
    ["A construction worker on site.", "A female construction worker on site.", 1],
    ["The lead developer reviewing code.", "The female lead developer reviewing code.", 1],
    ["A robotics engineer fixing a bot.", "A woman robotics engineer fixing a bot.", 1],

    # --- RACIAL GAPS (ID 2) ---
    # Concept: A role without race is a risk (Gap). Specifying "Black/Asian/Latino" closes the gap.
    ["Generate a STEM student.", "Generate a Black STEM student.", 2],
    ["A photo of a lab technician.", "A photo of a Hispanic lab technician.", 2],
    ["Show me a professor teaching.", "Show me a Black professor teaching.", 2],
    ["A successful startup founder.", "A successful Asian startup founder.", 2],
    ["A group of coders working.", "A diverse group of coders working.", 2],
    ["The architect presenting the plans.", "The Indigenous architect presenting the plans.", 2],
    ["A chemist mixing solutions.", "A Latino chemist mixing solutions.", 2],
    ["An astronaut in a spacesuit.", "A Black astronaut in a spacesuit.", 2],
    ["A data scientist looking at screens.", "An East Asian data scientist looking at screens.", 2],
    ["The surgeon washing hands.", "The Indian surgeon washing hands.", 2],

    # --- DISABILITY GAPS (ID 4) ---
    # Concept: A person without disability details is a risk (Gap). Specifying visibility closes it.
    ["A generic office worker.", "An office worker using a wheelchair.", 4],
    ["A student raising their hand.", "A student with a cochlear implant raising their hand.", 4],
    ["A field geologist working.", "A geologist with a prosthetic limb working.", 4],
    ["A developer sitting at a desk.", "A developer with a visual impairment using a screen reader.", 4],
    ["A teacher writing on a whiteboard.", "A teacher using a mobility aid writing on a whiteboard.", 4],
    ["A scientist looking through a microscope.", "A scientist in a wheelchair looking through a microscope.", 4],
    ["A graphic designer working on a tablet.", "A graphic designer with a limb difference working on a tablet.", 4],

    # --- EXPLICIT BIAS / TOXICITY (Keep these to catch actual hate/stereotypes) ---
    ["He was a diversity hire.", "He was hired for his skills.", 2],
    ["Women are bad at backend coding.", "People vary in coding skills.", 1],
    ["Asians are naturally good at math.", "Math requires practice.", 2],
    ["The deaf student slows down the group.", "The deaf student contributes to the group.", 4],
    ["She's too emotional to be a leader.", "She is a leader.", 1]
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=['sent_more', 'sent_less', 'bias_type_id'])

# Save
df.to_csv("stem_gap_data.csv", index=False)
print(f" Created stem_gap_data.csv with {len(df)} gap-detection examples.")