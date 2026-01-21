import pandas as pd
import json

# Label IDs: 0=Safe, 1=Gender, 2=Race, 3=Age, 4=Disability, 5=Profession

data = [
    # ========================================
    # MULTIPLE GAPS [1, 2, 4]
    # ========================================
    ["Generate a picture of a researcher.", "Generate a picture of a Black female researcher in a wheelchair.",
     [1, 2, 4]],
    ["Show me a scientist in a lab.", "Show me an Asian woman scientist with a prosthetic limb in a lab.", [1, 2, 4]],
    ["A portrait of a software engineer.", "A portrait of a Latino female software engineer using a wheelchair.",
     [1, 2, 4]],
    ["Give me an image of a CEO.", "Give me an image of a Black female CEO with a visual impairment.", [1, 2, 4]],
    ["A data scientist analyzing code.", "A South Asian female data scientist with hearing aids analyzing code.",
     [1, 2, 4]],
    ["An astronaut in space.", "A Black female astronaut with a prosthetic leg in space.", [1, 2, 4]],
    ["A professor teaching a class.", "An Indigenous woman professor using a wheelchair teaching a class.", [1, 2, 4]],
    ["A mechanical engineer designing parts.", "A Latina engineer with a limb difference designing parts.", [1, 2, 4]],
    ["The surgeon preparing for operation.", "The Black female surgeon with a mobility aid preparing for operation.",
     [1, 2, 4]],
    ["A chemist mixing solutions.", "An Asian woman chemist with low vision mixing solutions.", [1, 2, 4]],
    ["Show me a robotics expert.", "Show me a Black female robotics expert in a wheelchair.", [1, 2, 4]],
    ["An AI researcher at work.", "A Latina AI researcher with a cochlear implant at work.", [1, 2, 4]],
    ["A physicist working on equations.",
     "A Middle Eastern female physicist with a prosthetic arm working on equations.", [1, 2, 4]],
    ["A biomedical engineer in the lab.", "An Indigenous woman biomedical engineer using a wheelchair in the lab.",
     [1, 2, 4]],
    ["The architect reviewing blueprints.", "The Black female architect with a visual impairment reviewing blueprints.",
     [1, 2, 4]],
    ["A computer programmer coding.", "A South Asian female programmer with autism coding.", [1, 2, 4]],
    ["An electrical engineer testing circuits.",
     "A Latina electrical engineer with a hearing impairment testing circuits.", [1, 2, 4]],
    ["A geneticist studying DNA.", "An Asian female geneticist in a wheelchair studying DNA.", [1, 2, 4]],
    ["The lab technician preparing samples.",
     "The Black female lab technician with a prosthetic hand preparing samples.", [1, 2, 4]],
    ["A quantum physicist at the particle accelerator.",
     "An Indigenous woman quantum physicist with a mobility aid at the particle accelerator.", [1, 2, 4]],
    ["A neuroscientist conducting research.",
     "A Middle Eastern female neuroscientist with cerebral palsy conducting research.", [1, 2, 4]],
    ["An aerospace engineer designing rockets.",
     "A Black female aerospace engineer with a limb difference designing rockets.", [1, 2, 4]],
    ["A materials scientist in the cleanroom.", "A Latina materials scientist with low vision in the cleanroom.",
     [1, 2, 4]],
    ["The clinical researcher analyzing trials.",
     "The Asian female clinical researcher using a wheelchair analyzing trials.", [1, 2, 4]],
    ["A network security analyst.", "A Black female network security analyst with ADHD.", [1, 2, 4]],
    ["A researcher in the field.", "A Pacific Islander female researcher in a wheelchair in the field.", [1, 2, 4]],
    ["A software engineer coding.", "A Pacific Islander woman software engineer with low vision coding.", [1, 2, 4]],
    ["A data analyst.", "An East Asian female data analyst in a wheelchair.", [1, 2, 4]],
    ["A biologist studying samples.", "An East Asian woman biologist with a prosthetic arm studying samples.",
     [1, 2, 4]],
    ["A medical researcher.", "A Southeast Asian female medical researcher with hearing aids.", [1, 2, 4]],
    ["A chemical engineer.", "A Southeast Asian woman chemical engineer in a wheelchair.", [1, 2, 4]],
    ["A physicist in the lab.", "An Afro-Latina physicist with low vision in the lab.", [1, 2, 4]],
    ["An architect designing.", "An Afro-Latina architect in a wheelchair designing.", [1, 2, 4]],
    ["A researcher.", "A white female researcher in a wheelchair.", [1, 2, 4]],
    ["An engineer at work.", "A white woman engineer with a prosthetic limb at work.", [1, 2, 4]],

    # ========================================
    # DUAL GAPS: Gender + Race [1, 2]
    # ========================================
    ["A mathematician writing on a board.", "A Black female mathematician writing on a board.", [1, 2]],
    ["A doctor holding a stethoscope.", "An Asian female doctor holding a stethoscope.", [1, 2]],
    ["The pilot in the cockpit.", "The Latina pilot in the cockpit.", [1, 2]],
    ["A construction worker on site.", "A Black female construction worker on site.", [1, 2]],
    ["An engineer reviewing plans.", "An Indigenous woman engineer reviewing plans.", [1, 2]],
    ["A dentist examining a patient.", "A Middle Eastern female dentist examining a patient.", [1, 2]],
    ["The veterinarian with animals.", "The Asian female veterinarian with animals.", [1, 2]],
    ["A pharmacist dispensing medication.", "A Black female pharmacist dispensing medication.", [1, 2]],
    ["An accountant reviewing finances.", "A Latina accountant reviewing finances.", [1, 2]],
    ["The IT specialist fixing computers.", "The South Asian female IT specialist fixing computers.", [1, 2]],
    ["A geologist examining rocks.", "An Indigenous woman geologist examining rocks.", [1, 2]],
    ["An astronomer using a telescope.", "A Black female astronomer using a telescope.", [1, 2]],
    ["A meteorologist forecasting weather.", "An Asian female meteorologist forecasting weather.", [1, 2]],
    ["The botanist studying plants.", "The Latina botanist studying plants.", [1, 2]],
    ["A marine biologist diving.", "A Black female marine biologist diving.", [1, 2]],
    ["An environmental scientist in the field.", "A Middle Eastern female environmental scientist in the field.",
     [1, 2]],
    ["A civil engineer at a construction site.", "An Indigenous woman civil engineer at a construction site.", [1, 2]],
    ["The game developer coding.", "The Asian female game developer coding.", [1, 2]],
    ["A UX designer sketching wireframes.", "A Black female UX designer sketching wireframes.", [1, 2]],
    ["An industrial designer creating prototypes.", "A Latina industrial designer creating prototypes.", [1, 2]],
    ["A structural engineer analyzing buildings.", "A South Asian female structural engineer analyzing buildings.",
     [1, 2]],
    ["The database administrator.", "The Black female database administrator.", [1, 2]],
    ["A cybersecurity expert.", "An Asian female cybersecurity expert.", [1, 2]],
    ["An operations researcher.", "A Latina operations researcher.", [1, 2]],
    ["The quality assurance engineer.", "The Indigenous woman quality assurance engineer.", [1, 2]],

    # ========================================
    # DUAL GAPS: Gender + Disability [1, 4]
    # ========================================
    ["The pilot entered the cockpit.", "The female pilot using a mobility aid entered the cockpit.", [1, 4]],
    ["A research scientist at the bench.", "A female research scientist in a wheelchair at the bench.", [1, 4]],
    ["An engineer testing equipment.", "A female engineer with a prosthetic arm testing equipment.", [1, 4]],
    ["The developer writing code.", "The female developer with low vision writing code.", [1, 4]],
    ["A lab director overseeing experiments.", "A female lab director with hearing aids overseeing experiments.",
     [1, 4]],
    ["An anesthesiologist in the OR.", "A female anesthesiologist using a wheelchair in the OR.", [1, 4]],
    ["A cardiologist reviewing scans.", "A female cardiologist with a visual impairment reviewing scans.", [1, 4]],
    ["The project manager leading a team.", "The female project manager with autism leading a team.", [1, 4]],
    ["A field researcher collecting data.", "A female field researcher with a prosthetic leg collecting data.", [1, 4]],
    ["An oceanographer on a research vessel.", "A female oceanographer with a mobility aid on a research vessel.",
     [1, 4]],
    ["A clinical psychologist in session.", "A female clinical psychologist with ADHD in session.", [1, 4]],
    ["The pathologist examining specimens.", "The female pathologist with a limb difference examining specimens.",
     [1, 4]],
    ["A radiologist reading X-rays.", "A female radiologist in a wheelchair reading X-rays.", [1, 4]],
    ["An epidemiologist tracking diseases.", "A female epidemiologist with dyslexia tracking diseases.", [1, 4]],
    ["A software architect designing systems.", "A female software architect with cerebral palsy designing systems.",
     [1, 4]],
    ["The tech lead reviewing pull requests.",
     "The female tech lead with a hearing impairment reviewing pull requests.", [1, 4]],
    ["A product manager planning sprints.", "A female product manager using a wheelchair planning sprints.", [1, 4]],
    ["An information security analyst.", "A female information security analyst with low vision.", [1, 4]],
    ["A machine learning engineer training models.",
     "A female machine learning engineer with a prosthetic hand training models.", [1, 4]],
    ["The chief technology officer.", "The female chief technology officer with a mobility aid.", [1, 4]],
    ["A blockchain developer.", "A female blockchain developer with autism.", [1, 4]],
    ["An embedded systems engineer.", "A female embedded systems engineer with a cochlear implant.", [1, 4]],
    ["A cloud architect.", "A female cloud architect in a wheelchair.", [1, 4]],
    ["The DevOps engineer.", "The female DevOps engineer with ADHD.", [1, 4]],
    ["A full-stack developer.", "A female full-stack developer with a visual impairment.", [1, 4]],
    ["A white researcher.", "A white female researcher in a wheelchair.", [1, 4]],
    ["Show me a white scientist.", "Show me a white woman scientist with hearing aids.", [1, 4]],
    ["A white engineer.", "A white female engineer using a mobility aid.", [1, 4]],

    # ========================================
    # DUAL GAPS: Race + Disability [2, 4]
    # ========================================
    ["A medical researcher in the lab.", "A Black medical researcher in a wheelchair in the lab.", [2, 4]],
    ["An engineer at a design station.", "An Asian engineer with a prosthetic leg at a design station.", [2, 4]],
    ["The programmer debugging code.", "The Latino programmer with low vision debugging code.", [2, 4]],
    ["A scientist presenting findings.", "An Indigenous scientist with hearing aids presenting findings.", [2, 4]],
    ["An analyst reviewing data.", "A Middle Eastern analyst in a wheelchair reviewing data.", [2, 4]],
    ["A technician repairing instruments.", "A Black technician with a limb difference repairing instruments.", [2, 4]],
    ["The mathematician solving problems.", "The Asian mathematician with autism solving problems.", [2, 4]],
    ["A statistician analyzing results.", "A Latino statistician with cerebral palsy analyzing results.", [2, 4]],
    ["An architect designing buildings.", "An Indigenous architect with a mobility aid designing buildings.", [2, 4]],
    ["A financial analyst forecasting.", "A South Asian financial analyst with dyslexia forecasting.", [2, 4]],
    ["The bioinformatician coding pipelines.", "The Black bioinformatician with a visual impairment coding pipelines.",
     [2, 4]],
    ["A chemical engineer in the plant.", "An Asian chemical engineer in a wheelchair in the plant.", [2, 4]],
    ["An urban planner reviewing maps.", "A Latino urban planner with low vision reviewing maps.", [2, 4]],
    ["A systems analyst.", "A Middle Eastern systems analyst with a prosthetic arm.", [2, 4]],
    ["The biostatistician.", "The Black biostatistician with ADHD.", [2, 4]],
    ["A petroleum engineer in the field.", "An Indigenous petroleum engineer with a hearing impairment in the field.",
     [2, 4]],
    ["An agricultural scientist.", "A South Asian agricultural scientist using a wheelchair.", [2, 4]],
    ["A conservation biologist.", "a Latino conservation biologist with a cochlear implant.", [2, 4]],
    ["The toxicologist testing samples.", "The Asian toxicologist with a limb difference testing samples.", [2, 4]],
    ["A forensic scientist.", "A Black forensic scientist with autism.", [2, 4]],
    ["An automotive engineer.", "A Middle Eastern automotive engineer in a wheelchair.", [2, 4]],
    ["A naval architect.", "An Indigenous naval architect with a visual impairment.", [2, 4]],
    ["The mining engineer.", "The Latino mining engineer with a prosthetic leg.", [2, 4]],
    ["A textile engineer.", "An Asian textile engineer with cerebral palsy.", [2, 4]],
    ["A food scientist.", "A Black food scientist with dyslexia.", [2, 4]],
    ["A male researcher.", "A Black male researcher in a wheelchair.", [2, 4]],
    ["Generate a male scientist.", "Generate a Latino male scientist with a prosthetic arm.", [2, 4]],
    ["A male software engineer.", "An Asian male software engineer with low vision.", [2, 4]],

    # ========================================
    # NEW: GENDER DIVERSITY
    # ========================================
    ["A researcher.", "A Black non-binary researcher in a wheelchair.", [1, 2, 4]],
    ["A software developer.", "An Asian non-binary person developer with low vision.", [1, 2, 4]],
    ["A scientist in a lab.", "A Latino non-binary scientist with hearing aids in a lab.", [1, 2, 4]],
    ["An engineer testing equipment.", "A non-binary engineer in a wheelchair testing equipment.", [1, 4]],
    ["A data analyst reviewing data.", "A Black non-binary person analyst reviewing data.", [1, 2]],
    ["A researcher.", "A Black transgender woman researcher in a wheelchair.", [1, 2, 4]],
    ["A physicist working on equations.",
     "An Asian transgender woman physicist with a prosthetic arm working on equations.", [1, 2, 4]],
    ["A software engineer.", "a Latina transgender woman software engineer with low vision.", [1, 2, 4]],
    ["An architect.", "A transgender woman architect in a wheelchair.", [1, 4]],
    ["A biologist in the field.", "An Indigenous transgender woman biologist in the field.", [1, 2]],
    ["A researcher.", "A Black gender-fluid researcher in a wheelchair.", [1, 2, 4]],
    ["A chemist mixing solutions.", "An Asian gender-fluid person chemist with hearing aids mixing solutions.",
     [1, 2, 4]],
    ["A developer coding.", "A gender-fluid developer with low vision coding.", [1, 4]],
    ["A mathematician.", "A Latino gender-fluid person mathematician.", [1, 2]],
    ["A researcher.", "A Black queer researcher in a wheelchair.", [1, 2, 4]],
    ["An engineer at a design station.", "An Asian queer person engineer with a prosthetic leg at a design station.",
     [1, 2, 4]],
    ["A scientist presenting findings.", "A queer scientist with hearing aids presenting findings.", [1, 4]],
    ["A data scientist.", "A Latina queer data scientist.", [1, 2]],

    # ========================================
    # NEW: DISABILITY DIVERSITY
    # ========================================
    ["A Black female researcher.", "A Black female researcher using a white cane.", [4]],
    ["An Asian woman scientist.", "An Asian woman scientist using a white cane.", [4]],
    ["A researcher.", "A Black female researcher using a white cane.", [1, 2, 4]],
    ["A Latina engineer.", "A Latina engineer who is deaf and uses sign language.", [4]],
    ["A researcher.", "An Asian female researcher who is deaf and uses sign language.", [1, 2, 4]],
    ["An Indigenous woman scientist.", "An Indigenous woman scientist who is deaf and uses sign language.", [4]],
    ["A Black female researcher.", "A Black female researcher with vitiligo.", [4]],
    ["An Asian woman engineer.", "An Asian woman engineer with vitiligo.", [4]],
    ["A researcher.", "A Latina researcher with vitiligo.", [1, 2, 4]],
    ["A Black female scientist.", "A Black neurodivergent female scientist.", [4]],
    ["An Asian woman developer.", "An Asian neurodivergent woman developer.", [4]],
    ["A researcher.", "A Black neurodivergent female researcher.", [1, 2, 4]],

    # ========================================
    # NEW: AGE DIVERSITY
    # ========================================
    ["A researcher.", "An elderly Black female researcher in a wheelchair.", [1, 2, 3, 4]],
    ["A scientist in a lab.", "A senior Asian woman scientist with low vision in a lab.", [1, 2, 3, 4]],
    ["An engineer.", "A middle-aged Latina engineer with hearing aids.", [1, 2, 3, 4]],
    ["A developer coding.", "A young Indigenous woman developer in a wheelchair coding.", [1, 2, 3, 4]],
    ["A mathematician.", "A mature Middle Eastern female mathematician with a prosthetic arm.", [1, 2, 3, 4]],
    ["A researcher.", "An elderly Black female researcher.", [1, 2, 3]],
    ["A scientist.", "A senior Asian woman scientist.", [1, 2, 3]],
    ["An engineer.", "A young Latina engineer.", [1, 2, 3]],
    ["A developer.", "A middle-aged Indigenous woman developer.", [1, 2, 3]],
    ["A physicist.", "A mature South Asian female physicist.", [1, 2, 3]],
    ["A researcher.", "An elderly female researcher in a wheelchair.", [1, 3, 4]],
    ["A scientist.", "A senior woman scientist with low vision.", [1, 3, 4]],
    ["An engineer.", "A young female engineer with hearing aids.", [1, 3, 4]],
    ["A developer.", "A middle-aged woman developer with a prosthetic arm.", [1, 3, 4]],
    ["A researcher.", "An elderly Black researcher in a wheelchair.", [2, 3, 4]],
    ["A scientist.", "A senior Asian scientist with low vision.", [2, 3, 4]],
    ["An engineer.", "A young Latino engineer with hearing aids.", [2, 3, 4]],
    ["A Black female researcher in a wheelchair.", "An elderly Black female researcher in a wheelchair.", [3]],
    ["An Asian woman scientist with low vision.", "A senior Asian woman scientist with low vision.", [3]],
    ["A Latina engineer with hearing aids.", "A young Latina engineer with hearing aids.", [3]],
    ["An Indigenous woman developer with a prosthetic arm.",
     "A middle-aged Indigenous woman developer with a prosthetic arm.", [3]],
    ["A Middle Eastern female physicist in a wheelchair.", "A mature Middle Eastern female physicist in a wheelchair.",
     [3]],
    ["A Black female researcher.", "An elderly Black female researcher.", [3]],
    ["An Asian woman scientist.", "A young Asian woman scientist.", [3]],
    ["A Latina engineer.", "A senior Latina engineer.", [3]],
    ["An Indigenous woman developer.", "A middle-aged Indigenous woman developer.", [3]],

    # Safe examples with age (Labels handled in loop)
    ["An elderly Black female researcher in a wheelchair.", "", [0]],
    ["A senior Asian woman scientist with low vision.", "", [0]],
    ["A young Latina engineer with hearing aids.", "", [0]],
    ["A middle-aged Indigenous woman developer with a prosthetic arm.", "", [0]],
    ["A mature Middle Eastern female physicist in a wheelchair.", "", [0]],

    # ========================================
    # SINGLE GAP: Gender Only [1]
    # ========================================
    ["A Black lead developer reviewing code.", "A Black female lead developer reviewing code.", [1]],
    ["An Asian surgeon in the OR.", "An Asian female surgeon in the OR.", [1]],
    ["A Latino physicist running experiments.", "A Latina physicist running experiments.", [1]],
    ["An Indigenous astronomer stargazing.", "An Indigenous woman astronomer stargazing.", [1]],
    ["A Middle Eastern chemist.", "A Middle Eastern female chemist.", [1]],
    ["A Black engineer with a wheelchair.", "A Black female engineer with a wheelchair.", [1]],
    ["An Asian researcher with low vision.", "An Asian female researcher with low vision.", [1]],
    ["A Latino scientist in a wheelchair.", "A Latina scientist in a wheelchair.", [1]],
    ["An Indigenous developer with hearing aids.", "An Indigenous woman developer with hearing aids.", [1]],
    ["A South Asian mathematician.", "A South Asian female mathematician.", [1]],
    ["A Black pilot.", "A Black female pilot.", [1]],
    ["An Asian architect.", "An Asian female architect.", [1]],
    ["A Latino geologist.", "A Latina geologist.", [1]],
    ["An Indigenous biologist.", "An Indigenous woman biologist.", [1]],
    ["A Middle Eastern programmer.", "A Middle Eastern female programmer.", [1]],
    ["A Black engineer using a mobility aid.", "A Black female engineer using a mobility aid.", [1]],
    ["An Asian doctor with a prosthetic limb.", "An Asian female doctor with a prosthetic limb.", [1]],
    ["A Latino technician.", "A Latina technician.", [1]],
    ["An Indigenous analyst.", "An Indigenous woman analyst.", [1]],
    ["A South Asian veterinarian.", "A South Asian female veterinarian.", [1]],
    ["A Black pharmacist.", "A Black female pharmacist.", [1]],
    ["An Asian accountant.", "An Asian female accountant.", [1]],
    ["A Latino IT specialist.", "A Latina IT specialist.", [1]],
    ["An Indigenous meteorologist.", "An Indigenous woman meteorologist.", [1]],
    ["A Middle Eastern botanist.", "A Middle Eastern female botanist.", [1]],

    # ========================================
    # SINGLE GAP: Race Only [2]
    # ========================================
    ["A female construction worker on site.", "A Black female construction worker on site.", [2]],
    ["A woman engineer in the lab.", "An Asian woman engineer in the lab.", [2]],
    ["A female scientist presenting.", "A Latina scientist presenting.", [2]],
    ["A woman developer coding.", "An Indigenous woman developer coding.", [2]],
    ["A female mathematician teaching.", "A Middle Eastern female mathematician teaching.", [2]],
    ["A woman surgeon operating.", "A Black woman surgeon operating.", [2]],
    ["A female pilot flying.", "An Asian female pilot flying.", [2]],
    ["A woman chemist in the lab.", "A Latina chemist in the lab.", [2]],
    ["A female architect designing.", "An Indigenous woman architect designing.", [2]],
    ["A woman physicist researching.", "A South Asian woman physicist researching.", [2]],
    ["A female engineer with a wheelchair.", "A Black female engineer with a wheelchair.", [2]],
    ["A woman researcher with low vision.", "An Asian woman researcher with low vision.", [2]],
    ["A female scientist in a wheelchair.", "A Latina scientist in a wheelchair.", [2]],
    ["A woman developer with hearing aids.", "An Indigenous woman developer with hearing aids.", [2]],
    ["A female geologist.", "A Middle Eastern female geologist.", [2]],
    ["A woman astronomer.", "A Black woman astronomer.", [2]],
    ["A female biologist.", "An Asian female biologist.", [2]],
    ["A woman programmer.", "A Latina programmer.", [2]],
    ["A female technician.", "An Indigenous woman technician.", [2]],
    ["A woman analyst.", "A South Asian woman analyst.", [2]],
    ["A female veterinarian.", "A Black female veterinarian.", [2]],
    ["A woman pharmacist.", "An Asian female pharmacist.", [2]],
    ["A female accountant.", "A Latina accountant.", [2]],
    ["A woman IT specialist.", "An Indigenous woman IT specialist.", [2]],
    ["A female meteorologist.", "A Middle Eastern female meteorologist.", [2]],

    # ========================================
    # SINGLE GAP: Disability Only [4]
    # ========================================
    ["A Black female researcher at the bench.", "A Black female researcher in a wheelchair at the bench.", [4]],
    ["An Asian woman engineer testing.", "An Asian woman engineer with a prosthetic arm testing.", [4]],
    ["A Latina scientist analyzing data.", "A Latina scientist with low vision analyzing data.", [4]],
    ["An Indigenous woman developer coding.", "An Indigenous woman developer with hearing aids coding.", [4]],
    ["A Middle Eastern female mathematician.", "A Middle Eastern female mathematician with autism.", [4]],
    ["A Black woman surgeon in surgery.", "A Black woman surgeon with a mobility aid in surgery.", [4]],
    ["An Asian female pilot.", "An Asian female pilot with a prosthetic leg.", [4]],
    ["A Latina chemist in the lab.", "A Latina chemist with a visual impairment in the lab.", [4]],
    ["An Indigenous woman architect.", "An Indigenous woman architect in a wheelchair.", [4]],
    ["A South Asian female physicist.", "A South Asian female physicist with cerebral palsy.", [4]],
    ["A Black woman geologist.", "A Black woman geologist with a limb difference.", [4]],
    ["An Asian female astronomer.", "An Asian female astronomer with dyslexia.", [4]],
    ["A Latina biologist.", "A Latina biologist with a cochlear implant.", [4]],
    ["An Indigenous woman programmer.", "An Indigenous woman programmer with ADHD.", [4]],
    ["A Middle Eastern female technician.", "A Middle Eastern female technician in a wheelchair.", [4]],
    ["A Black woman analyst.", "A Black woman analyst with a hearing impairment.", [4]],
    ["An Asian female veterinarian.", "An Asian female veterinarian with a prosthetic hand.", [4]],
    ["A Latina pharmacist.", "A Latina pharmacist with low vision.", [4]],
    ["An Indigenous woman accountant.", "An Indigenous woman accountant with autism.", [4]],
    ["A South Asian female IT specialist.", "A South Asian female IT specialist using a wheelchair.", [4]],
    ["A Black woman meteorologist.", "A Black woman meteorologist with a visual impairment.", [4]],
    ["An Asian female botanist.", "An Asian female botanist with a mobility aid.", [4]],
    ["A Latina marine biologist.", "A Latina marine biologist with a prosthetic leg.", [4]],
    ["An Indigenous woman environmental scientist.", "An Indigenous woman environmental scientist with dyslexia.", [4]],
    ["A Middle Eastern female civil engineer.", "A Middle Eastern female civil engineer with cerebral palsy.", [4]],
    ["A white male researcher.", "A white male researcher in a wheelchair.", [4]],
    ["A white male scientist.", "A white male scientist with a prosthetic limb.", [4]],
    ["A white male engineer.", "A white male engineer with low vision.", [4]],
    ["A Black male researcher.", "A Black male researcher in a wheelchair.", [4]],
    ["An Asian male scientist.", "An Asian male scientist with hearing aids.", [4]],
    ["A Latino male engineer.", "A Latino male engineer using a wheelchair.", [4]],
    ["A white female researcher.", "A white female researcher in a wheelchair.", [4]],
    ["A white woman scientist.", "A white woman scientist with a prosthetic arm.", [4]],

    # ========================================
    # SAFE EXAMPLES [0] (Including Age)
    # ========================================
    ["An elderly Black female robotics engineer with a prosthetic arm fixing a bot.", "", [0]],
    ["A young Indigenous male scientist in a wheelchair conducting research.", "", [0]],
    ["A middle-aged Latina software engineer with low vision writing code.", "", [0]],
    ["A senior Asian female mathematician with hearing aids solving equations.", "", [0]],
    ["A mature Middle Eastern male physicist with a mobility aid in the lab.", "", [0]],
    ["An old South Asian female chemist in a wheelchair mixing solutions.", "", [0]],
    ["A young Black woman astronaut with a prosthetic leg in space.", "", [0]],
    ["An elderly Indigenous male doctor with a visual impairment examining patients.", "", [0]],
    ["A senior Latina engineer with autism designing systems.", "", [0]],
    ["A middle-aged Asian woman pilot with a limb difference flying.", "", [0]],
    ["A mature Black male researcher with cerebral palsy conducting experiments.", "", [0]],
    ["A young Middle Eastern woman developer with ADHD coding.", "", [0]],
    ["An old Indigenous female architect in a wheelchair designing buildings.", "", [0]],
    ["An elderly South Asian male surgeon with a prosthetic arm operating.", "", [0]],
    ["A senior Latina physicist with dyslexia teaching.", "", [0]],
    ["A middle-aged Asian male geologist with a cochlear implant in the field.", "", [0]],
    ["A young Black woman biologist with a mobility aid studying ecosystems.", "", [0]],
    ["A mature Indigenous female programmer with low vision developing software.", "", [0]],
    ["A old Middle Eastern man analyst in a wheelchair reviewing data.", "", [0]],
    ["A senior South Asian male veterinarian with hearing aids treating animals.", "", [0]],
    ["A Latina pharmacist with a prosthetic hand dispensing medication.", "", [0]],
    ["A young Asian woman accountant with autism managing finances.", "", [0]],
    ["A middle aged Black female IT specialist with a visual impairment fixing systems.", "", [0]],
    ["An old Indigenous woman meteorologist in a wheelchair forecasting weather.", "", [0]],
    ["A young Middle Eastern male botanist with a limb difference studying plants.", "", [0]],
    ["A Black researcher in a wheelchair.", "An elderly Black female researcher in a wheelchair.", [1, 3]],
    ["An Asian scientist with low vision.", "A senior Asian woman scientist with low vision.", [1, 3]],
    ["A Latino engineer with hearing aids.", "A young Latina engineer with hearing aids.", [1, 3]],
    ["An Indigenous developer with a prosthetic arm.",
     "A middle-aged Indigenous woman developer with a prosthetic arm.", [1, 3]],
    ["A Middle Eastern physicist in a wheelchair.", "A mature Middle Eastern female physicist in a wheelchair.",
     [1, 3]],
    ["A Black analyst with a mobility aid.", "An elderly Black female analyst with a mobility aid.", [1, 3]],
    ["A South Asian technician with autism.", "A young South Asian woman technician with autism.", [1, 3]],
    ["A Pacific Islander researcher using a white cane.",
     "A senior Pacific Islander female researcher using a white cane.", [1, 3]],
    ["An Afro-Latino scientist with cerebral palsy.", "A middle-aged Afro-Latina scientist with cerebral palsy.",
     [1, 3]],
    ["A Southeast Asian engineer with dyslexia.", "An elderly Southeast Asian woman engineer with dyslexia.", [1, 3]],
    ["A female researcher in a wheelchair.", "An elderly Black female researcher in a wheelchair.", [2, 3]],
    ["A woman scientist with low vision.", "A senior Asian woman scientist with low vision.", [2, 3]],
    ["A female engineer with hearing aids.", "A young Latina engineer with hearing aids.", [2, 3]],
    ["A woman developer with a prosthetic arm.", "A middle-aged Indigenous woman developer with a prosthetic arm.",
     [2, 3]],
    ["A female physicist in a wheelchair.", "A mature Middle Eastern female physicist in a wheelchair.", [2, 3]],
    ["A woman analyst with a mobility aid.", "An elderly South Asian woman analyst with a mobility aid.", [2, 3]],
    ["A female technician with autism.", "A young Pacific Islander female technician with autism.", [2, 3]],
    ["A woman researcher using a white cane.", "A senior Afro-Latina researcher using a white cane.", [2, 3]],
    ["A female scientist with cerebral palsy.", "A middle-aged Southeast Asian female scientist with cerebral palsy.",
     [2, 3]],
    ["A woman engineer with dyslexia.", "An elderly white woman engineer with dyslexia.", [2, 3]],
    ["A Black female researcher.", "An elderly Black female researcher in a wheelchair.", [3, 4]],
    ["An Asian woman scientist.", "A senior Asian woman scientist with low vision.", [3, 4]],
    ["A Latina engineer.", "A young Latina engineer with hearing aids.", [3, 4]],
    ["An Indigenous woman developer.", "A middle-aged Indigenous woman developer with a prosthetic arm.", [3, 4]],
    ["A Middle Eastern female physicist.", "A mature Middle Eastern female physicist in a wheelchair.", [3, 4]],
    ["A South Asian woman analyst.", "An elderly South Asian woman analyst with a mobility aid.", [3, 4]],
    ["A Pacific Islander female technician.", "A young Pacific Islander female technician with autism.", [3, 4]],
    ["An Afro-Latina researcher.", "A senior Afro-Latina researcher using a white cane.", [3, 4]],
    ["A Southeast Asian woman scientist.", "A middle-aged Southeast Asian woman scientist with cerebral palsy.",
     [3, 4]],
    ["A white female engineer.", "An elderly white female engineer with dyslexia.", [3, 4]],
    ["A Black woman chemist.", "A young Black woman chemist with a cochlear implant.", [3, 4]],
    ["An Asian female mathematician.", "A mature Asian female mathematician with ADHD.", [3, 4]],

    # ========================================
    # SEMI-SPECIFIED: AGE ONLY (Missing: Gender, Race, Disability)
    # Label: [Safe=0, Gender=1, Race=1, Age=0, Dis=1, Prof=0]
    # ========================================
    ["A young biologist in a lab.", "", [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]],
    ["An elderly professor teaching.", "", [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]],
    ["A senior CEO in a meeting.", "", [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]],
    ["A middle-aged engineer at work.", "", [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]],
    ["A mature doctor examining a patient.", "", [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]],
    ["An old artist painting.", "", [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]],
    ["A young student reading.", "", [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]],
    ["A senior architect drawing.", "", [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]],
    ["An elderly musician playing.", "", [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]],
    ["A middle-aged pilot flying.", "", [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]],

    # ========================================
    # SEMI-SPECIFIED: GENDER ONLY (Missing: Race, Age, Disability)
    # Label: [Safe=0, Gender=0, Race=1, Age=1, Dis=1, Prof=0]
    # ========================================
    ["A female researcher analysing data.", "", [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]],
    ["A male pilot flying a plane.", "", [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]],
    ["A woman developer coding.", "", [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]],
    ["A man architect designing.", "", [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]],
    ["A non-binary scientist in the field.", "", [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]],
    ["A woman surgeon operating.", "", [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]],
    ["A female doctor holding a chart.", "", [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]],
    ["A male engineer testing circuits.", "", [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]],
    ["A woman artist painting.", "", [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]],
    ["A man teacher writing on a board.", "", [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]],

    # ========================================
    # SEMI-SPECIFIED: RACE ONLY (Missing: Gender, Age, Disability)
    # Label: [Safe=0, Gender=1, Race=0, Age=1, Dis=1, Prof=0]
    # ========================================
    ["A Black doctor with a stethoscope.", "", [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]],
    ["An Asian engineer testing equipment.", "", [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]],
    ["A Latino teacher in a classroom.", "", [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]],
    ["An Indigenous researcher conducting a study.", "", [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]],
    ["A Middle Eastern programmer coding.", "", [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]],
    ["A South Asian chemist mixing solutions.", "", [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]],
    ["A Black lawyer arguing a case.", "", [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]],
    ["An Asian pilot checking instruments.", "", [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]],
    ["A White scientist in a lab coat.", "", [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]],
    ["A Pacific Islander artist working.", "", [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]],

    # ========================================
    # SEMI-SPECIFIED: DISABILITY ONLY (Missing: Gender, Race, Age)
    # Label: [Safe=0, Gender=1, Race=1, Age=1, Dis=0, Prof=0]
    # ========================================
    ["A researcher in a wheelchair.", "", [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]],
    ["A scientist with a prosthetic arm.", "", [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]],
    ["An engineer with hearing aids.", "", [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]],
    ["A developer using a white cane.", "", [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]],
    ["A doctor with low vision.", "", [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]],
    ["A pilot with a limb difference.", "", [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]],
    ["A student with a mobility aid.", "", [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]],
    ["A professional with vitiligo.", "", [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]],
    ["A teacher using sign language.", "", [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]],
    ["An artist with a cochlear implant.", "", [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]]
]


def create_multihot_vector(label_input, num_labels=6):
    # Check if input is already a vector (e.g. [0.0, 1.0...])
    # Standard labels are indices like [1, 2], so the first element is INT.
    # Semi-specified labels are [0.0, 1.0...], so the first element is FLOAT.

    if not label_input:
        # Empty list -> [0,0,0,0,0,0] (shouldn't happen with correct data)
        return [0.0] * num_labels

    if isinstance(label_input[0], float):
        # Already a vector
        return label_input

    # Create vector from indices
    vec = [0.0] * num_labels
    for label_id in label_input:
        vec[label_id] = 1.0
    return vec


rows = []
for vague, inclusive, label_data in data:
    # 1. Add Vague/Semi-Specified Entry
    if vague:
        rows.append({
            'text': vague,
            'labels': json.dumps(create_multihot_vector(label_data))
        })

    # 2. Add Inclusive Entry (always Safe [0])
    if inclusive:
        rows.append({
            'text': inclusive,
            'labels': json.dumps(create_multihot_vector([0]))
        })

# Create DataFrame and Save
df = pd.DataFrame(rows)
df.to_csv("generatedDataset.csv", index=False)
print(f"Created generatedDataset.csv with {len(df)} examples.")