NOMINAL_CATEGORICAL = [
    "_STATE",  # State FIPS code (1 to 72)
    "MARITAL",  # Marital status (1=Married, 2=Divorced, 3=Widowed, 4=Separated, 5=Never, 6=Unmarried couple)
    "EMPLOY1",  # Employment (1=Employed, 2=Self-employed, 3=Out of work 1yr+, 5=Homemaker, 7=Retired, etc.)
    "RENTHOM1",  # Home ownership (1=Own, 2=Rent, 3=Other)
    "EXRACT11",  # Primary physical activity type (1 to 76)
    "EXRACT21",  # Secondary physical activity type (1 to 76)
    "IMFVPLAC",  # Flu shot location (1=Doctor, 2=Health dept, 5=Store, 8=Work, etc.)
    "WHRTST10",  # Location of last HIV test (1=Doctor, 4=Clinic, etc.)
    "CRGVREL1",  # Caregiver relationship (1=Mother, 2=Father, 6=Husband, etc.)
    "CRGVPRB1",  # Caregiver recipient main health problem (1=Arthritis, 3=Cancer, etc.)
    "CRGVMST2",  # Most needed support service (1 to 6)
    "VINOCRE2",  # Reason did not visit eye doctor (1=Cost, 2=No doctor, 5=No problem, etc.)
    "PCPSARS1",  # Main reason for PSA test (1=Routine, 2=Problem, 3=Family history, etc.)
    "PCPSADE1",  # Who decided PSA test (1=Alone, 2=Doctor alone, 3=Together)
    "SCNTPAID",  # How paid for work (1=Salary, 2=Hourly, 3=Commission, 4=Other)
    "SCNTLPAD",  # How paid at last job
    "SXORIENT",  # Sexual orientation (1=Straight, 2=Gay, 3=Bisexual, 4=Other)
    "TRNSGNDR",  # Transgender status (1=MtF, 2=FtM, 3=Nonconforming, 4=No)
    "RCSGENDR",  # Child gender (1=Boy, 2=Girl)
    "RCSRLTN2",  # Relationship to child (1=Parent, 2=Grandparent, etc.)
    "MSCODE",  # Metro status (1=Center city, 2=Outside city, 3=Suburban, 5=Rural)
    "_PRACE1",  # Preferred race
    "_MRACE1",  # Multiracial race category
    "_RACE",  # Race-ethnicity grouping
    "_RACEGR3",  # 5-level race/ethnicity
    "_RACE_G1",  # Race group
    "_ASTHMS1",  # Computed asthma status (1=Current, 2=Former, 3=Never)
    "QSTVER",  # Survey questionnaire version
    "QSTLANG",  # Language (1=English, 2=Spanish)
    "_DUALUSE",  # Dual phone use category
    # Core Health Conditions
    "HLTHPLN1",
    "MEDCOST",
    "BPMEDS",
    "BLOODCHO",
    "TOLDHI2",
    "CVDINFR4",
    "CVDCRHD4",
    "CVDSTRK3",
    "ASTHMA3",
    "ASTHNOW",
    "CHCSCNCR",
    "CHCOCNCR",
    "CHCCOPD1",
    "HAVARTH3",
    "ADDEPEV2",
    "CHCKIDNY",
    # Demographics & Lifestyle
    "SEX",  # 1=Male, 2=Female
    "NUMHHOL2",  # Multiple phones (1=Yes, 2=No)
    "CPDEMO1",  # Cell phone personal use (1=Yes, 2=No)
    "VETERAN3",  # Veteran (1=Yes, 2=No)
    "INTERNET",  # Used internet past 30 days (1=Yes, 2=No)
    "PREGNANT",  # Pregnant (1=Yes, 2=No)
    "QLACTLM2",  # Activity limited by health (1=Yes, 2=No)
    "USEEQUIP",  # Special equipment required (1=Yes, 2=No)
    "BLIND",  # Blind or serious difficulty seeing (1=Yes, 2=No)
    "DECIDE",  # Difficulty concentrating (1=Yes, 2=No)
    "DIFFWALK",  # Difficulty walking/stairs (1=Yes, 2=No)
    "DIFFDRES",  # Difficulty dressing/bathing (1=Yes, 2=No)
    "DIFFALON",  # Difficulty errands alone (1=Yes, 2=No)
    "SMOKE100",  # Smoked 100 cigarettes (1=Yes, 2=No)
    "STOPSMK2",  # Tried to stop smoking (1=Yes, 2=No)
    "EXERANY2",  # Exercise in past 30 days (1=Yes, 2=No)
    "FLUSHOT6",  # Flu shot past 12 mos (1=Yes, 2=No)
    "PNEUVAC3",  # Pneumonia shot (1=Yes, 2=No)
    "HIVTST6",  # Ever tested HIV (1=Yes, 2=No)
    # Modules
    "PDIABTST",
    "PREDIAB1",
    "INSULIN",
    "DIABEYE",
    "DIABEDU",
    "CAREGIV1",
    "CRGVPERS",
    "CRGVHOUS",
    "CRGVEXPT",
    "VIINSUR2",
    "VIGLUMA2",
    "VIMACDG2",
    "CIMEMLOS",
    "CDDISCUS",
    "WTCHSALT",
    "DRADVISE",
    "ASATTACK",
    "HAREHAB1",
    "STREHAB1",
    "CVDASPRN",
    "RLIVPAIN",
    "RDUCHART",
    "RDUCSTRK",
    "LMTJOIN3",
    "ARTHDIS2",
    "ARTHWGT",
    "ARTHEXER",
    "ARTHEDU",
    "HPVADVC2",
    "SHINGLE2",
    "HADMAM",
    "HADPAP2",
    "HPVTEST",
    "HADHYST2",
    "PROFEXAM",
    "BLDSTOOL",
    "HADSIGM3",
    "PCPSAAD2",
    "PCPSADI1",
    "PCPSARE1",
    "PSATEST1",
    "CASTHDX2",
    "CASTHNO2",
    "MISTMNT",
    "ADANXEV",
    # CDC Calculated Binary Flags
    "_RFHLTH",
    "_HCVU651",
    "_RFHYPE5",
    "_RFCHOL",
    "_MICHD",
    "_LTASTH1",
    "_CASTHM1",
    "_DRDXAR1",
    "_HISPANC",
    "_RACEG21",
    "_AGE65YR",
    "_RFBMI5",
    "_RFSMOK3",
    "DRNKANY5",
    "_RFBING5",
    "_RFDRHV5",
    "_FRTLT1",
    "_VEGLT1",
    "_TOTINDA",
    "_PAINDX1",
    "_PA150R2",
    "_PA30021",
    "_PASTRNG",
    "_PASTAE1",
    "_RFSEAT2",
    "_RFSEAT3",
    "_FLSHOT6",
    "_PNEUMO2",
    "_AIDTST3",
]

ORDINAL_CATEGORICAL = [
    "GENHLTH",  # 1=Excellent ... 5=Poor
    "CHECKUP1",  # Routine checkup: 1=Past year, 2=Past 2yrs, 3=Past 5yrs, 4=5+ yrs, 8=Never
    "CHOLCHK",  # Cholesterol checkup recency
    "EDUCA",  # Education (1=Kindergarten ... 6=College grad)
    "_EDUCAG",  # 4-level education
    "INCOME2",  # Income brackets (1=<$10k ... 8=>$75k)
    "_INCOMG",  # 5-level income
    "_AGEG5YR",  # 14 age brackets (1=18-24 ... 13=80+)
    "_AGE_G",  # 6 age brackets
    "SMOKDAY2",  # Smoking frequency (1=Every day, 2=Some days, 3=Not at all)
    "USENOW3",  # Smokeless tobacco (1=Every day, 2=Some days, 3=Not at all)
    "SEATBELT",  # Seatbelt use (1=Always, 2=Nearly always, 3=Sometimes, 4=Seldom, 5=Never)
    "_BMI5CAT",  # 1=Underweight, 2=Normal, 3=Overweight, 4=Obese
    "_SMOKER3",  # 1=Every day, 2=Some days, 3=Former, 4=Never
    "_PACAT1",  # 1=Highly active, 2=Active, 3=Insufficiently active, 4=Inactive
    "ARTTODAY",  # Joint function (1=Can do everything ... 4=Hardly anything)
    "ARTHSOCL",  # Joint symptoms limit social (1=A lot, 2=A little, 3=Not at all)
    # Frequency scales (1=Always, 2=Usually, 3=Sometimes, 4=Rarely, 5=Never):
    "CDHOUSE",
    "CDASSIST",
    "CDHELP",
    "CDSOCIAL",
    "EMTSUPRT",
    "LSATISFY",
    "SCNTMNY1",
    "SCNTMEL1",
]

DROP_COLS = [
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_CLLCPWT",
    "_DUALCOR",
    "_LLCPWT",
    "EXACTOT1",
    "EXACTOT2",
]
