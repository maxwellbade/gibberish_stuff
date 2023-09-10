
from fuzzywuzzy import fuzz
import Levenshtein
import re
import math

df = df.fillna('nostring')
df['full_name'] = df['full_name'].str.replace(' ','none')
df['email_handle'] = df['email_handle'].str.replace(' ','none')
df['fuzzy_score'] = df[['full_name','email_handle']].apply(lambda x : fuzz.partial_ratio(*x),axis=1)
df['jaro_score'] = df[['full_name','email_handle']].apply(lambda x: Levenshtein.jaro_winkler(*x), axis=1) * 100
df['jaro_score'] = df['jaro_score'].astype(int)
df['levenshtein_distance'] = df[['full_name','email_handle']].apply(lambda x: Levenshtein.distance(*x), axis=1)
df['avg_score'] = (df['fuzzy_score'] + df['jaro_score']) / 2
df['avg_score'] = df['avg_score'].astype(int)

def is_gibberish(input_string, threshold=0.7):
    clean_string = re.sub(r'[^a-zA-Z]', '', input_string).lower()
    entropy = calculate_entropy(clean_string)
    if entropy < threshold:
        return 1
    else:
        return 0

def calculate_entropy(s):
    # Calculate the probability of each character
    prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
    # Calculate the entropy
    entropy = -sum(p * math.log2(p) for p in prob)
    return entropy

def is_nonsense(text):
    if is_gibberish(text):
        return 1
    else:
        return 0
        
df['gibberish_email_handle'] = df['email_handle'].apply(is_nonsense)
df['gibberish_full_name'] = df['full_name'].apply(is_nonsense)

#have to insalle this first!!!
# !pip install git+https://github.com/casics/nostril.git
from nostril import ng
from nostril import nonsense_detector as nd

def is_nonsense_text(text):
    if nd.nonsense(text):
        return 1
    else:
        return 0
    
df['nonsense_email_handle'] = df['email_handle'].apply(is_nonsense_text)
df['nonsense_full_name'] = df['full_name'].apply(is_nonsense_text)

df = df[['user_id','full_name','email_handle','is_softblocked'
         ,'fuzzy_score','jaro_score','avg_score'
         ,'levenshtein_distance'
         ,'gibberish_email_handle','nonsense_email_handle'
         ,'gibberish_full_name','nonsense_full_name']]

# get riskiest lev distances
percent = .6
which_col = 'levenshtein_distance'
nunique_lev = df[which_col].nunique()
nunique_max = df[which_col].max()
pcnt_of_max_lev = int(round(percent * nunique_max,0))

#get username and email mismatches
name_mismatch_df = df[
    (df['avg_score'] <= 50)
    | (df['levenshtein_distance'] >= pcnt_of_max_lev)
].sort_values(by='avg_score',ascending=True)
display('name_mismatch_df:', name_mismatch_df.shape, name_mismatch_df.head())

#get gibberish emails/usernames
gibberish_df = df[
    (df['nonsense_email_handle'] == 1)
    | (df['nonsense_full_name'] == 1)
]
display('gibberish_df:', gibberish_df.shape, gibberish_df.head())
