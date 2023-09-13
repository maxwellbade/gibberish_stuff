#in case you need to reinstall libraries
# !pip install fuzzywuzzy Levenshtein git+https://github.com/casics/nostril.git boto3 delorean bs4 zenpy stripe slack slacker fastparquet plotly jupyterthemes pandasql gspread gspread_dataframe spam_lists nltk shap textdistance google-cloud-secret-manager seaborn

from my_set_up import *
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import pandas as pd
import itertools
from ast import literal_eval
import ast
import math
import re
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from collections import Counter
import collections
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)
    
def gibberish_text(num_display_rows,numdays,avg_fuzzy_score_lte,min_bad_lev_dist_pct_total,gibberish_threshold):
    start_time = datetime.now()
    
    #query function
    def query(query,rows=5):
        start_time = datetime.now()
        print('start time: ', start_time)
        query = query
        df = gcp_client.query(query).to_dataframe()
        end_time = datetime.now()
        print('script duration: {}'.format(end_time - start_time))
        print('\n original df: \n', df.shape)
        display(df.head(rows))
        return df

    print('import done')

    df = query(rows=num_display_rows
               ,query = """
        declare numdays INT64 DEFAULT""" + numdays + """;

        with targets as (
            select
                u.user_id as target_user_id
            from `prod-edw-1e61.common.dim_user` u 
            where
                timestamp(u.created_timestamp) >= timestamp_add(current_timestamp(), interval -numdays day)
                and (u.is_target = True
                    or u.is_disabled = True)
        )

        select
            a.user_id
            ,a.full_name
            ,a.email_handle
            ,a.is_target as target
        from (
            select
                m.user_id
                ,m.full_name
                ,m.email_handle
                ,length(m.full_name) as len_full_name
                ,length(m.email_handle) as len_email_handle
                ,m.is_target
            from (
                select
                    z.user_id
                    ,regexp_replace(z.full_name, r'[^a-zA-Z]', '') full_name
                    ,regexp_replace(z.email_handle, r'[^a-zA-Z]', '') email_handle
                    ,z.is_target
                from (
                    select
                        u.header.user_id
                        ,case when u.last_name is null and u.first_name is not null then lower(u.first_name)
                            when u.first_name is null and u.last_name is not null then lower(u.last_name)
                            when u.first_name is not null and u.last_name is not null then lower(concat(cast(u.first_name as string), cast(u.last_name as string),' '))
                            end as full_name
                        ,regexp_extract(u.username, r"^[a-za-z0-9_.+-]+") as email_handle
                        ,case when s.target_user_id is not null then 1 else 0 end as is_target
                        ,row_number() over (partition by u.header.user_id order by u.event_timestamp desc) as row_number
                    from `some_table` u
                    left join targets s on u.header.user_id = s.target_user_id
                    where
                        timestamp(u.event_timestamp) >= timestamp_add(current_timestamp(), interval -numdays day)
                        and u.event_timestamp is not null
                        and (u.first_name is not null
                            or u.last_name is not null)
                    ) z
                where
                    z.row_number = 1
                ) m
            where
                m.email_handle is not null
            ) a
        where
            a.len_full_name > 5
            and a.len_email_handle > 5
        """
    )

    ### Create new fuzzy match and gibberish/nonsense cols ###
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

    def is_gibberish(input_string, threshold=gibberish_threshold):
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

    #keep main cols
    df = df[['user_id','full_name','email_handle','target'
             ,'fuzzy_score','jaro_score','avg_score'
             ,'levenshtein_distance'
             ,'gibberish_email_handle','nonsense_email_handle'
             ,'gibberish_full_name','nonsense_full_name']]
    display('df with new cols:', df.shape, df.head(num_display_rows))

    # get riskiest lev distances
    percent = min_bad_lev_dist_pct_total
    which_col = 'levenshtein_distance'
    nunique_lev = df[which_col].nunique()
    nunique_max = df[which_col].max()
    pcnt_of_max_lev = int(round(percent * nunique_max,0))

    ### Create Dataframes ###
    #get username and email mismatches
    name_mismatch_df = df[
        (df['avg_score'] <= avg_fuzzy_score_lte)
        | (df['levenshtein_distance'] >= pcnt_of_max_lev)
    ].sort_values(by='avg_score',ascending=True)
    name_mismatch_df = name_mismatch_df[[
        'user_id','full_name','email_handle','target'
        ,'fuzzy_score','jaro_score','avg_score','levenshtein_distance'
    ]]
    display('name_mismatch_df:', name_mismatch_df.shape, name_mismatch_df.head(num_display_rows))

    #get nonsense emails/fullnames
    nonsense_df = df[
        (df['nonsense_email_handle'] == 1)
        | (df['nonsense_full_name'] == 1)
    ]
    nonsense_df = nonsense_df[[
        'user_id','full_name','email_handle','target'
        ,'nonsense_full_name','nonsense_email_handle'
    ]]
    display('nonsense_df:', nonsense_df.shape, nonsense_df.head(num_display_rows))
    
    #get gibberish emails/fullnames
    gibberish_df = df[
        (df['gibberish_email_handle'] == 1)
        | (df['gibberish_full_name'] == 1)
    ]
    gibberish_df = gibberish_df[[
        'user_id','full_name','email_handle','target'
        ,'gibberish_full_name','gibberish_email_handle'
    ]]
    display('gibberish_df:', gibberish_df.shape, gibberish_df.head(num_display_rows))
    
    end_time = datetime.now()
    print('script duration: {}'.format(end_time - start_time))
    return df, name_mismatch_df, nonsense_df, gibberish_df

df, name_mismatch_df, nonsense_df, gibberish_df = gibberish_text(
    num_display_rows = 5 #display rows in jupyter
    ,numdays = ' 30' #number of lookback days of registered users in query
    ,avg_fuzzy_score_lte = 50 #lower scores mean lower the match between fullname and email handle
    ,min_bad_lev_dist_pct_total = .6 #the percent of levenshtein_distance to grab above; so grabbing >= x percent of lev distances (total unique distances mayb be 30, .6 of 30 = 18
    ,gibberish_threshold = .7 #string gibberish entropy threshold
)

df['lev_dist_gte_5'] = np.where(df['levenshtein_distance'] >= 5,1,0)
df['lev_dist_gte_10'] = np.where(df['levenshtein_distance'] >= 10,1,0)
df['lev_dist_gte_15'] = np.where(df['levenshtein_distance'] >= 15,1,0)
df['lev_dist_gte_20'] = np.where(df['levenshtein_distance'] >= 20,1,0)
df['lev_dist_gte_25'] = np.where(df['levenshtein_distance'] >= 25,1,0)
df['lev_dist_gte_30'] = np.where(df['levenshtein_distance'] >= 30,1,0)
df['fuzzy_lte_50'] = np.where(df['fuzzy_score'] <= 50,1,0)
df['fuzzy_lte_45'] = np.where(df['fuzzy_score'] <= 45,1,0)
df['fuzzy_lte_40'] = np.where(df['fuzzy_score'] <= 40,1,0)
df['fuzzy_lte_35'] = np.where(df['fuzzy_score'] <= 35,1,0)
df['fuzzy_lte_30'] = np.where(df['fuzzy_score'] <= 30,1,0)
df['fuzzy_lte_25'] = np.where(df['fuzzy_score'] <= 25,1,0)
df['fuzzy_lte_20'] = np.where(df['fuzzy_score'] <= 20,1,0)
df['fuzzy_lte_15'] = np.where(df['fuzzy_score'] <= 15,1,0)
df['fuzzy_lte_10'] = np.where(df['fuzzy_score'] <= 10,1,0)
df['fuzzy_lte_5'] = np.where(df['fuzzy_score'] <= 5,1,0)
df['jaro_lte_50'] = np.where(df['jaro_score'] <= 50,1,0)
df['jaro_lte_45'] = np.where(df['jaro_score'] <= 45,1,0)
df['jaro_lte_40'] = np.where(df['jaro_score'] <= 40,1,0)
df['jaro_lte_35'] = np.where(df['jaro_score'] <= 35,1,0)
df['jaro_lte_30'] = np.where(df['jaro_score'] <= 30,1,0)
df['jaro_lte_25'] = np.where(df['jaro_score'] <= 25,1,0)
df['jaro_lte_20'] = np.where(df['jaro_score'] <= 20,1,0)
df['jaro_lte_15'] = np.where(df['jaro_score'] <= 15,1,0)
df['jaro_lte_10'] = np.where(df['jaro_score'] <= 10,1,0)
df['jaro_lte_5'] = np.where(df['jaro_score'] <= 5,1,0)
df['avg_score_lte_50'] = np.where(df['avg_score'] <= 50,1,0)
df['avg_score_lte_45'] = np.where(df['avg_score'] <= 45,1,0)
df['avg_score_lte_40'] = np.where(df['avg_score'] <= 40,1,0)
df['avg_score_lte_35'] = np.where(df['avg_score'] <= 35,1,0)
df['avg_score_lte_30'] = np.where(df['avg_score'] <= 30,1,0)
df['avg_score_lte_25'] = np.where(df['avg_score'] <= 25,1,0)
df['avg_score_lte_20'] = np.where(df['avg_score'] <= 20,1,0)
df['avg_score_lte_15'] = np.where(df['avg_score'] <= 15,1,0)
df['avg_score_lte_10'] = np.where(df['avg_score'] <= 10,1,0)
df['avg_score_lte_5'] = np.where(df['avg_score'] <= 5,1,0)
df['avg_score_gte_50'] = np.where(df['avg_score'] <= 50,1,0)
df['avg_score_gte_55'] = np.where(df['avg_score'] <= 55,1,0)
df['avg_score_gte_60'] = np.where(df['avg_score'] <= 60,1,0)
df['avg_score_gte_65'] = np.where(df['avg_score'] <= 65,1,0)
df['avg_score_gte_70'] = np.where(df['avg_score'] <= 70,1,0)
df['avg_score_gte_75'] = np.where(df['avg_score'] <= 75,1,0)
df['avg_score_gte_80'] = np.where(df['avg_score'] <= 80,1,0)
df['avg_score_gte_85'] = np.where(df['avg_score'] <= 85,1,0)
df['avg_score_gte_90'] = np.where(df['avg_score'] <= 90,1,0)
df['avg_score_gte_95'] = np.where(df['avg_score'] <= 95,1,0)

df['risky_nonsense_text'] = np.select([
    (df['nonsense_email_handle'] == 1 | (df['nonsense_full_name'] == 1))]
    ,[1]
    ,default=0
)

df['risky_nonsense_text_lev'] = np.select([
    (df['nonsense_email_handle'] == 1 | (df['nonsense_full_name'] == 1))
    & (df['lev_dist_gte_25'] == 1)]
    ,[1]
    ,default=0
)

df['risky_gibberish_text'] = np.select([
    (df['gibberish_email_handle'] == 1 | (df['gibberish_full_name'] == 1))]
    ,[1]
    ,default=0
)

df.head()
