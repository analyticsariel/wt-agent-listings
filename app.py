import streamlit as st
from annotated_text import annotated_text
import pandas as pd
import numpy as np
import plotly.express as px
import boto3
import gspread
import json
import tempfile
from oauth2client.service_account import ServiceAccountCredentials


##################################
#           FUNCTIONS            #
##################################
def read_gshseets_dataset(file_name):
    sheet = client.open(file_name).sheet1   
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    df['zpid'] = df.apply(lambda x: str(x['zpid']).split('.')[0], axis=1)
    df['num_users_label'] = df.apply(lambda x: get_total_user_labels(x), axis=1)
    df['label_category'] = df.apply(lambda x: get_label_category(x), axis=1)
    df['final_label'] = df.apply(lambda x: get_final_label(x), axis=1)
    return df, sheet


def get_total_user_labels(x):
    i = 0
    for c in ['ariel', 'liam', 'maddy']:
        if x[c] != None:
            if x[c] != '':
                i += 1
    return i

def get_label_category(x):
    labels = []
    for c in ['ariel', 'liam', 'maddy']:
        if x[c] != None:
            if x[c] != '':
                labels.append(x[c])
    if len(labels) == 0:
        return 'not labeled'
    if len(labels) == 1:
        return 'single label'
    if len(labels) == 2:
        if len(list(set(labels))) == 1:
            return 'confirmed label'
        else:
            return 'discrepancy'
    if len(labels) == 3:
        if len(list(set(labels))) == 2:
            return 'confirmed label'
        elif len(list(set(labels))) == 3:
            return 'discrepancy'
        
def get_final_label(x):
    labels = []
    for c in ['ariel', 'liam', 'maddy']:
        if x[c] != None:
            if x[c] != '':
                labels.append(x[c])

    if x['label_category'] == 'confirmed label':
        return max(labels,key=labels.count)
    else:
        return None
    
def read_json_file(bucket, key):
    """
    Reads json file from s3

    Args:
        bucket [string]: Bucket path
        key [string]: File path

    Returns:
        json object
    """
    
    # Initialize boto3 to use the S3 client.
    s3_client = boto3.client('s3', 
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY"],
          aws_secret_access_key=st.secrets["AWS_SECRET_KEY"])

    # Get the file inside the S3 Bucket
    s3_response = s3_client.get_object(
        Bucket=bucket,
        Key=key
    )

    # Get the Body object in the S3 get_object() response
    s3_object_body = s3_response.get('Body')

    # Read the data in bytes format
    content = s3_object_body.read()

    return json.loads(content)


##################################
#             SET UP             #
##################################
st.set_page_config(layout="wide")
st.title("ML Labeling Tool: Property Condition")
st.markdown("###### Label property description to determine condition")

# read gdrive credentials
config = read_json_file(bucket='residentialpropertydata', key='api/gdrive_creds.json')
tfile = tempfile.NamedTemporaryFile(mode="w+")
json.dump(config, tfile)
tfile.flush()

# define the scope
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']

# add credentials to the account
creds = ServiceAccountCredentials.from_json_keyfile_name(tfile.name, scope)

# authorize the clientsheet 
client = gspread.authorize(creds)

labeled_data_fn = "ml_training_prop_cond_dscr_20231228"


##################################
#              APP               #
##################################
tab1, tab2, tab3 = st.tabs(["Login", "Label", "Analytics"])


##################################
#             TAB 1              #
##################################
user_name_selection = tab1.radio(
    "Select User Name",
    [":gray[Non selected]", ":red[Ariel]", ":blue[Liam]", ":green[Maddy]"],
    # captions = ["", "Tampa Mamacita 💃", "Witty Brit 🇬🇧💂", "Cali Queen ☀️"]
    )

session_button = tab1.button("Start session", type="primary")
valid_session = False
if "user_name" not in st.session_state:
    st.session_state['user_name'] = ''
if session_button:
    # log labels
    st.session_state['user_labeled_zpid'] = {'Ariel': [], 'Liam': [], 'Maddy': []}
    for x in ['Non selected', 'Ariel', 'Liam', 'Maddy']:
        if x in user_name_selection:
            valid_session = True
            st.session_state['user_name'] = x
            if st.session_state['user_name'] in ['Ariel', 'Liam', 'Maddy']:
                with st.spinner('Session starting loading dataset...'):
                    # load dataset
                    st.session_state['dataset_initial'], st.session_state['sheet_name'] = read_gshseets_dataset(file_name=labeled_data_fn)
                    st.session_state['dataset_labeled'] = st.session_state['dataset_initial']
                tab1.write('Session started')


# sheet.update_cell(2,12,'updated')
# st.write(data[0])


##################################
#             TAB 2              #
##################################
user_name = st.session_state['user_name']
if (st.session_state['user_name'] in ['Ariel', 'Liam', 'Maddy']):
    tab2.write(f'Logged in as: {user_name}')

    # prepare dataset
    df = st.session_state['dataset_labeled']
    
    # df = df.loc[(~df['zpid'] == '') & (~df['description'].isnull())]
    df.loc[(df[user_name.lower()] == '') & (df['label_category'] == 'discrepancy'), 'label_priority'] = 4
    df.loc[(df[user_name.lower()] == '') & (df['label_category'] == 'single label'), 'label_priority'] = 3
    df.loc[(df['label_category'] == 'not labeled') & (df['fixer_upper_flag'] == 'TRUE'), 'label_priority'] = 2
    df.loc[(df['label_category'] == 'not labeled') & (df['fixer_upper_flag'] == 'FALSE'), 'label_priority'] = 1
    df.loc[(df['label_category'] == 'not labeled') & (df['proba_distressed'] <= 0.60) & (df['proba_distressed'] >= 0.40), 'low_confidence_v1_model'] = 1
    
    user_labeled_zpids_in_session = st.session_state['user_labeled_zpid'][user_name]
    if str(len(user_labeled_zpids_in_session)).split('.')[0][-1] in ['11']: # not using right now
        df = df.sort_values(by=['low_confidence_v1_model', 'proba_distressed', 'price_prct_diff'], ascending=True)
    elif str(len(user_labeled_zpids_in_session)).split('.')[0][-1] in ['7', '8', '9']:
        df = df\
            .sort_values(by=['price_prct_diff'], ascending=False)\
            .sort_values(by=['label_priority', 'proba_distressed'], ascending=True)
    else:
        df = df\
            .sort_values(by=['price_prct_diff'], ascending=False)\
            .sort_values(by=['label_priority', 'proba_distressed'], ascending=False)
    
    # if len(user_labeled_zpids_in_session) != 0:
    df = df.loc[~df['zpid'].isin(user_labeled_zpids_in_session)]

    # tab2.write(df.head())
    # tab2.write(user_labeled_zpids_in_session)

    # display
    col1, col2 = tab2.columns(2)

    col1.markdown("### Inferences")
    zpid = df.iloc[0]['zpid']
    model_clf = df.iloc[0]['fixer_upper_flag']
    model_proba = df.iloc[0]['proba_distressed']
    prop_dscr_raw = df.iloc[0]['description']
    street_address = df.iloc[0]['streetAddress']
    city = df.iloc[0]['city']
    # if model_clf == 'TRUE':
    #     col1.markdown(f"ZPID: {zpid}; Fixer upper model classification: **:red[{model_clf}]**")#; Model probability {model_proba}")
    # else:
    #     col1.markdown(f"ZPID: {zpid}; Fixer upper model classification: **:blue[{model_clf}]**")
    col1.markdown(f"ZPID: {zpid}; Address: {street_address}, {city}")

    col1.markdown("### Label")
    prop_cond_label = col1.radio(
        "Property condition based on description label",
        [":red[Distressed]", ":blue[Maintained]", ":green[Updated]", ":gray[Unknown]"],
        captions = ["Fixer upper.", "Well maintained but not recently updated.", "Updated / Remodeled.", "Not enough information to draw a conclusion."])
    sub_col1, sub_col2 = col1.columns(2)
    label_button = sub_col1.button("Label", type="primary")
    if label_button:
        st.session_state['user_labeled_zpid'][user_name].append(zpid)
        # write in gsheets
        df_initial = st.session_state['dataset_initial']
        idx = df_initial.index[df_initial['zpid'] == zpid].tolist()[0] + 2
        if user_name == 'Ariel':
            col_pos = 11
        elif user_name == 'Liam':
            col_pos = 12
        elif user_name == 'Maddy':
            col_pos = 13
        row_lbl = prop_cond_label.split('[')[-1].replace(']','').lower()
        st.session_state['sheet_name'].update_cell(idx,col_pos,row_lbl)
        # feedback to user
        sub_col1.write('Submitted')
    next_button = sub_col2.button("Next")
    if next_button:
        df = df.loc[~df['zpid'].isin(user_labeled_zpids_in_session)]
        # load dataset
        # st.session_state['dataset_labeled'] = read_gshseets_dataset(file_name="ml_training_prop_cond_description_20231228")


    col2.markdown("### Property Description")

    # with col2.expander("See property attributes", expanded=True):
    #     col2.write(df.head(1).reset_index(drop=True))
    
    # cleanse word and use keywords
    annotate_word_list = []

    distressed_keywords = [
        'asis', 'as-is', 'cash', 'cosmetic',
        'fix', 'flip', 'handyman', 'income', 'investor', 'investment',
        'opportunity', 'potential', 'quick', 
        'rehab', 'rent', 'repair', 'tenant', 'tlc', 'sell', 'unfinished'
    ]
                        
    maintained_keywords = ['maintain']

    updated_keywords = [
        'adorable', 'beautiful', 'charm', 'clean', 'entertain', 'gorgeous', 'granite',
        'hardwood', 'island', 'love', 'luxury', 'modern', 
        'new', 'quartz', 'ready', 'redone', 'remarkable', 'remodel', 'reno', 'stainless',
        'upgrade', 'update'
    ]
        
    for x in prop_dscr_raw.split(' '):
        # cleansing
        x = x.lower()

        w = x + ' ' 
        # distressed
        for kw in distressed_keywords:
            if (kw in w) and ('fixture' not in w) and ('different' not in w):
                w = (w, "Distressed", "#faa")
            # rm = ['fixture', 'different']
            # if (kw in w):
            #     for r in rm:
            #         if r not in w:
            #             w = (w, "Distressed", "#faa")
        # maintained
        for kw in maintained_keywords:
            if kw in w:
                w = (w, "Maintained", "#8ef")
        # updated
        for kw in updated_keywords:
            if (kw in w):
                w = (w, "Updated", "#afa")

        annotate_word_list.append(w)
    with col2:
        annotated_text(annotate_word_list)




##################################
#             TAB 3              #
##################################
metrics_button = tab3.button("View Metrics")
if (st.session_state['user_name'] in ['Ariel', 'Liam', 'Maddy']) and metrics_button:
    # 1. check what is labeled and if issues
    st.session_state['dataset_labeled'], st.session_state['sheet_name'] = read_gshseets_dataset(file_name=labeled_data_fn)
    df = st.session_state['dataset_labeled']
    tab3.markdown('## Summary of Data Labeled')
    df_label_cat = df.groupby(['label_category'])['zpid'].count().reset_index().rename(columns={'zpid': 'count'})
    label_cat_not_labeled = df_label_cat.loc[df_label_cat['label_category'] == 'not labeled']['count'].iloc[0]
    _df_cat_confirmed = df_label_cat.loc[df_label_cat['label_category'] == 'confirmed label']
    if len(_df_cat_confirmed) == 0:
        label_cat_confirmed = 0
    else:
        label_cat_confirmed = _df_cat_confirmed['count'].iloc[0]
    _df_cat_discrepancy = df_label_cat.loc[df_label_cat['label_category'] == 'discrepancy']
    if len(_df_cat_discrepancy) == 0:
        label_cat_discrepancy = 0
    else:
        label_cat_discrepancy = _df_cat_discrepancy['count'].iloc[0]
    _df_cat_single_label = df_label_cat.loc[df_label_cat['label_category'] == 'single label']
    if len(_df_cat_single_label) == 0:
        label_cat_single_label = 0
    else:
        label_cat_single_label = _df_cat_single_label['count'].iloc[0]

    tab3_col1, tab3_col2, tab3_col3, tab3_col4 = tab3.columns(4)
    tab3_col1.metric("Valid Labels", label_cat_confirmed)
    tab3_col2.metric("Single Labels", label_cat_single_label)
    tab3_col3.metric("Discrepancy Labels", label_cat_discrepancy)
    tab3_col4.metric("Not Labeled", label_cat_not_labeled)

    # labels by user
    tab3.markdown('## Breakdown')
    tab3_col1, tab3_col2, tab3_col3 = tab3.columns(3)
    tab3_col1.markdown('### Records Labeled by User')
    total_labeled_by_user = {}
    for c in ['ariel', 'liam', 'maddy']:
        total_labeled_by_user[c] = len([x for x in df[c].tolist() if x != ''])
    df_user_plot = pd.json_normalize(total_labeled_by_user).T.reset_index()\
        .rename(columns={'index': 'user', 0: 'num_records_labeled'})
    user_fig = px.bar(df_user_plot, x='user', y='num_records_labeled')
    tab3_col1.plotly_chart(user_fig, use_container_width=True)

    # categories labeled (pie)
    tab3_col2.markdown('### Categories Labeled Prct')
    df_sub_label = df.loc[df['label_category'] == 'confirmed label']\
        .groupby(['final_label'])['zpid'].count().reset_index()\
        .rename(columns={'zpid': 'count'})
    sub_label_fig = px.pie(df_sub_label, values='count', names='final_label')
    tab3_col2.plotly_chart(sub_label_fig, use_container_width=True)

    # categories labeled (bar)
    tab3_col3.markdown('### Categories Labeled Count')
    user_fig = px.bar(df_sub_label, x='final_label', y='count')
    tab3_col3.plotly_chart(user_fig, use_container_width=True)