import pandas as pd
import json
from datetime import datetime
import streamlit as st
import boto3

# # internal functions
# import api_data_functions as adf
# import property_functions as pf
# import s3_functions as sf
# import common_functions as cf

st.set_page_config(layout="wide")

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


def read_df_from_s3(bucket, key):
    
    return pd.read_csv(
        f's3://{bucket}/{key}',
        storage_options={
            "key":st.secrets["AWS_ACCESS_KEY"],
            "secret":st.secrets["AWS_SECRET_KEY"]
        }
    )

def get_latest_listings_dt(prefix='api/rapid_api/zillow/property_listings'):

    s3 = boto3.resource('s3', 
                        aws_access_key_id=st.secrets["AWS_ACCESS_KEY"],
                        aws_secret_access_key=st.secrets["AWS_SECRET_KEY"]
                        )
    objects = list(s3.Bucket('residentialpropertydata').objects.filter(Prefix=prefix))
    objects.sort(key=lambda o: o.last_modified)
    return objects[-1].key.split('/')[-1].split('_')[0] # return date in str format

def latest_sale_listing_dt(x):
    """
    Price changes

    Args:
        price_history_dict [dict]: price history changes

    Returns:
        latest sale listing date
    """
    if x['datePosted'] != None:
        return x['datePosted']
    else:
        try:
            price_history_dict = x['priceHistory']
            latest_listed_dt = [x for x in price_history_dict if (x['event'] in ['Listed for sale'])][0]['date']
            if len(latest_listed_dt) > 0:
                return latest_listed_dt
            else:
                return x['datePosted']
        except:
            return x['datePosted'] # invalid date


def get_days_on_zillow(latest_listings_dt, post_dt, x):
    if post_dt == None:
        post_dt = x['datePosted']
    # check if date exists
    if len(post_dt) >= 5:
        retrieved_dt = datetime.strptime(latest_listings_dt, '%Y%m%d')
        posted_dt = datetime.strptime(post_dt, '%Y-%m-%d')
        return (retrieved_dt - posted_dt).days
    # if not get days on zillow
    else:
        try:
            return x['resoFacts']['daysOnZillow']
        except:
            try:
                if 'hours' in x['timeOnZillow']:
                    return 0
                elif 'days' in x['timeOnZillow']:
                    return int(x['timeOnZillow'].split(' ')[0])
            except:
                return None


st.title('Agents 👩‍💼 with Active Listings 🏡')

# read all active cities
df_cities = read_df_from_s3('geographydata', "active_zip_city_state.csv")
df_cities['location_name'] = df_cities.apply(lambda x: x['city'].lower() + ', ' + x['state'].lower(), axis=1)
df_cities['parent_metro_region'] = df_cities.apply(lambda x: 
    ' '.join(x.capitalize() for x in x['city'].split(' ')) + ', ' + x['state'], axis=1)

city_option = st.selectbox(
    'Select a city 👇',
    df_cities['location_name'].tolist())

if 'initial_run' not in st.session_state:
    st.session_state['initial_run'] = False

if st.button('Run') or st.session_state['initial_run'] == True:
    st.session_state['initial_run'] = True
    tab1, tab2 = st.tabs(["City Search", "Agent"])
    with tab1:
        # get latest listing
        latest_listings_dt = get_latest_listings_dt()
        # latest_listings_dt = '20230928'
        year = latest_listings_dt[:4]
        month = latest_listings_dt[4:6]
        day = latest_listings_dt[-2:]

        city_id = df_cities.loc[df_cities['location_name'] == city_option]['city_id'].iloc[0].split('_')[-1]
        city = df_cities.loc[df_cities['location_name'] == city_option]['city'].iloc[0]
        state = df_cities.loc[df_cities['location_name'] == city_option]['state'].iloc[0]
        ###########################
        #        READ PROPS       #
        ###########################
        data_prep_fn = f'data_preparation/{year}/{month}/{day}/{latest_listings_dt}_dataprep_{city_id}_{city}_{state}.json'
        data_prep_json_raw = read_json_file(bucket='residentialpropertydata', key=data_prep_fn)
        data_prep_json = json.loads(data_prep_json_raw)
        st.session_state['dataprep'] = pd.DataFrame(data_prep_json) # TEMP
        df_props = st.session_state['dataprep']
        print(f'Starting run for {city}, {state} ({city_id}) for {len(df_props)} properties')

        # get agent info
        df_props['latest_listing_dt'] = df_props.apply(lambda x: latest_sale_listing_dt(x), axis=1)
        df_props['days_on_zillow'] = df_props.apply(
            lambda x: get_days_on_zillow(latest_listings_dt, x['latest_listing_dt'], x), axis=1)
        df_props['agent_name'] = df_props.apply(lambda x: x['attributionInfo']['agentName'], axis=1)
        df_props['agent_name_valid'] = df_props.apply(lambda x: False if (x['agent_name'] == None) else True, axis=1)
        df_props['agent_email'] = df_props.apply(lambda x: x['attributionInfo']['agentEmail'], axis=1)
        # phone number
        df_props['agent_phone_number'] = df_props.apply(lambda x: x['attributionInfo']['agentPhoneNumber'], axis=1)
        df_props['agent_phone_number_valid'] = df_props.apply(lambda x: False if (x['agent_phone_number'] == None) else True, axis=1)
        # phone number - check exists
        df_props = df_props.loc[df_props['agent_phone_number_valid'] == True]
        # phone number - check if digit
        df_props['agent_phone_number'] = df_props.apply(lambda x: x['agent_phone_number'].replace("-", ""), axis=1)
        df_props['agent_phone_number_valid'] = df_props.apply(lambda x: False if (x['agent_phone_number'].isdigit() == False) else True, axis=1)
        # profile url
        df_props['agent_profile_url'] = df_props.apply(lambda x: x['listed_by']['profile_url'] if 'profile_url' in x['listed_by'] else None, axis=1)
        # latest listing
        df_props['listing_link'] = df_props.apply(lambda x: 
            'https://www.coffeeclozers.com/properties/' + x['city_id'].split('_')[-1] + '/' + x['zpid_norm'], axis=1)

        ###########################
        #        READ PROPS       #
        ###########################
        df_filter = df_props.loc[
            (df_props['agent_name_valid'] == True) & 
            (df_props['agent_phone_number_valid'] == True) #&
            # (df_props['days_on_zillow'] <= 90)
        ]
        df_filter['total_listings'] = df_filter.groupby('agent_phone_number')['zpid_norm'].transform('count')
        df_filter['days_on_zillow_rank'] = df_filter.groupby("agent_phone_number")['days_on_zillow'].rank(method='first')
        df_filter = df_filter.loc[df_filter['days_on_zillow_rank'] == 1]
        df_filter['agent_first_name'] = df_filter.apply(lambda x: x['agent_name'].split(' ')[0], axis=1)
        df_filter = df_filter[[
                'agent_first_name', 'agent_name', 'agent_phone_number', 'agent_email', 'days_on_zillow', 'total_listings',
                'streetAddress', 'city', 'state', 'city_id', 'zpid_norm', 'listing_link', 'agent_profile_url']].\
            sort_values(by=['total_listings'], ascending=False).reset_index(drop=True)
        
        st.write(df_filter)

        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(df_filter)

        st.download_button(
        "Press to Download",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
        )

    with tab2:
        st.write('Search agent phone number 📱 to get all listings')
        agent_phone_number = st.text_input('Agent phone number')

        if st.button('Search'):
            cols_list = [
                'streetAddress', 'postal_code', 'price', 'bedrooms', 'bathrooms', 
                'derived_prop_type', 'days_on_zillow', 'listing_link']
            df_props_by_phone = df_props.loc[df_props['agent_phone_number'] == agent_phone_number]
            if len(df_props_by_phone) != 0:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(label="Number of Active Listings", value=len(df_props_by_phone))
                col2.metric(label="Avg Days on Market", value=int(df_props_by_phone['days_on_zillow'].mean()))
                col3.metric(label="Avg Price", value=int(df_props_by_phone['price'].mean()))
                col4.metric(label="Ratio Fixer Uppers", value=str(round(
                    len(df_props_by_phone.loc[df_props_by_phone['fixer_upper_flag'] == True]) / len(df_props_by_phone) *100, 2)) + '%')
                st.write(df_props_by_phone[cols_list].reset_index(drop=True))