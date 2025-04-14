import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests
import google.generativeai as genai
import os

# Set page configuration for better layout
st.set_page_config(layout="wide", page_title="EU Gas Infrastructure Dashboard")

# Configure the Gemini API with your API key from secrets
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")

if not GOOGLE_API_KEY:
    st.error("API Key not found. Please set the GOOGLE_API_KEY in secrets.toml.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Configure Gemini model only once
MODEL_NAME = 'gemini-2.0-flash'
generation_config = genai.GenerationConfig(
    temperature=0.7,
    top_p=1.0,
    top_k=1,
    max_output_tokens=2048,
)
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Load model only once at app startup
@st.cache_resource
def load_model():
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
        safety_settings=safety_settings
    )

model = load_model()

# Improved data loading with better caching
@st.cache_data(ttl=3600)
def load_gaspowerplants_data():
    return pd.read_excel('gaspowerplants.xlsx')

@st.cache_data(ttl=3600)
def load_lng_data():
    return pd.read_excel('eu_lng.xlsx')

@st.cache_data(ttl=3600)
def load_pipeline_data():
    return pd.read_excel('gaspipeline.xlsx')

@st.cache_data(ttl=3600)
def load_text_file(filename):
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return f"File {filename} not found."

@st.cache_data
def aggregate_lng_data(lng_df):
    return lng_df.groupby(['TerminalName', 'Latitude', 'Longitude']).agg({
        'CapacityInMtpa': 'sum',
        'UnitName': lambda x: ', '.join(set(map(str, x.dropna()))),
        'Owner': lambda x: ', '.join(set(map(str, x.dropna()))),
        'Parent': lambda x: ', '.join(set(map(str, x.dropna()))),
        'ParentHQCountry': lambda x: ', '.join(set(map(str, x.dropna()))),
        'CapacityInBcm/y': 'sum',
        'ProposalYear': 'first',
        'Location': 'first'
    }).reset_index()

# Parse text files to dictionaries - cached for performance
@st.cache_data
def parse_text_to_dict(text):
    definitions = {}
    for line in text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            definitions[key.strip()] = value.strip()
    return definitions

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["EU O&G Power Plants Map and Charts", "EU LNG Terminals", "EU Gas Pipeline Map", "Dictionary", "GasGPT"])

# Load all data at once when the app starts
df = load_gaspowerplants_data()
lng_df = load_lng_data()
pipeline_df = load_pipeline_data()

# Ensure data types are correct
lng_df['CapacityInMtpa'] = pd.to_numeric(lng_df['CapacityInMtpa'], errors='coerce')
lng_df['CapacityInMtpa'] = lng_df['CapacityInMtpa'].fillna(0)

# Preprocess LNG data
aggregated_lng_df = aggregate_lng_data(lng_df)

# Load text files
status_text = load_text_file('status.txt')
turbine_text = load_text_file('turbinetech.txt')
lng_text = load_text_file('lng.txt')

# Create sidebar filters - outside of tabs for global availability
with st.sidebar:
    st.header("Filters")
    
    # Create containers for each tab's filters
    plant_filters = st.container()
    lng_filters = st.container()
    pipeline_filters = st.container()

    with plant_filters:
        st.subheader("Oil & Gas Power Plants Filters")
        country_options = ['All'] + sorted(df['Country/Area'].dropna().unique().tolist())
        fuel_options = ['All'] + sorted(df['Fuel'].dropna().unique().tolist())
        status_options = ['All'] + sorted(df['Status'].dropna().unique().tolist())
        technology_options = ['All'] + sorted(df['Turbine/Engine Technology'].dropna().unique().tolist())
        manufacturer_options = ['All'] + sorted(df['Equipment Manufacturer/Model'].dropna().unique().tolist())
        hydrogen_options = ['All'] + sorted(df['Hydrogen capable?'].dropna().unique().tolist())

        selected_plant_country = st.selectbox('Country/Area', country_options, key='plant_country')
        selected_plant_fuel = st.selectbox('Fuel', fuel_options, key='plant_fuel')
        selected_plant_status = st.selectbox('Status', status_options, key='plant_status')
        selected_plant_technology = st.selectbox('Turbine/Engine Technology', technology_options, key='plant_tech')
        selected_plant_manufacturer = st.selectbox('Equipment Manufacturer/Model', manufacturer_options, key='plant_manuf')
        selected_plant_hydrogen = st.selectbox('Hydrogen Capable?', hydrogen_options, key='plant_hydrogen')
    
    with lng_filters:
        st.subheader("LNG Terminals Filters")
        facility_type_options = ['All'] + sorted(lng_df['FacilityType'].dropna().unique().tolist())
        lng_status_options = ['All'] + sorted(lng_df['Status'].dropna().unique().tolist())
        lng_country_options = ['All'] + sorted(lng_df['Country'].dropna().unique().tolist())

        selected_facility_type = st.selectbox('Facility Type', facility_type_options, key='lng_facility')
        selected_lng_status = st.selectbox('Status', lng_status_options, key='lng_status')
        selected_lng_country = st.selectbox('Country', lng_country_options, key='lng_country')
    
    with pipeline_filters:
        st.subheader("Gas Pipeline Filters")
        pipeline_fuel_options = ['All'] + sorted(pipeline_df['Fuel'].dropna().unique().tolist())
        pipeline_status_options = ['All'] + sorted(pipeline_df['Status'].dropna().unique().tolist())

        selected_pipeline_fuel = st.selectbox('Fuel', pipeline_fuel_options, key='pipeline_fuel')
        selected_pipeline_status = st.selectbox('Status', pipeline_status_options, key='pipeline_status')

# Apply filters - moved outside tabs
# Power Plants filtering
filtered_df = df[
    ((df['Country/Area'] == selected_plant_country) | (selected_plant_country == 'All')) &
    ((df['Fuel'] == selected_plant_fuel) | (selected_plant_fuel == 'All')) &
    ((df['Status'] == selected_plant_status) | (selected_plant_status == 'All')) &
    ((df['Turbine/Engine Technology'] == selected_plant_technology) | (selected_plant_technology == 'All')) &
    ((df['Equipment Manufacturer/Model'] == selected_plant_manufacturer) | (selected_plant_manufacturer == 'All')) &
    ((df['Hydrogen capable?'] == selected_plant_hydrogen) | (selected_plant_hydrogen == 'All'))
]

# LNG filtering
filtered_lng_df = lng_df[
    ((lng_df['FacilityType'] == selected_facility_type) | (selected_facility_type == 'All')) &
    ((lng_df['Status'] == selected_lng_status) | (selected_lng_status == 'All')) &
    ((lng_df['Country'] == selected_lng_country) | (selected_lng_country == 'All'))
]

# Pipeline filtering
filtered_pipeline_df = pipeline_df[
    ((pipeline_df['Fuel'] == selected_pipeline_fuel) | (selected_pipeline_fuel == 'All')) &
    ((pipeline_df['Status'] == selected_pipeline_status) | (selected_pipeline_status == 'All'))
]

# Precompute aggregated data for power plants
if all(col in filtered_df.columns for col in ['Latitude', 'Longitude', 'Plant name', 'Unit name', 'Start year', 'Retired year', 'Owner(s)', 'City']):
    aggregated_plant_df = filtered_df.groupby(['Plant name', 'Latitude', 'Longitude']).agg({
        'Capacity (MW)': 'sum', 
        'Unit name': lambda x: ', '.join(map(str, x)),
        'Start year': 'first',
        'Retired year': 'first',
        'Owner(s)': lambda x: ', '.join(set(map(str, x))),
        'City': 'first'
    }).reset_index()
    
    # Precompute charts data
    country_capacity = filtered_df.groupby('Country/Area')['Capacity (MW)'].sum().reset_index()
    top_countries = country_capacity.sort_values(by='Capacity (MW)', ascending=False).head(10)
    
    technology_capacity = filtered_df[filtered_df['Turbine/Engine Technology'] != 'Unknown'].groupby('Turbine/Engine Technology')['Capacity (MW)'].sum().reset_index()
    technology_capacity = technology_capacity.sort_values(by='Capacity (MW)', ascending=False)
    
    status_capacity = filtered_df.groupby('Status')['Capacity (MW)'].sum().reset_index()
    status_capacity = status_capacity.sort_values(by='Capacity (MW)', ascending=False)
else:
    aggregated_plant_df = pd.DataFrame()
    top_countries = pd.DataFrame()
    technology_capacity = pd.DataFrame()
    status_capacity = pd.DataFrame()

# Precompute LNG charts data
country_lng_capacity = filtered_lng_df.groupby('Country')['CapacityInMtpa'].sum().reset_index()
country_lng_capacity = country_lng_capacity.sort_values(by='CapacityInMtpa', ascending=False)

# Parse pipeline WKT data
def parse_wkt_linestring(wkt_string):
    if isinstance(wkt_string, str) and wkt_string.startswith("LINESTRING"):
        coords = wkt_string.replace("LINESTRING (", "").replace(")", "").split(", ")
        return [tuple(map(float, coord.split())) for coord in coords]
    return []

# Precompute pipeline data
if all(col in filtered_pipeline_df.columns for col in ['PipelineName', 'WKTFormat', 'Fuel', 'Countries', 'Status', 'Owner', 'StartYear1', 'CapacityBcm/y', 'CapacityBOEd', 'LengthKnownKm', 'StartLocation']):
    # Apply the parsing function to extract coordinates
    filtered_pipeline_df['coordinates'] = filtered_pipeline_df['WKTFormat'].apply(parse_wkt_linestring)
    
    # Flatten the coordinates for Plotly - do this efficiently
    pipeline_data = []
    for _, row in filtered_pipeline_df[filtered_pipeline_df['coordinates'].apply(len) > 0].iterrows():
        for coord in row['coordinates']:
            pipeline_data.append({
                'PipelineName': row['PipelineName'],
                'Latitude': coord[1],
                'Longitude': coord[0],
                'Fuel': row['Fuel'],
                'Countries': row['Countries'],
                'Status': row['Status'],
                'Owner': row['Owner'],
                'StartYear1': row['StartYear1'],
                'CapacityBcm/y': row['CapacityBcm/y'],
                'CapacityBOEd': row['CapacityBOEd'],
                'LengthKnownKm': row['LengthKnownKm'],
                'StartLocation': row['StartLocation']
            })
    
    pipeline_map_df = pd.DataFrame(pipeline_data)
else:
    pipeline_map_df = pd.DataFrame()

# Parse dictionary files
status_dict = parse_text_to_dict(status_text)
turbine_dict = parse_text_to_dict(turbine_text)
lng_dict = parse_text_to_dict(lng_text)

# Tab 1: EU Oil & Gas Power Plants
with tab1:
    st.header("EU Oil & Gas Power Plants Map")
    
    if not aggregated_plant_df.empty:
        # Create a bubble map using plotly.express with Mapbox
        fig = px.scatter_mapbox(
            aggregated_plant_df,
            lat='Latitude',
            lon='Longitude',
            hover_name='Plant name',
            hover_data={
                'Capacity (MW)': True, 
                'Unit name': True,
                'Start year': True,
                'Retired year': True,
                'Owner(s)': True,
                'City': True,
                'Latitude': False,
                'Longitude': False
            },
            size='Capacity (MW)',
            color='Capacity (MW)',
            color_continuous_scale=px.colors.cyclical.IceFire,
            size_max=15,
            zoom=3,
            mapbox_style="carto-positron"
        )

        fig.update_layout(
            mapbox_accesstoken='YOUR_MAPBOX_ACCESS_TOKEN',
            height=600,
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.header("EU Oil & Gas Power Plants Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart for top countries by total capacity
            bar_fig_countries = px.bar(
                top_countries, 
                x='Country/Area', 
                y='Capacity (MW)',
                title='Top 10 Countries/Areas by Total Capacity'
            )
            
            bar_fig_countries.update_layout(
                xaxis_title="Country/Area",
                yaxis_title="Total Capacity (MW)",
                xaxis={'tickangle': 45},
                height=400
            )
            
            st.plotly_chart(bar_fig_countries, use_container_width=True)
        
        with col2:
            # Bar chart for turbine/engine technology by total capacity
            bar_fig_tech = px.bar(
                technology_capacity,
                x='Turbine/Engine Technology',
                y='Capacity (MW)',
                title='Turbine/Engine Technology by Total Capacity'
            )
            
            bar_fig_tech.update_layout(
                xaxis_title="Turbine/Engine Technology",
                yaxis_title="Total Capacity (MW)",
                xaxis={'tickangle': 45},
                height=400
            )
            
            st.plotly_chart(bar_fig_tech, use_container_width=True)
        
        # Bar chart for total capacity by status
        bar_fig_status = px.bar(
            status_capacity,
            x='Status',
            y='Capacity (MW)',
            title='Total Capacity by Status'
        )
        
        bar_fig_status.update_layout(
            xaxis_title="Status",
            yaxis_title="Total Capacity (MW)",
            xaxis={'tickangle': 45},
            height=400
        )
        
        st.plotly_chart(bar_fig_status, use_container_width=True)
    else:
        st.error("No data available or missing required columns in the power plants dataset.")

# Tab 2: EU LNG Terminals
with tab2:
    st.header("EU LNG Terminals Map")
    
    required_columns_lng = ['Latitude', 'Longitude', 'TerminalName', 'CapacityInMtpa', 'UnitName', 'Status', 'Country', 'Owner', 'Parent', 'ParentHQCountry', 'CapacityInBcm/y', 'ProposalYear', 'Location']
    
    if all(column in lng_df.columns for column in required_columns_lng):
        # Create a bubble map for LNG terminals
        lng_fig = px.scatter_mapbox(
            aggregated_lng_df,
            lat='Latitude',
            lon='Longitude',
            hover_name='TerminalName',
            hover_data={
                'CapacityInMtpa': True,
                'UnitName': True,
                'Owner': True,
                'Parent': True,
                'ParentHQCountry': True,
                'CapacityInBcm/y': True,
                'ProposalYear': True,
                'Location': True,
                'Latitude': False,
                'Longitude': False
            },
            size='CapacityInMtpa',
            color='CapacityInMtpa',
            color_continuous_scale=px.colors.cyclical.IceFire,
            size_max=15,
            zoom=3,
            mapbox_style="carto-positron"
        )

        lng_fig.update_layout(
            mapbox_accesstoken='YOUR_MAPBOX_ACCESS_TOKEN',
            height=600,
            margin={"r":0,"t":0,"l":0,"b":0}
        )

        st.plotly_chart(lng_fig, use_container_width=True)
        
        st.header("LNG Capacity Analysis")
        
        # Bar chart for total LNG capacity by country
        bar_fig_lng = px.bar(
            country_lng_capacity,
            x='Country',
            y='CapacityInMtpa',
            title='Total LNG Capacity by Country',
            labels={'CapacityInMtpa': 'Total Capacity (Mtpa)', 'Country': 'Country'},
            color='CapacityInMtpa',
            color_continuous_scale=px.colors.sequential.Blues
        )

        bar_fig_lng.update_layout(
            xaxis_title="Country",
            yaxis_title="Total Capacity (Mtpa)",
            xaxis={'tickangle': 45},
            height=400
        )

        st.plotly_chart(bar_fig_lng, use_container_width=True)
    else:
        st.error("Required columns for the LNG map are missing in the dataset.")

# Tab 3: EU Gas Pipeline Map
with tab3:
    st.header("EU Gas Pipeline Map")
    
    required_columns_pipeline = ['PipelineName', 'WKTFormat', 'Fuel', 'Countries', 'Status', 'Owner', 'StartYear1', 'CapacityBcm/y', 'CapacityBOEd', 'LengthKnownKm', 'StartLocation']
    
    if not pipeline_map_df.empty and all(col in filtered_pipeline_df.columns for col in required_columns_pipeline):
        # Create a line map using Plotly Express
        pipeline_fig = px.line_mapbox(
            pipeline_map_df,
            lat='Latitude',
            lon='Longitude',
            color='PipelineName',
            mapbox_style="carto-positron",
            zoom=3,
            hover_data={
                'PipelineName': True,
                'Fuel': True,
                'Countries': True,
                'Status': True,
                'Owner': True,
                'StartYear1': True,
                'CapacityBcm/y': True,
                'CapacityBOEd': True,
                'LengthKnownKm': True,
                'StartLocation': True,
                'Latitude': False,
                'Longitude': False
            }
        )

        pipeline_fig.update_layout(
            height=600,
            mapbox_accesstoken='YOUR_MAPBOX_ACCESS_TOKEN',
            showlegend=False,
            margin={"r":0,"t":0,"l":0,"b":0}
        )

        st.plotly_chart(pipeline_fig, use_container_width=True)
    else:
        missing_cols = [col for col in required_columns_pipeline if col not in filtered_pipeline_df.columns]
        if missing_cols:
            st.error(f"The following required columns are missing: {missing_cols}")
        else:
            st.error("No pipeline data available with the current filters.")

# Tab 4: Dictionary
with tab4:
    st.header("Reference Dictionaries")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Display status definitions
        if status_dict:
            st.subheader("Status Definitions")
            status_df = pd.DataFrame({
                'Status': status_dict.keys(),
                'Definition': status_dict.values()
            })
            st.dataframe(status_df, use_container_width=True)
        else:
            st.error("Status definitions not available.")
    
    with col2:
        # Display LNG facility types
        if lng_dict:
            st.subheader("LNG Facility Types")
            lng_def_df = pd.DataFrame({
                'LNG Term': lng_dict.keys(),
                'Definition': lng_dict.values()
            })
            st.dataframe(lng_def_df, use_container_width=True)
        else:
            st.error("LNG definitions not available.")
    
    # Display turbine technology definitions
    if turbine_dict:
        st.subheader("Turbine Technology Definitions")
        turbine_df = pd.DataFrame({
            'Technology Type': turbine_dict.keys(),
            'Definition': turbine_dict.values()
        })
        st.dataframe(turbine_df, use_container_width=True)
    else:
        st.error("Turbine technology definitions not available.")

# Tab 5: GasGPT
with tab5:
    st.header("Chat with Google Gemini on EU Gas Data")
    
    # Create a conversational memory to store the chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    user_prompt = st.chat_input("Ask a question about gas power plants:")
    
    if user_prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                response = model.generate_content(user_prompt)
                message_placeholder.markdown(response.text)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                message_placeholder.error(f"An error occurred: {e}")