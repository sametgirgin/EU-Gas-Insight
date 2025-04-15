import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests  # For API calls
import google.generativeai as genai
import os

@st.cache_data
def load_gaspowerplants_data():
    return pd.read_excel('gaspowerplants.xlsx')

@st.cache_data
def load_lng_data():
    return pd.read_excel('eu_lng.xlsx')

@st.cache_data
def load_pipeline_data():
    return pd.read_excel('gaspipeline.xlsx')

@st.cache_data
def aggregate_lng_data(lng_df):
    return lng_df.groupby(['TerminalName', 'Latitude', 'Longitude']).agg(
        {
            'CapacityInMtpa': 'sum',
            'UnitName': lambda x: ', '.join(set(map(str, x.dropna()))),
            'Owner': lambda x: ', '.join(set(map(str, x.dropna()))),
            'Parent': lambda x: ', '.join(set(map(str, x.dropna()))),
            'ParentHQCountry': lambda x: ', '.join(set(map(str, x.dropna()))),
            'CapacityInBcm/y': 'sum',
            'ProposalYear': 'first',
            'Location': 'first'
        }
    ).reset_index()

# Use the cached functions
df = load_gaspowerplants_data()
lng_df = load_lng_data()
pipeline_df = load_pipeline_data()

# Ensure the CapacityInMtpa column is numeric
lng_df['CapacityInMtpa'] = pd.to_numeric(lng_df['CapacityInMtpa'], errors='coerce')

# Aggregate LNG data
aggregated_lng_df = aggregate_lng_data(lng_df)

# Replace NaN values with 0 (optional, depending on your use case)
lng_df['CapacityInMtpa'] = lng_df['CapacityInMtpa'].fillna(0)

# Read the status.txt file
try:
    with open('status.txt', 'r') as file:
        status_text = file.read()
except FileNotFoundError:
    status_text = "Status file not found."

# Sidebar for filters
#st.sidebar.header("Filter")

# Create tabs (Gemini tab removed)
tab1, tab2, tab3, tab4 = st.tabs([
    "EU O&G Power Plants Map and Charts",
    "EU LNG Terminals",
    "EU Gas Pipeline Map",
    "Dictionary"
])

# Tab 1: EU Oil & Gas Power Plants
with tab1:
    st.write("## EU Oil & Gas Power Plants Map")
    
    # Filters for gaspowerplants.xlsx
    st.sidebar.subheader("Filters for Oil & Gas Power Plants")
    country_options = ['All'] + df['Country/Area'].dropna().unique().tolist()
    fuel_options = ['All'] + df['Fuel'].dropna().unique().tolist()
    status_options = ['All'] + df['Status'].dropna().unique().tolist()
    technology_options = ['All'] + df['Turbine/Engine Technology'].dropna().unique().tolist()
    manufacturer_options = ['All'] + df['Equipment Manufacturer/Model'].dropna().unique().tolist()
    hydrogen_options = ['All'] + df['Hydrogen capable?'].dropna().unique().tolist()

    # Create filters in the sidebar as dropdowns
    selected_country = st.sidebar.selectbox('Country/Area', country_options)
    selected_fuel = st.sidebar.selectbox('Fuel', fuel_options)
    selected_status = st.sidebar.selectbox('Status', status_options)
    selected_technology = st.sidebar.selectbox('Turbine/Engine Technology', technology_options)
    selected_manufacturer = st.sidebar.selectbox('Equipment Manufacturer/Model', manufacturer_options)
    selected_hydrogen = st.sidebar.selectbox('Hydrogen Capable?', hydrogen_options)

    # Apply filters
    filtered_df = df[
        ((df['Country/Area'] == selected_country) | (selected_country == 'All')) &
        ((df['Fuel'] == selected_fuel) | (selected_fuel == 'All')) &
        ((df['Status'] == selected_status) | (selected_status == 'All')) &
        ((df['Turbine/Engine Technology'] == selected_technology) | (selected_technology == 'All')) &
        ((df['Equipment Manufacturer/Model'] == selected_manufacturer) | (selected_manufacturer == 'All')) &
        ((df['Hydrogen capable?'] == selected_hydrogen) | (selected_hydrogen == 'All'))
    ]

    # Ensure your DataFrame has the necessary columns
    required_columns = ['Latitude', 'Longitude', 'Plant name', 'Unit name', 'Start year', 'Retired year', 'Owner(s)', 'City']
    if all(column in filtered_df.columns for column in required_columns):
        # Aggregate capacity by Plant name, Latitude, and Longitude
        aggregated_df = filtered_df.groupby(['Plant name', 'Latitude', 'Longitude']).agg(
            {
                'Capacity (MW)': 'sum', 
                'Unit name': lambda x: ', '.join(map(str, x)),
                'Start year': 'first',
                'Retired year': 'first',
                'Owner(s)': lambda x: ', '.join(set(map(str, x))),
                'City': 'first'
            }
        ).reset_index()

        # Create a bubble map using plotly.express with Mapbox
        fig = px.scatter_mapbox(aggregated_df,
                                lat='Latitude',
                                lon='Longitude',
                                hover_name='Plant name',
                                hover_data={
                                    'Capacity (MW)': True, 
                                    'Unit name': True,
                                    'Start year': True,
                                    'Retired year': True,
                                    'Owner(s)': True,
                                    'City': True
                                },
                                size='Capacity (MW)',
                                color='Capacity (MW)',
                                color_continuous_scale=px.colors.cyclical.IceFire,
                                size_max=15,
                                zoom=3,
                                mapbox_style="carto-positron")

        # Update layout to make the map bigger
        fig.update_layout(
            mapbox_accesstoken='YOUR_MAPBOX_ACCESS_TOKEN',
            height=800,  # Set the height of the map
            width=900   # Set the width of the map
        )
        st.plotly_chart(fig)

    st.write("## EU Oil & Gas Power Plants Charts")

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart for top countries by total capacity
        country_capacity = filtered_df.groupby('Country/Area')['Capacity (MW)'].sum().reset_index()
        top_countries = country_capacity.sort_values(by='Capacity (MW)', ascending=False).head(10)
        bar_fig_countries = px.bar(top_countries, 
                                   x='Country/Area', 
                                   y='Capacity (MW)',
                                   title='Top 10 Countries/Areas by Total Capacity')
        
        bar_fig_countries.update_layout(
            xaxis_title="Country/Area",
            yaxis_title="Total Capacity (MW)",
            xaxis={'tickangle': 45}
        )
        
        st.plotly_chart(bar_fig_countries, use_container_width=True)

    with col2:
        # Bar chart for turbine/engine technology by total capacity
        technology_capacity = filtered_df[filtered_df['Turbine/Engine Technology'] != 'Unknown'].groupby('Turbine/Engine Technology')['Capacity (MW)'].sum().reset_index()
        technology_capacity = technology_capacity.sort_values(by='Capacity (MW)', ascending=False)
        bar_fig_tech = px.bar(technology_capacity,
                              x='Turbine/Engine Technology',
                              y='Capacity (MW)',
                              title='Turbine/Engine Technology by Total Capacity')
        
        bar_fig_tech.update_layout(
            xaxis_title="Turbine/Engine Technology",
            yaxis_title="Total Capacity (MW)",
            xaxis={'tickangle': 45}
        )
        
        st.plotly_chart(bar_fig_tech, use_container_width=True)

    # Bar chart for total capacity by status    
    status_capacity = filtered_df.groupby('Status')['Capacity (MW)'].sum().reset_index()
    status_capacity = status_capacity.sort_values(by='Capacity (MW)', ascending=False)
    bar_fig_status = px.bar(status_capacity,
                            x='Status',
                            y='Capacity (MW)',
                            title='Total Capacity by Status')
    
    bar_fig_status.update_layout(
        xaxis_title="Status",
        yaxis_title="Total Capacity (MW)",
        xaxis={'tickangle': 45}
    )
    
    st.plotly_chart(bar_fig_status, use_container_width=True)

# Tab 2: EU LNG Terminals
with tab2:
    st.sidebar.subheader("Filters for LNG Terminals")
    facility_type_options = ['All'] + lng_df['FacilityType'].dropna().unique().tolist()
    status_options = ['All'] + lng_df['Status'].dropna().unique().tolist()
    country_options = ['All'] + lng_df['Country'].dropna().unique().tolist()

    selected_facility_type = st.sidebar.selectbox('Facility Type', facility_type_options)
    selected_status = st.sidebar.selectbox('Status', status_options)
    selected_country = st.sidebar.selectbox('Country', country_options)

    filtered_lng_df = lng_df[
        ((lng_df['FacilityType'] == selected_facility_type) | (selected_facility_type == 'All')) &
        ((lng_df['Status'] == selected_status) | (selected_status == 'All')) &
        ((lng_df['Country'] == selected_country) | (selected_country == 'All'))
    ]

    required_columns_lng = ['Latitude', 'Longitude', 'TerminalName', 'CapacityInMtpa', 'UnitName', 'Status', 'Country', 'Owner', 'Parent', 'ParentHQCountry', 'CapacityInBcm/y', 'ProposalYear', 'Location']
    if all(column in lng_df.columns for column in required_columns_lng):
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
            height=800,
            width=900,
            title="EU LNG Terminals Bubble Map"
        )

        st.plotly_chart(lng_fig, use_container_width=True)
    else:
        st.error("The required columns for the LNG map are missing in the 'eu_lng.xlsx' file.")

    country_capacity = filtered_lng_df.groupby('Country')['CapacityInMtpa'].sum().reset_index()
    country_capacity = country_capacity.sort_values(by='CapacityInMtpa', ascending=False)

    bar_fig_lng = px.bar(
        country_capacity,
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
        xaxis={'tickangle': 45}
    )

    st.plotly_chart(bar_fig_lng, use_container_width=True)

# Tab 3: EU Gas Pipeline Map (using NewWKTFormat)
with tab3:
    # Sidebar filters for pipelines
    st.sidebar.subheader("Filters for European Gas Pipelines")
    fuel_options = ['All'] + pipeline_df['Fuel'].dropna().unique().tolist()
    status_options = ['All'] + pipeline_df['Status'].dropna().unique().tolist()
    end_country_options = ['All'] + pipeline_df['EndCountry'].dropna().unique().tolist()

    # Create filters in the sidebar
    selected_fuel = st.sidebar.selectbox('Fuel', fuel_options)
    selected_status = st.sidebar.selectbox('Status', status_options)
    selected_end_country = st.sidebar.selectbox('End Country', end_country_options)

    # Apply filters
    filtered_pipeline_df = pipeline_df[
        ((pipeline_df['Fuel'] == selected_fuel) | (selected_fuel == 'All')) &
        ((pipeline_df['Status'] == selected_status) | (selected_status == 'All')) &
        ((pipeline_df['EndCountry'] == selected_end_country) | (selected_end_country == 'All'))
    ]

    # Ensure the required columns exist
    required_columns_pipeline = [
        'PipelineName', 'Coordinates', 'Fuel', 'Countries', 'Status', 'Owner', 
        'StartYear1', 'CapacityBcm/y', 'CapacityBOEd', 'LengthKnownKm', 'StartLocation', 'EndCountry'
    ]
    if all(column in filtered_pipeline_df.columns for column in required_columns_pipeline):
        # Parse the WKTFormat column manually to extract coordinates
    
        def parse_wkt_linestring(wkt_string):
            """
            Converts a cleaned WKT string (no LINESTRING/MULTILINESTRING)
            to a list of (x, y) tuples.
            """
            if pd.isna(wkt_string):
                return []
            try:
                return [tuple(map(float, point.strip().split())) for point in wkt_string.split(",")]
            except:
                return []
        
        # Apply the parsing function to extract coordinates
        filtered_pipeline_df['coordinates'] = filtered_pipeline_df['NewWKTFormat'].apply(parse_wkt_linestring)
        # Flatten the coordinates for Plotly
        pipeline_data = []
        for _, row in filtered_pipeline_df.iterrows():
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
                    'StartLocation': row['StartLocation'],
                    'EndCountry': row['EndCountry']
                })

        pipeline_map_df = pd.DataFrame(pipeline_data)

        # Create a line map using Plotly Express
        fig = px.line_mapbox(
            pipeline_map_df,
            lat='Latitude',
            lon='Longitude',
            color='PipelineName',
            title="EU Gas Pipeline Map",
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
                'EndCountry': True,
                'Latitude': False,  # Hide latitude in hover data
                'Longitude': False  # Hide longitude in hover data
            }
        )

        # Update layout for better visualization
        fig.update_layout(
            height=800,
            width=900,
            mapbox_accesstoken='YOUR_MAPBOX_ACCESS_TOKEN',  # Replace with your Mapbox token
            showlegend=False  # Disable the legend
        )

        # Display the map
        st.plotly_chart(fig, use_container_width=True)
    else:
        missing_columns = [col for col in required_columns_pipeline if col not in filtered_pipeline_df.columns]
        st.error(f"The following required columns are missing in 'gaspipeline.xlsx': {missing_columns}")

# Tab 4: Dictionary
with tab4:
    try:
        with open('status.txt', 'r') as file:
            status_text = file.read()

        status_definitions = {}
        for line in status_text.split('\n'):
            if ':' in line:
                status, definition = line.split(':', 1)
                status_definitions[status.strip()] = definition.strip()

        status_df = pd.DataFrame({
            'Status': status_definitions.keys(),
            'Definition': status_definitions.values()
        })

        st.write("### Status Definitions")
        st.dataframe(status_df.style.hide(axis="index"))

    except FileNotFoundError:
        st.error("The file 'status.txt' was not found.")

    try:
        with open('turbinetech.txt', 'r') as file:
            turbine_text = file.read()

        turbine_definitions = {}
        for line in turbine_text.split('\n'):
            if ':' in line:
                tech_type, definition = line.split(':', 1)
                turbine_definitions[tech_type.strip()] = definition.strip()

        turbine_df = pd.DataFrame({
            'Technology Type': turbine_definitions.keys(),
            'Definition': turbine_definitions.values()
        })

        styled_turbine_df = turbine_df.style.set_table_styles(
            [{
                'selector': 'td',
                'props': [('white-space', 'normal'), ('word-wrap', 'break-word')]
            }]
        ).hide(axis="index")

        st.write("### Turbine Technology Definitions")
        st.dataframe(styled_turbine_df)

    except FileNotFoundError:
        st.error("The file 'turbinetech.txt' was not found.")

    try:
        with open('lng.txt', 'r') as file:
            lng_text = file.read()

        lng_definitions = {}
        for line in lng_text.split('\n'):
            if ':' in line:
                lng_term, definition = line.split(':', 1)
                lng_definitions[lng_term.strip()] = definition.strip()

        lng_df = pd.DataFrame({
            'LNG Term': lng_definitions.keys(),
            'Definition': lng_definitions.values()
        })

        styled_lng_df = lng_df.style.set_table_styles(
            [{
                'selector': 'td',
                'props': [('white-space', 'normal'), ('word-wrap', 'break-word')]
            }]
        ).hide(axis="index")

        st.write("### LNG Facility Types")
        st.dataframe(styled_lng_df)

    except FileNotFoundError:
        st.error("The file 'lng.txt' was not found.")
