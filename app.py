import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests  # For API calls
import google.generativeai as genai
import os

st.set_page_config(layout="wide")

# Create two columns for the header
left_col, right_col = st.columns([4, 1])

# Add title, subtitle, and logo
with left_col:
    st.markdown(
        """
        <h2 style="color: #007BFF; margin: 0;">Sustainable Energy Analytics</h2>
        <h3 style="color: #007BFF; margin: 0;">European Gas Report 🇪🇺</h>
        """,
        unsafe_allow_html=True
    )

with right_col:
    # Display the logo
    st.image("logo.png", width=30)  # Adjust the width as needed
# Configure the Gemini API with your API key from secrets
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("API Key not found. Please set the GOOGLE_API_KEY in secrets.toml.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Available models (you can choose others, like 'gemini-1.5-pro-latest')
MODEL_NAME = 'gemini-2.0-flash'
generation_config = genai.GenerationConfig(
    temperature=0.7,
    top_p=1.0,
    top_k=1,
    max_output_tokens=2048,
)
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

model = genai.GenerativeModel(model_name=MODEL_NAME,
                              generation_config=generation_config,
                              safety_settings=safety_settings)

@st.cache_data
def load_gaspowerplants_data():
    return pd.read_excel('gaspowerplants.xlsx')

@st.cache_data
def load_lng_data():
    return pd.read_excel('eu_lng.xlsx')

@st.cache_data
def load_pipeline_data():
    #return pd.read_excel('gaspipeline.xlsx') 
    return pd.read_excel('pipeline_map_data.xlsx')

# Load the oil and gas extraction data
@st.cache_data
def load_oil_and_gas_extraction_data():
    return pd.read_excel('oilandgasextraction.xlsx')

# Use the cached functions
df = load_gaspowerplants_data()
lng_df = load_lng_data()
pipeline_df = load_pipeline_data()
extraction_df = load_oil_and_gas_extraction_data()

# Ensure the CapacityInMtpa column is numeric
lng_df['CapacityInMtpa'] = pd.to_numeric(lng_df['CapacityInMtpa'], errors='coerce')


# Aggregate LNG data
#aggregated_lng_df = aggregate_lng_data(lng_df)

# Replace NaN values with 0 (optional, depending on your use case)
lng_df['CapacityInMtpa'] = lng_df['CapacityInMtpa'].fillna(0)

# Read the status.txt file
try:
    with open('status.txt', 'r') as file:
        status_text = file.read()
except FileNotFoundError:
    status_text = "Status file not found."

# Sidebar for filters

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Gas Reports","EU O&G Power Plants Map and Charts", "EU LNG Terminals", "EU Gas Pipeline Map", "Oil and Gas Extraction", "GasGPT", "Dictionary"])

with tab1:
    #st.write("## Gas Reports")

    # Load the gas_reports.xlsx file
    try:
        gas_reports_df = pd.read_excel('gas_reports.xlsx')

        # Ensure the required columns exist
        required_columns = ['Institutions', 'Report Category', 'Month', 'Link', 'Summary']
        if all(column in gas_reports_df.columns for column in required_columns):
            # Display the reports grouped by category
            for category, group in gas_reports_df.groupby('Institutions'):
                st.write(f"### {category}")
                for _, row in group.iterrows():
                    st.markdown(f"**Report**: {row['Report Category']} / {row['Month']}")
                    st.markdown(f"**Summary**: {row['Summary']}")
                    st.markdown(f"[Read the Report]({row['Link']})")
                    st.markdown("---")
        else:
            missing_columns = [col for col in required_columns if col not in gas_reports_df.columns]
            st.error(f"The following required columns are missing in 'gas_reports.xlsx': {missing_columns}")
    except FileNotFoundError:
        st.error("The file 'gas_reports.xlsx' was not found.")

# Tab 1: EU Oil & Gas Power Plants
with tab2:
    st.write("## EU Oil & Gas Power Plants Map")
    
    # Filters for gaspowerplants.xlsx
    st.sidebar.subheader("Filters for Oil & Gas Power Plants")
    country_options = ['All'] + sorted(df['Country/Area'].dropna().unique().tolist())
    fuel_options = ['All'] + sorted(df['Fuel'].dropna().unique().tolist())
    status_options = ['All'] + sorted(df['Status'].dropna().unique().tolist())
    technology_options = ['All'] + sorted(df['Turbine/Engine Technology'].dropna().unique().tolist())
    manufacturer_options = ['All'] + sorted(df['Equipment Manufacturer/Model'].dropna().unique().tolist())
    hydrogen_options = ['All'] + sorted(df['Hydrogen capable?'].dropna().unique().tolist())

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
            {'Capacity (MW)': 'sum', 
             'Unit name': lambda x: ', '.join(map(str, x)),
             'Start year': 'first',
             'Retired year': 'first',
             'Owner(s)': lambda x: ', '.join(set(map(str, x))),
             'City': 'first'}
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

    # Add bar charts to Tab 1
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
        
        # Update layout for better readability
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
        
        # Update layout for better readability
        bar_fig_tech.update_layout(
            xaxis_title="Turbine/Engine Technology",
            yaxis_title="Total Capacity (MW)",
            xaxis={'tickangle': 45}
        )
        
        st.plotly_chart(bar_fig_tech, use_container_width=True)

    # Add a new bar chart for total capacity by status    
    # Group data by status and calculate total capacity
    status_capacity = filtered_df.groupby('Status')['Capacity (MW)'].sum().reset_index()
    
    # Sort by capacity
    status_capacity = status_capacity.sort_values(by='Capacity (MW)', ascending=False)
    
    # Create a bar chart for status
    bar_fig_status = px.bar(status_capacity,
                            x='Status',
                            y='Capacity (MW)',
                            title='Total Capacity by Status')
    
    # Update layout for better readability
    bar_fig_status.update_layout(
        xaxis_title="Status",
        yaxis_title="Total Capacity (MW)",
        xaxis={'tickangle': 45}
    )
    
    # Display the bar chart
    st.plotly_chart(bar_fig_status, use_container_width=True)

# Tab 3: EU LNG Terminals
with tab3:
    # Sidebar filters for LNG Terminals
    st.sidebar.subheader("Filters for LNG Terminals")
    facility_type_options = ['All'] + sorted(lng_df['FacilityType'].dropna().unique().tolist())
    status_options = ['All'] + sorted(lng_df['Status'].dropna().unique().tolist())
    country_options = ['All'] + sorted(lng_df['Country'].dropna().unique().tolist())

    # Create filters in the sidebar
    selected_facility_type = st.sidebar.selectbox('Facility Type', facility_type_options)
    selected_status = st.sidebar.selectbox('Status', status_options)
    selected_country = st.sidebar.selectbox('Country', country_options)

    # Apply filters
    filtered_lng_df = lng_df[
        ((lng_df['FacilityType'] == selected_facility_type) | (selected_facility_type == 'All')) &
        ((lng_df['Status'] == selected_status) | (selected_status == 'All')) &
        ((lng_df['Country'] == selected_country) | (selected_country == 'All'))
    ]

    # Ensure the required columns exist
    required_columns_lng = ['Latitude', 'Longitude', 'TerminalName', 'CapacityInMtpa', 'UnitName', 'Status', 'Country', 'Owner', 'Parent', 'ParentHQCountry', 'CapacityInBcm/y', 'ProposalYear', 'Location']
    if all(column in filtered_lng_df.columns for column in required_columns_lng):
        # Aggregate the filtered data
        aggregated_filtered_lng_df = filtered_lng_df.groupby(['TerminalName', 'Latitude', 'Longitude']).agg(
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

        # Create a bubble map using plotly.express with Mapbox
        lng_fig = px.scatter_mapbox(
            aggregated_filtered_lng_df,
            lat='Latitude',
            lon='Longitude',
            hover_name='TerminalName',
            hover_data={
                'CapacityInMtpa': True,       # Show total capacity in Mtpa
                'UnitName': True,            # Show concatenated unit names
                'Owner': True,               # Show owner
                'Parent': True,              # Show parent
                'ParentHQCountry': True,     # Show parent HQ country
                'CapacityInBcm/y': True,     # Show total capacity in Bcm/y
                'ProposalYear': True,        # Show proposal year
                'Location': True,            # Show location
                'Latitude': False,           # Hide latitude in hover data
                'Longitude': False           # Hide longitude in hover data
            },
            size='CapacityInMtpa',  # Bubble size based on total capacity
            color='CapacityInMtpa',  # Color based on total capacity
            color_continuous_scale=px.colors.cyclical.IceFire,
            size_max=15,
            zoom=3,
            mapbox_style="carto-positron"
        )

        # Update layout to make the map visually appealing
        lng_fig.update_layout(
            mapbox_accesstoken='YOUR_MAPBOX_ACCESS_TOKEN',  # Replace with your Mapbox token
            height=800,  # Set the height of the map
            width=900,  # Set the width of the map
            title="EU LNG Terminals Bubble Map"
        )

        # Display the map
        st.plotly_chart(lng_fig, use_container_width=True)
    else:
        st.error("The required columns for the LNG map are missing in the 'eu_lng.xlsx' file.")

    # Add a bar chart for total LNG capacity by country
    country_capacity = filtered_lng_df.groupby('Country')['CapacityInMtpa'].sum().reset_index()
    country_capacity = country_capacity.sort_values(by='CapacityInMtpa', ascending=False)

    # Create the bar chart
    bar_fig_lng = px.bar(
        country_capacity,
        x='Country',
        y='CapacityInMtpa',
        title='Total LNG Capacity by Country',
        labels={'CapacityInMtpa': 'Total Capacity (Mtpa)', 'Country': 'Country'},
        color='CapacityInMtpa',
        color_continuous_scale=px.colors.sequential.Blues
    )

    # Update layout for better readability
    bar_fig_lng.update_layout(
        xaxis_title="Country",
        yaxis_title="Total Capacity (Mtpa)",
        xaxis={'tickangle': 45}
    )

    # Display the bar chart
    st.plotly_chart(bar_fig_lng, use_container_width=True)

# Tab 4: EU Gas Pipeline Map
with tab4:
    # Sidebar filters for pipelines
    st.sidebar.subheader("Filters for European Gas Pipelines")
    fuel_options = ['All'] + sorted(pipeline_df['Fuel'].dropna().unique().tolist())
    status_options = ['All'] + sorted(pipeline_df['Status'].dropna().unique().tolist())
    end_country_options = ['All'] + sorted(pipeline_df['EndCountry'].dropna().unique().tolist())

    # Create filters in the sidebar
    selected_fuel = st.sidebar.selectbox('Fuel', fuel_options)
    selected_status = st.sidebar.selectbox('Status', status_options)
    selected_end_country = st.sidebar.selectbox('End Country', end_country_options)

    # Add a dropdown filter for Countries
    country_options = [
        'All', 'Algeria', 'Austria', 'Benin', 'Bulgaria', 'France', 'Germany', 'Ghana', 'Greece', 'Guinea',
        'Guinea-Bissau', 'Iran', 'Ireland', 'Israel', 'Italy', 'Jordan', 'Latvia', 'Liberia', 'Libya', 'Morocco',
        'Netherlands', 'Nigeria', 'Portugal', 'Qatar', 'Saudi Arabia', 'Senegal', 'Spain', 'Switzerland', 'Syria', 'Türkiye'
    ]
    #country_options = sorted(country_options)  # Sort the country options
    selected_country = st.sidebar.selectbox('Pipeline Systems (via Country)', country_options)

    # Apply filters
    filtered_pipeline_df = pipeline_df[
        ((pipeline_df['Fuel'] == selected_fuel) | (selected_fuel == 'All')) &
        ((pipeline_df['Status'] == selected_status) | (selected_status == 'All')) &
        ((pipeline_df['EndCountry'] == selected_end_country) | (selected_end_country == 'All')) &
        ((pipeline_df['Countries'].str.contains(selected_country, case=False, na=False)) if selected_country != 'All' else True)
    ]

    # Ensure the required columns exist
    required_columns_pipeline = [
        'PipelineName', 'Fuel', 'Countries', 'Status', 'Owner', 
        'StartYear1', 'CapacityBcm/y', 'CapacityBOEd', 'LengthKnownKm', 'StartLocation', 'EndCountry'
    ]
    if all(column in filtered_pipeline_df.columns for column in required_columns_pipeline):
        """
        # Create a line map using Plotly Express
        fig = px.line_mapbox(
            filtered_pipeline_df,
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

        """

        # Display the table below the map with distinct rows and no index
        st.write("## Pipelines List")
        st.dataframe(
            filtered_pipeline_df[
                ['PipelineName', 'Fuel', 'Countries', 'Owner', 'CapacityBcm/y', 'CapacityBOEd', 'LengthKnownKm']
            ].drop_duplicates().reset_index(drop=True)  # Reset index and drop the old one
        )
    else:
        missing_columns = [col for col in required_columns_pipeline if col not in filtered_pipeline_df.columns]
        st.error(f"The following required columns are missing in 'gaspipeline.xlsx': {missing_columns}")

# Tab 4: Oil and Gas Extraction Map
with tab5:
    st.write("## Oil and Gas Extraction Map")

    # Sidebar filters for oil and gas extraction
    st.sidebar.subheader("Filters for Oil and Gas Extraction")
    country_options = ['All'] + sorted(extraction_df['Country/Area'].dropna().unique().tolist())
    status_options = ['All'] + sorted(extraction_df['Status'].dropna().unique().tolist())

    # Create filters in the sidebar
    selected_country = st.sidebar.selectbox('Country/Area', country_options)
    selected_status = st.sidebar.selectbox('Status', status_options)

    # Apply filters
    filtered_extraction_df = extraction_df[
        ((extraction_df['Country/Area'] == selected_country) | (selected_country == 'All')) &
        ((extraction_df['Status'] == selected_status) | (selected_status == 'All')) 
    ]

    # Ensure the required columns exist
    required_columns_extraction = ['Name', 'Status', 'Country/Area', 'Latitude', 'Longitude', 'Discovery year', 'Production start year', 'Operator', 'Total Hydrocarbon Prod (Mboe/y)']
    #required_columns_extraction = ['Name', 'Status', 'Country/Area', 'Latitude', 'Longitude', 'Discovery year', 'Production start year', 'Operator']
  
    if all(column in filtered_extraction_df.columns for column in required_columns_extraction):
        # Create a bubble map using Plotly Express with Mapbox
        extraction_fig = px.scatter_mapbox(
            filtered_extraction_df,
            lat='Latitude',
            lon='Longitude',
            hover_name='Name',
            hover_data={
                'Status': True,
                'Country/Area': True,
                'Operator': True,
                'Total Hydrocarbon Prod (Mboe/y)': True,
                'Latitude': False,
                'Longitude': False
            },
            color='Status',  # Color based on resource type
            color_continuous_scale=px.colors.cyclical.IceFire,
            size_max=15,
            zoom=3,
            mapbox_style="carto-positron"
        )

        # Update layout to make the map visually appealing
        extraction_fig.update_layout(
            mapbox_accesstoken='YOUR_MAPBOX_ACCESS_TOKEN',  # Replace with your Mapbox token
            height=800,  # Set the height of the map
            width=900,  # Set the width of the map
            title="Oil and Gas Extraction Map"
        )

        # Display the map
        st.plotly_chart(extraction_fig, use_container_width=True)
    else:
        st.error("The required columns for the Oil and Gas Extraction map are missing in the 'oilandgasextraction.xlsx' file.")

with tab6:
    st.markdown(
        "<h2 style='text-align: center; color: black;'>Chat with Google Gemini on EU Gas Data</h2>",
        unsafe_allow_html=True
    )
    # Input for user question
    user_prompt = st.text_area("Ask a question about gas power plants:", "What are the advantages of gas power plants?")

    if st.button("Generate Response"):
        if user_prompt:
            try:
                # Generate a response using the Gemini API
                response = model.generate_content(user_prompt)
                st.write("### Gemini's Response:")
                st.write(response.text)  # Access the generated text
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a question.")

with tab7:
    # Read and display the status.txt file
    try:
        with open('status.txt', 'r') as file:
            status_text = file.read()

        # Parse the status text into a dictionary
        status_definitions = {}
        for line in status_text.split('\n'):
            if ':' in line:
                status, definition = line.split(':', 1)
                status_definitions[status.strip()] = definition.strip()

        # Create a DataFrame from the status definitions
        status_df = pd.DataFrame({
            'Status': status_definitions.keys(),
            'Definition': status_definitions.values()
        })

        # Display the status DataFrame without an index
        st.write("### Status Definitions")
        st.dataframe(status_df.style.hide(axis="index"))

    except FileNotFoundError:
        st.error("The file 'status.txt' was not found.")

    # Read and display the turbine_tech.txt file
    try:
        with open('turbinetech.txt', 'r') as file:
            turbine_text = file.read()

        # Parse the turbine text into a dictionary
        turbine_definitions = {}
        for line in turbine_text.split('\n'):
            if ':' in line:
                tech_type, definition = line.split(':', 1)
                turbine_definitions[tech_type.strip()] = definition.strip()

        # Create a DataFrame from the turbine definitions
        turbine_df = pd.DataFrame({
            'Technology Type': turbine_definitions.keys(),
            'Definition': turbine_definitions.values()
        })

        # Apply styles to wrap text in the table
        styled_turbine_df = turbine_df.style.set_table_styles(
            [{
                'selector': 'td',
                'props': [('white-space', 'normal'), ('word-wrap', 'break-word')]
            }]
        ).hide(axis="index")

        # Display the turbine DataFrame with wrapped text
        st.write("### Turbine Technology Definitions")
        st.dataframe(styled_turbine_df)

    except FileNotFoundError:
        st.error("The file 'turbinetech.txt' was not found.")

    # Read and display the lng.txt file
    try:
        with open('lng.txt', 'r') as file:
            lng_text = file.read()

        # Parse the LNG text into a dictionary
        lng_definitions = {}
        for line in lng_text.split('\n'):
            if ':' in line:
                lng_term, definition = line.split(':', 1)
                lng_definitions[lng_term.strip()] = definition.strip()

        # Create a DataFrame from the LNG definitions
        lng_df = pd.DataFrame({
            'LNG Term': lng_definitions.keys(),
            'Definition': lng_definitions.values()
        })

        # Apply styles to wrap text in the table
        styled_lng_df = lng_df.style.set_table_styles(
            [{
                'selector': 'td',
                'props': [('white-space', 'normal'), ('word-wrap', 'break-word')]
            }]
        ).hide(axis="index")

        # Display the LNG DataFrame with wrapped text
        st.write("### LNG Facility Types")
        st.dataframe(styled_lng_df)

    except FileNotFoundError:
        st.error("The file 'lng.txt' was not found.")
