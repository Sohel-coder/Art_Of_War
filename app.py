import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from PIL import Image
import base64
import os
import plotly.express as px
import plotly.graph_objects as go
import country_converter as coco

# Helper function to ensure dataframes are Arrow-compatible
def make_arrow_compatible(df):
    """Convert all columns to string to avoid Arrow conversion issues."""
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Convert all non-numeric columns to string
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].astype(str)
        # For numeric columns containing mixed data, also convert to string
        elif col != 'Year' and col != 'Metric':
            try:
                # Try to format numbers nicely
                df_copy[col] = df_copy[col].apply(
                    lambda x: "{:,}".format(int(x)) if isinstance(x, (int, float)) and abs(x) >= 1 and not pd.isna(x)
                    else (f"{x:.2f}" if isinstance(x, (int, float)) and not pd.isna(x) 
                          else ("N/A" if pd.isna(x) else str(x)))
                )
            except:
                # If that fails, just convert to string
                df_copy[col] = df_copy[col].astype(str)
    
    return df_copy

# Set page title and layout
st.set_page_config(page_title="Art of War - Military Data Analysis", layout="wide")

# Add custom CSS
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: black ;
        --secondary-color: #991B1B;
        --background-color: #F1F5F9;
        --text-color: #1E293B;
        --accent-color: #4F46E5;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: var(--primary-color);
        font-weight: 600;
    }
    
    h1 {
        font-size: 2.5rem;
        border-bottom: 3px solid var(--secondary-color);
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    h2 {
        font-size: 1.8rem;
        margin-top: 30px;
    }
    
    h3 {
        font-size: 1.3rem;
        margin-top: 20px;
    }
    
    /* Card-like containers */
    .stMarkdown, .stDataFrame, .stTable, .css-1r6slb0, div[data-testid="stMetricValue"] {
        background-color: black;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }
    
    /* Button styling */
    button[kind="primary"] {
        background-color: var(--primary-color);
        border-radius: 6px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    button[kind="primary"]:hover {
        background-color: var(--accent-color);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F8FAFC;
    }
    
    /* Hover effects for interactive elements */
    .stSelectbox:hover, .stNumberInput:hover {
        border-color: var(--accent-color);
    }
    
    /* Table styling */
    .dataframe {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 20px;
    }
    
    .dataframe th {
        background-color: var(--primary-color);
        color: white;
        font-weight: 600;
        text-align: left;
        padding: 12px;
    }
    
    .dataframe td {
        padding: 10px;
        border-bottom: 1px solid #E2E8F0;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: #F8FAFC;
    }
    
    .dataframe tr:hover {
        background-color: #E2E8F0;
    }
    
    /* Map container styling */
    .stPlotlyChart {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Image styling */
    .military-img {
        width: 100%;
        max-width: 800px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease;
    }
    
    .military-img:hover {
        transform: scale(1.02);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: var(--accent-color);
    }
    
    /* Container styling */
    .container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Add navigation, but hide it initially if on welcome page
# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Welcome", "World Map", "Military Strength", "Defense Budget", "Defense Companies", "Trade Data", "2047 Predictions"])

# Title
st.title("Art of War - Military Data Analysis")

# Load datasets
@st.cache_data
def load_data():
    military_strength = pd.read_csv("2024_military_strength_by_country.csv")
    defense_budget = pd.read_csv("Defence_budget_cleaned.csv")
    defense_companies = pd.read_csv("defence_companies_from_2005_final.csv")
    exports_imports = pd.read_csv("exports_imports_cleaned.csv")
    return military_strength, defense_budget, defense_companies, exports_imports

military_strength, defense_budget, defense_companies, exports_imports = load_data()

# Welcome page
if page == "Welcome":
    # Add welcome header
    st.header("Welcome to the Art of War - Military Data Analysis Platform")
    
    # Create two columns layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display the military image using st.image instead of HTML
        military_image_url = "https://www.armyrecognition.com/images/stories/north_america/united_states/military_equipment/uh-60_black_hawk/UH-60_Black_Hawk_United_States_US_American_army_aviation_helicopter_001.jpg"
        
        try:
            # Use st.image with proper error handling
            st.image(military_image_url, use_container_width=True, caption="Modern Military Forces - The Art of War")
        except Exception as e:
            # Display fallback text if image fails to load
            st.error(f"Unable to load image. Please check your internet connection.")
            st.write("Modern Military Forces - The Art of War")
        
        # Add introductory text with enhanced styling
        st.markdown( """
        <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h2 style="color: var(--primary-color); margin-top: 0;">Explore Global Military Power Analysis</h2>
            
            <p style="font-size: 1.1rem; line-height: 1.6;">
                This interactive dashboard provides comprehensive analysis of global military powers, defense budgets,
                major defense companies, and international military trade data.
            </p>
            
            <h3 style="color: var(--primary-color);">Available Analysis:</h3>
            
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 15px;">
                <div style="background-color: #F1F5F9; padding: 15px; border-radius: 6px; border-left: 4px solid var(--primary-color);">
                    <h4 style="margin-top: 0; color: var(--primary-color);">Military Strength Comparison</h4>
                    <p>Compare military capabilities between countries</p>
                </div>
                <div style="background-color: #F1F5F9; padding: 15px; border-radius: 6px; border-left: 4px solid var(--secondary-color);">
                    <h4 style="margin-top: 0; color: var(--secondary-color);">Defense Budget Analysis</h4>
                    <p>Track defense expenditure trends over time</p>
                </div>
                <div style="background-color: #F1F5F9; padding: 15px; border-radius: 6px; border-left: 4px solid var(--accent-color);">
                    <h4 style="margin-top: 0; color: var(--accent-color);">Defense Companies</h4>
                    <p>Analyze top defense contractors and their performance</p>
                </div>
                <div style="background-color: #F1F5F9; padding: 15px; border-radius: 6px; border-left: 4px solid var(--primary-color);">
                    <h4 style="margin-top: 0; color: var(--primary-color);">Trade Data</h4>
                    <p>Explore military exports and imports worldwide</p>
                </div>
            </div>
            
            <div style="background-color: #F1F5F9; padding: 15px; border-radius: 6px; border-left: 4px solid var(--secondary-color); margin-top: 15px;">
                <h4 style="margin-top: 0; color: var(--secondary-color);">2047 Predictions</h4>
                <p>View projections of future military power rankings</p>
            </div>
            
            <p style="margin-top: 20px; font-weight: 500;">Use the sidebar navigation to explore different aspects of the analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Add key statistics with enhanced styling
        st.markdown("""
        <h3 style="color: var(--primary-color); margin-bottom: 15px;">Key Global Statistics</h3>
        """, unsafe_allow_html=True)
        
        # Calculate some interesting stats and filter out Afghanistan
        total_countries = len(military_strength)
        
        # Filter out Afghanistan for top military power calculation
        filtered_military_strength = military_strength[military_strength['country'] != 'Afghanistan']
        # Get top military power based on power index (lower is better)
        filtered_military_strength = filtered_military_strength.sort_values('pwr_index', ascending=True)
        top_military_power = filtered_military_strength.iloc[0]['country'] if not filtered_military_strength.empty else "N/A"
        
        try:
            # Calculate total global defense budget excluding Afghanistan
            filtered_military_strength = military_strength[military_strength['country'] != 'Afghanistan']
            total_global_defense_budget = sum(pd.to_numeric(filtered_military_strength['national_annual_defense_budgets'], errors='coerce'))
            formatted_budget = f"${total_global_defense_budget/1000000000000:.2f} Trillion"
        except:
            formatted_budget = "Data unavailable"
        
        # Display stats with custom styling
        st.markdown("""
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 15px;">
            <p style="font-size: 0.9rem; margin-bottom: 5px; color: var(--text-color);">Countries Analyzed</p>
            <p style="font-size: 2rem; font-weight: 700; margin: 0; color: var(--primary-color);">{0}</p>
        </div>
        """.format(total_countries), unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 15px;">
            <p style="font-size: 0.9rem; margin-bottom: 5px; color: var(--text-color);">Top Military Power</p>
            <p style="font-size: 2rem; font-weight: 700; margin: 0; color: var(--secondary-color);">{0}</p>
        </div>
        """.format(top_military_power), unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
            <p style="font-size: 0.9rem; margin-bottom: 5px; color: var(--text-color);">Global Defense Spending</p>
            <p style="font-size: 2rem; font-weight: 700; margin: 0; color: var(--accent-color);">{0}</p>
        </div>
        """.format(formatted_budget), unsafe_allow_html=True)
        
        # Add a "Get Started" button with custom styling
        st.markdown("""
        <h3 style="color: var(--primary-color); margin-top: 25px; margin-bottom: 15px;">Begin Your Analysis</h3>
        """, unsafe_allow_html=True)
        
        if st.button("Explore Military Strength Comparison", use_container_width=True):
            st.session_state["page"] = "Military Strength"
            st.rerun()

# World Map page
elif page == "World Map":
    st.header("Interactive Global Military Power Map")
    
    st.write("""
    Hover over countries to see key military statistics. The map displays military power data 
    for countries around the world. Colors indicate relative military strength based on the Power Index.
    """)
    
    # Create a DataFrame with country codes for mapping
    @st.cache_data
    def prepare_map_data(military_df):
        # Convert country names to ISO codes for the map
        try:
            # Create a copy of the dataframe to avoid modifying the original
            map_data = military_df.copy()
            
            # Convert country names to standard ISO codes
            country_names = map_data['country'].tolist()
            iso3_codes = coco.convert(names=country_names, to='ISO3')
            map_data['iso_alpha'] = iso3_codes
            
            # Convert power index to display format (invert so higher values = stronger military)
            # Lower pwr_index actually means stronger military
            map_data['power_score'] = 1 - pd.to_numeric(map_data['pwr_index'], errors='coerce')
            
            # Convert numeric columns to actual numbers
            numeric_cols = [
                'total_national_populations', 
                'active_service_military_manpower',
                'total_military_aircraft_strength',
                'total_combat_tank_strength',
                'navy_strength',
                'national_annual_defense_budgets'
            ]
            
            for col in numeric_cols:
                if col in map_data.columns:
                    map_data[col] = pd.to_numeric(map_data[col], errors='coerce')
            
            # Format values for display
            map_data['formatted_population'] = map_data['total_national_populations'].apply(lambda x: f"{x:,}" if not pd.isna(x) else "N/A")
            map_data['formatted_military'] = map_data['active_service_military_manpower'].apply(lambda x: f"{x:,}" if not pd.isna(x) else "N/A")
            map_data['formatted_aircraft'] = map_data['total_military_aircraft_strength'].apply(lambda x: f"{x:,}" if not pd.isna(x) else "N/A")
            map_data['formatted_tanks'] = map_data['total_combat_tank_strength'].apply(lambda x: f"{x:,}" if not pd.isna(x) else "N/A")
            map_data['formatted_navy'] = map_data['navy_strength'].apply(lambda x: f"{x:,}" if not pd.isna(x) else "N/A")
            map_data['formatted_budget'] = map_data['national_annual_defense_budgets'].apply(lambda x: f"${x:,}" if not pd.isna(x) else "N/A")
            
            return map_data
        except Exception as e:
            st.error(f"Error preparing map data: {e}")
            # Return a simple version if there's an error
            return military_df
    
    # Get map data
    try:
        map_data = prepare_map_data(military_strength)
        
        # Select metric to visualize on the map
        map_metric_options = [
            "Military Power Index",
            "Population",
            "Active Military",
            "Aircraft",
            "Tanks",
            "Naval Vessels",
            "Defense Budget"
        ]
        
        selected_map_metric = st.selectbox("Select a metric to display on the map:", map_metric_options)
        
        # Set the color scale and metric to display
        if selected_map_metric == "Military Power Index":
            color_column = "power_score"
            color_scale = px.colors.sequential.Reds
            hover_data = {
                "country": True,
                "power_score": False,
                "pwr_index": True,
                "formatted_population": True,
                "formatted_military": True,
                "formatted_aircraft": True,
                "formatted_tanks": True,
                "formatted_navy": True,
                "formatted_budget": True,
                "iso_alpha": False
            }
            title = "Global Military Power Index (2024)"
            
        elif selected_map_metric == "Population":
            color_column = "total_national_populations"
            color_scale = px.colors.sequential.Viridis
            hover_data = {
                "country": True,
                "total_national_populations": False,
                "formatted_population": True,
                "formatted_military": True,
                "pwr_index": True,
                "iso_alpha": False
            }
            title = "National Populations"
            
        elif selected_map_metric == "Active Military":
            color_column = "active_service_military_manpower"
            color_scale = px.colors.sequential.Greens
            hover_data = {
                "country": True,
                "active_service_military_manpower": False,
                "formatted_military": True,
                "formatted_population": True,
                "pwr_index": True,
                "iso_alpha": False
            }
            title = "Active Military Personnel"
            
        elif selected_map_metric == "Aircraft":
            color_column = "total_military_aircraft_strength"
            color_scale = px.colors.sequential.Blues
            hover_data = {
                "country": True,
                "total_military_aircraft_strength": False,
                "formatted_aircraft": True,
                "formatted_military": True,
                "pwr_index": True,
                "iso_alpha": False
            }
            title = "Military Aircraft"
            
        elif selected_map_metric == "Tanks":
            color_column = "total_combat_tank_strength"
            color_scale = px.colors.sequential.Oranges
            hover_data = {
                "country": True,
                "total_combat_tank_strength": False,
                "formatted_tanks": True,
                "formatted_military": True,
                "pwr_index": True,
                "iso_alpha": False
            }
            title = "Combat Tanks"
            
        elif selected_map_metric == "Naval Vessels":
            color_column = "navy_strength"
            color_scale = px.colors.sequential.Purples
            hover_data = {
                "country": True,
                "navy_strength": False,
                "formatted_navy": True,
                "formatted_military": True,
                "pwr_index": True,
                "iso_alpha": False
            }
            title = "Naval Vessels"
            
        else:  # Defense Budget
            color_column = "national_annual_defense_budgets"
            color_scale = px.colors.sequential.Plasma
            hover_data = {
                "country": True,
                "national_annual_defense_budgets": False,
                "formatted_budget": True,
                "formatted_military": True,
                "pwr_index": True,
                "iso_alpha": False
            }
            title = "Defense Budget"
        
        # Create the choropleth map
        fig = px.choropleth(
            map_data,
            locations="iso_alpha",
            color=color_column,
            hover_name="country",
            hover_data=hover_data,
            color_continuous_scale=color_scale,
            labels={
                "formatted_population": "Population",
                "formatted_military": "Military Personnel",
                "formatted_aircraft": "Aircraft",
                "formatted_tanks": "Tanks",
                "formatted_navy": "Navy Vessels",
                "formatted_budget": "Defense Budget",
                "pwr_index": "Power Index (lower is better)"
            },
            title=title
        )
        
        # Update the layout for better appearance
        fig.update_layout(
            height=700,
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type="natural earth"
            ),
            coloraxis_colorbar=dict(
                title=selected_map_metric
            )
        )
        
        # Display the map
        st.plotly_chart(fig, use_container_width=True)
        
        # Add map insights
        st.subheader("Map Insights")
        st.markdown("""
        The interactive world map displays military power distribution across countries.
        
        **Key observations:**
        - Darker colors indicate stronger military capabilities based on the selected metric
        - The Power Index (PWR) is a comprehensive score where lower values indicate stronger overall military power
        - Hover over any country to see detailed statistics
        - Use the dropdown above to visualize different military metrics
        
        This visualization helps identify military power concentration regions and compare relative strengths.
        """)
        
    except Exception as e:
        st.error(f"Error creating the map: {e}")
        st.info("Please ensure you have the country-converter package installed: `pip install country-converter`")

# Military Strength page
elif page == "Military Strength":
    st.header("Military Strength Comparison (2024)")
    
    # Replace data overview with country selection for comparison
    st.subheader("Compare Countries")
    
    # Get list of all countries
    countries = sorted(military_strength['country'].unique().tolist())
    
    # Allow user to select number of countries to compare
    num_countries = st.slider("Number of countries to compare:", min_value=2, max_value=5, value=2)
    
    # Create multiselect for countries
    default_countries = []
    if 'United States' in countries:
        default_countries.append('United States')
    if 'China' in countries and len(default_countries) < num_countries:
        default_countries.append('China')
    if 'Russia' in countries and len(default_countries) < num_countries:
        default_countries.append('Russia')
    if 'India' in countries and len(default_countries) < num_countries:
        default_countries.append('India')
    
    # Fill with other countries if needed
    while len(default_countries) < num_countries and len(default_countries) < len(countries):
        for country in countries:
            if country not in default_countries:
                default_countries.append(country)
                break
                
    selected_countries = st.multiselect(
        "Select countries to compare:",
        options=countries,
        default=default_countries[:num_countries]
    )
    
    # Check if we have the right number of countries selected
    if len(selected_countries) < 2:
        st.warning("Please select at least 2 countries for comparison.")
    elif len(selected_countries) > 5:
        st.warning("Please select at most 5 countries for comparison.")
    else:
        # Get data for selected countries
        countries_data = {}
        for country in selected_countries:
            country_data = military_strength[military_strength['country'] == country]
            if not country_data.empty:
                countries_data[country] = country_data.iloc[0]
        
        # Display basic country info
        st.subheader("Basic Information: Multi-Country Comparison")
        
        # Create comparison metrics
        comparison_metrics = {
            'Population': 'total_national_populations',
            'Active Military': 'active_service_military_manpower',
            'Reserve Forces': 'active_service_reserve_components',
            'Defense Budget': 'national_annual_defense_budgets',
            'Fighter Aircraft': 'total_fighter/interceptor_aircraft_strength',
            'Attack Aircraft': 'total_attack_aircraft_strength',
            'Helicopters': 'total_helicopter_strength',
            'Attack Helicopters': 'total_attack_helicopter_strength',
            'Tanks': 'total_combat_tank_strength',
            'Armored Vehicles': 'total_armored_fighting_vehicle_strength',
            'Artillery (Self-Propelled)': 'total_self_propelled_artillery_strength',
            'Artillery (Towed)': 'total_towed_artillery_strength',
            'Navy Ships': 'navy_strength',
            'Aircraft Carriers': 'aircraft_carrier_strength',
            'Submarines': 'navy_submarine_strength',
            'Power Index (lower is better)': 'pwr_index'
        }
        
        # Create comparison table
        comparison_data = []
        for label, metric in comparison_metrics.items():
            metric_data = {'Metric': label}
            valid_data = True
            
            for country in selected_countries:
                if country in countries_data and metric in countries_data[country].index:
                    # Get value and format if needed
                    val = countries_data[country][metric]
                    
                    # Convert to numeric if needed
                    if isinstance(val, str):
                        try:
                            numeric_val = float(val.replace(',', ''))
                            # Store as string to avoid Arrow conversion issues
                            metric_data[country] = "{:,}".format(int(numeric_val)) if numeric_val >= 1 else str(numeric_val)
                        except:
                            metric_data[country] = val
                    else:
                        # Convert numeric values to strings
                        metric_data[country] = "{:,}".format(int(val)) if isinstance(val, (int, float)) and val >= 1 else str(val)
                else:
                    metric_data[country] = "N/A"
            
            comparison_data.append(metric_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        # Convert to strings to avoid Arrow conversion errors
        comparison_df = make_arrow_compatible(comparison_df)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Select category for detailed comparison
        category_options = [
            'Air Power',
            'Land Forces', 
            'Naval Power',
            'Economic Factors'
        ]
        
        selected_category = st.selectbox("Select Category for Detailed Comparison:", category_options)
        
        # Define metrics for each category
        category_metrics = {
            'Air Power': [
                'total_military_aircraft_strength',
                'total_fighter/interceptor_aircraft_strength',
                'total_attack_aircraft_strength',
                'total_military_transport_aircraft_strength',
                'total_military_trainer_aircraft_strength',
                'special_mission_aircraft_fleets',
                'aerial_tanker_aircraft_fleet_strength',
                'total_helicopter_strength',
                'total_attack_helicopter_strength'
            ],
            'Land Forces': [
                'active_service_military_manpower',
                'active_service_reserve_components',
                'active_paramilitary_force_strength',
                'total_combat_tank_strength',
                'total_armored_fighting_vehicle_strength',
                'total_self_propelled_artillery_strength',
                'total_towed_artillery_strength',
                'total_rocket_launcher_vehicle_strength'
            ],
            'Naval Power': [
                'navy_strength',
                'aircraft_carrier_strength',
                'helicopter_carrier_strength',
                'navy_submarine_strength',
                'destroyer_warship_strength',
                'navy_frigate_warship_strength',
                'navy_corvette_warship_strength',
                'navy_patrol_craft_strength',
                'navy_mine_warfare_craft_strength'
            ],
            'Economic Factors': [
                'national_annual_defense_budgets',
                'national_external_debts',
                'purchasing_power_parities',
                'national_reserves_of_foreign_exchange_and_gold',
                'total_labor_force_strength',
                'oil_production_figures',
                'oil_consumption_figures'
            ]
        }
        
        # Get metrics for selected category
        selected_metrics = category_metrics[selected_category]
        
        # Create a dataframe for the metrics in selected category
        category_data = []
        metric_labels = {
            'total_military_aircraft_strength': 'Total Aircraft',
            'total_fighter/interceptor_aircraft_strength': 'Fighter Aircraft',
            'total_attack_aircraft_strength': 'Attack Aircraft',
            'total_military_transport_aircraft_strength': 'Transport Aircraft',
            'total_military_trainer_aircraft_strength': 'Trainer Aircraft',
            'special_mission_aircraft_fleets': 'Special Mission Aircraft',
            'aerial_tanker_aircraft_fleet_strength': 'Aerial Tankers',
            'total_helicopter_strength': 'Helicopters',
            'total_attack_helicopter_strength': 'Attack Helicopters',
            'active_service_military_manpower': 'Active Military',
            'active_service_reserve_components': 'Reserves',
            'active_paramilitary_force_strength': 'Paramilitary',
            'total_combat_tank_strength': 'Tanks',
            'total_armored_fighting_vehicle_strength': 'Armored Vehicles',
            'total_self_propelled_artillery_strength': 'Self-Propelled Artillery',
            'total_towed_artillery_strength': 'Towed Artillery',
            'total_rocket_launcher_vehicle_strength': 'Rocket Launchers',
            'navy_strength': 'Navy (Total)',
            'aircraft_carrier_strength': 'Aircraft Carriers',
            'helicopter_carrier_strength': 'Helicopter Carriers',
            'navy_submarine_strength': 'Submarines',
            'destroyer_warship_strength': 'Destroyers',
            'navy_frigate_warship_strength': 'Frigates',
            'navy_corvette_warship_strength': 'Corvettes',
            'navy_patrol_craft_strength': 'Patrol Craft',
            'navy_mine_warfare_craft_strength': 'Mine Warfare Craft',
            'national_annual_defense_budgets': 'Defense Budget ($)',
            'national_external_debts': 'External Debt ($)',
            'purchasing_power_parities': 'PPP ($)',
            'national_reserves_of_foreign_exchange_and_gold': 'Reserves ($)',
            'total_labor_force_strength': 'Labor Force',
            'oil_production_figures': 'Oil Production',
            'oil_consumption_figures': 'Oil Consumption'
        }
        
        for metric in selected_metrics:
            metric_data = {'Metric': metric_labels.get(metric, metric.replace('_', ' ').title())}
            raw_numeric_data = {}
            
            for country in selected_countries:
                if country in countries_data and metric in countries_data[country].index:
                    # Convert to numeric
                    val = countries_data[country][metric]
                    
                    if isinstance(val, str):
                        try:
                            numeric_val = float(val.replace(',', ''))
                            raw_numeric_data[country] = numeric_val
                            # Store formatted string in display data
                            metric_data[country] = "{:,}".format(int(numeric_val)) if numeric_val >= 1 else str(numeric_val)
                        except:
                            raw_numeric_data[country] = 0
                            metric_data[country] = val
                    else:
                        raw_numeric_data[country] = float(val) if val is not None else 0
                        metric_data[country] = "{:,}".format(int(val)) if isinstance(val, (int, float)) and val >= 1 else str(val)
                else:
                    raw_numeric_data[country] = 0
                    metric_data[country] = "N/A"
            
            # Add both display data and raw numeric data for visualization
            metric_data['_raw_data'] = raw_numeric_data
            category_data.append(metric_data)
        
        # Create visualization for the comparison
        if category_data:
            # Create a copy without the raw data for display
            display_df = pd.DataFrame([{k: v for k, v in item.items() if k != '_raw_data'} for item in category_data])
            
            # Convert all columns to strings to avoid Arrow conversion issues
            display_df = make_arrow_compatible(display_df)
            
            # Create a dataframe with numeric values for the chart
            chart_data = []
            for item in category_data:
                metric = item['Metric']
                raw_data = item['_raw_data']
                for country, value in raw_data.items():
                    chart_data.append({
                        'Metric': metric,
                        'Country': country,
                        'Value': value
                    })

            chart_df = pd.DataFrame(chart_data)

            # Create the comparison chart
            st.subheader(f"Detailed Comparison: {selected_category}")

            # Create the comparison chart
            fig, ax = plt.subplots(figsize=(12, 8))

            # Use a color palette based on number of countries
            if len(selected_countries) <= 3:
                palette = "Set1"
            else:
                palette = "Set2"
                
            chart = sns.barplot(x='Metric', y='Value', hue='Country', data=chart_df, ax=ax, palette=palette)
            plt.xticks(rotation=45, ha='right')
            plt.yscale('log')  # Log scale for better comparison of different magnitudes
            plt.tight_layout()
            plt.legend(title='Country')
            st.pyplot(fig)

            # Show the actual values in a table
            st.subheader(f"{selected_category} Data")
            st.dataframe(display_df, use_container_width=True)
        else:
            st.write("No data available for this category.")
        
        # Add radar chart for overall comparison
        st.subheader("Overall Military Capability Comparison")
        
        # Select key metrics for radar chart
        radar_metrics = [
            'active_service_military_manpower',
            'total_military_aircraft_strength',
            'total_combat_tank_strength', 
            'navy_strength',
            'national_annual_defense_budgets'
        ]
        
        radar_labels = [
            'Military Personnel',
            'Aircraft',
            'Tanks',
            'Naval Vessels',
            'Defense Budget'
        ]
        
        # Prepare data for radar chart
        radar_data = []
        for metric, label in zip(radar_metrics, radar_labels):
            radar_row = {'Metric': label}
            raw_values = {}
            
            # Get values for all countries
            for country in selected_countries:
                if country in countries_data and metric in countries_data[country].index:
                    val = countries_data[country][metric]
                    
                    if isinstance(val, str):
                        try:
                            numeric_val = float(val.replace(',', ''))
                            raw_values[country] = numeric_val
                        except:
                            raw_values[country] = 0
                    else:
                        raw_values[country] = float(val) if val is not None else 0
                else:
                    raw_values[country] = 0
            
            # Find max value for normalization
            max_val = max(raw_values.values()) if raw_values else 1
            
            # Normalize values and add to row
            if max_val > 0:
                for country in selected_countries:
                    radar_row[country] = raw_values[country] / max_val
                    radar_row[f'{country} (Raw)'] = raw_values[country]
            
            radar_data.append(radar_row)
        
        # Create radar chart
        if radar_data and len(radar_data) >= 3:  # Need at least 3 metrics for radar chart
            # Create the radar chart
            categories = [item['Metric'] for item in radar_data]
            N = len(categories)
            
            # Create angles for the radar chart
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            # Set color palette
            if len(selected_countries) <= 3:
                colors = plt.cm.Set1(np.linspace(0, 1, len(selected_countries)))
            else:
                colors = plt.cm.Set2(np.linspace(0, 1, len(selected_countries)))
            
            # Draw the chart for each country
            for i, country in enumerate(selected_countries):
                values = [item[country] for item in radar_data]
                values += values[:1]  # Close the loop
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=country, color=colors[i])
                ax.fill(angles, values, alpha=0.1, color=colors[i])
            
            # Set labels
            plt.xticks(angles[:-1], categories)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # Show the plot
            st.pyplot(fig)
            
            # Show the raw data for radar chart
            st.subheader("Raw Data for Key Metrics")
            
            # Create display dataframe with formatted values
            display_data = []
            for item in radar_data:
                row_data = {'Metric': item['Metric']}
                for country in selected_countries:
                    raw_value = item[f'{country} (Raw)']
                    # Format as string to avoid Arrow conversion errors
                    row_data[country] = "{:,}".format(int(raw_value)) if raw_value >= 1 else str(raw_value)
                display_data.append(row_data)
            
            display_df = pd.DataFrame(display_data)
            
            # Convert to arrow-compatible format
            display_df = make_arrow_compatible(display_df)
            
            st.dataframe(display_df, use_container_width=True)
        elif len(radar_data) < 3:
            st.info("Radar chart requires at least 3 metrics with valid data for all selected countries.")

# Defense Budget page
elif page == "Defense Budget":
    st.header("Defense Budget Data")
    
    # Add option for single country analysis or comparative analysis
    analysis_type = st.radio("Select Analysis Type:", ["Single Country Analysis", "Comparative Analysis"])
    
    if analysis_type == "Single Country Analysis":
        # Show data summary
        st.subheader("Data Overview")
        st.dataframe(defense_budget.head(10))
        
        # Select country for time series visualization
        countries = sorted(defense_budget['Country Name'].unique().tolist())
        selected_country = st.selectbox("Select a country:", countries)
        
        # Filter data for the selected country
        country_data = defense_budget[defense_budget['Country Name'] == selected_country]
        
        # Prepare data for time series visualization
        if not country_data.empty:
            # Melt the dataframe to get years as a column
            year_columns = [col for col in country_data.columns if col.isdigit()]
            melted_data = pd.melt(country_data, 
                                id_vars=['Country Name'], 
                                value_vars=year_columns,
                                var_name='Year', 
                                value_name='Military Expenditure (% of GDP)')
            
            # Create time series plot
            st.subheader(f"Military Expenditure Trend for {selected_country}")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(x='Year', y='Military Expenditure (% of GDP)', data=melted_data, ax=ax)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.title(f"Military Expenditure as % of GDP for {selected_country} (1960-2020)")
            st.pyplot(fig)
    
    else:  # Comparative Analysis
        st.subheader("Compare Defense Budgets Between Countries")
        
        # Get list of all countries
        countries = sorted(defense_budget['Country Name'].unique().tolist())
        
        # Allow user to select number of countries to compare
        num_countries = st.slider("Number of countries to compare:", min_value=2, max_value=10, value=3)
        
        # Create multiselect for countries
        default_countries = []
        if 'United States' in countries:
            default_countries.append('United States')
        if 'China' in countries and len(default_countries) < num_countries:
            default_countries.append('China')
        if 'Russia' in countries and len(default_countries) < num_countries:
            default_countries.append('Russia')
        if 'India' in countries and len(default_countries) < num_countries:
            default_countries.append('India')
        
        # Fill with other countries if needed
        while len(default_countries) < num_countries and len(default_countries) < len(countries):
            for country in countries:
                if country not in default_countries:
                    default_countries.append(country)
                    break
                    
        selected_countries = st.multiselect(
            "Select countries to compare:",
            options=countries,
            default=default_countries[:num_countries]
        )
        
        # Check if we have the right number of countries selected
        if len(selected_countries) < 2:
            st.warning("Please select at least 2 countries for comparison.")
        elif len(selected_countries) > 10:
            st.warning("Please select at most 10 countries for comparison.")
        else:
            # Get data for selected countries
            filtered_data = defense_budget[defense_budget['Country Name'].isin(selected_countries)]
            
            if not filtered_data.empty:
                # Get year columns
                year_columns = [col for col in defense_budget.columns if col.isdigit()]
                
                # Prepare data for comparison
                melted_data = pd.melt(filtered_data,
                                    id_vars=['Country Name'],
                                    value_vars=year_columns,
                                    var_name='Year',
                                    value_name='Military Expenditure (% of GDP)')
                
                # Create time series comparison plot
                st.subheader(f"Military Expenditure Comparison ({len(selected_countries)} Countries)")
                
                # Use a color palette based on number of countries
                if len(selected_countries) <= 5:
                    palette = "Set1"
                elif len(selected_countries) <= 8:
                    palette = "Set2"
                else:
                    palette = "Set3"
                    
                fig, ax = plt.subplots(figsize=(14, 7))
                sns.lineplot(x='Year', y='Military Expenditure (% of GDP)', 
                           hue='Country Name', data=melted_data, 
                           ax=ax, palette=palette)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.legend(title="Country", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.title(f"Military Expenditure as % of GDP: Comparison (1960-2020)")
                st.pyplot(fig)
                
                # Calculate statistics for comparison
                recent_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
                recent_years = [year for year in recent_years if year in year_columns]
                
                if recent_years:
                    st.subheader("Recent Years Comparison")
                    
                    # Create a comparison dataframe for recent years
                    comparison_data = []
                    
                    # For each year, get values for all countries
                    for year in recent_years:
                        year_data = {'Year': year}
                        
                        for country in selected_countries:
                            country_year_data = filtered_data[filtered_data['Country Name'] == country]
                            if not country_year_data.empty and year in country_year_data.columns:
                                val = country_year_data[year].values[0] if not pd.isna(country_year_data[year].values[0]) else 0
                                year_data[country] = val
                            else:
                                year_data[country] = 0
                                
                        comparison_data.append(year_data)
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Calculate recent averages for each country
                    averages = {country: comparison_df[country].mean() for country in selected_countries}
                    
                    # Display comparison metrics
                    st.subheader("Average Military Expenditure (% of GDP) in Recent Years")
                    
                    # Create columns based on number of countries
                    metrics_cols = st.columns(min(len(selected_countries), 5))
                    
                    # First row of metrics
                    for i, country in enumerate(list(averages.keys())[:5]):
                        with metrics_cols[i % 5]:
                            st.metric(f"{country}", f"{averages[country]:.2f}% of GDP")
                    
                    # Second row of metrics if needed
                    if len(selected_countries) > 5:
                        metrics_cols2 = st.columns(min(len(selected_countries) - 5, 5))
                        for i, country in enumerate(list(averages.keys())[5:10]):
                            with metrics_cols2[i % 5]:
                                st.metric(f"{country}", f"{averages[country]:.2f}% of GDP")
                    
                    # Display the comparison table
                    st.subheader("Year-by-Year Comparison")
                    
                    # Make dataframe Arrow-compatible
                    display_df = make_arrow_compatible(comparison_df)
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Create bar chart for comparing recent averages
                    avg_data = pd.DataFrame({
                        'Country': list(averages.keys()),
                        'Average Military Expenditure (% of GDP)': list(averages.values())
                    })
                    
                    avg_data = avg_data.sort_values('Average Military Expenditure (% of GDP)', ascending=False)
                    
                    st.subheader("Average Military Expenditure in Recent Years")
                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    bars = sns.barplot(
                        x='Country', 
                        y='Average Military Expenditure (% of GDP)', 
                        data=avg_data, 
                        hue='Country',
                        palette=palette,
                        legend=False,
                        ax=ax2
                    )
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    # Add value labels on top of bars
                    for i, bar in enumerate(bars.patches):
                        bars.text(bar.get_x() + bar.get_width()/2., 
                                bar.get_height() + 0.1,
                                f'{avg_data["Average Military Expenditure (% of GDP)"].iloc[i]:.2f}%',
                                ha='center', va='bottom')
                    
                    st.pyplot(fig2)
                    
                    # Create heatmap for more detailed comparison if there are enough countries
                    if len(selected_countries) >= 3:
                        st.subheader("Year-by-Year Comparison Heatmap")
                        
                        # Create a version of comparison_df with proper numeric values for heatmap
                        heatmap_data = comparison_df.copy()
                        
                        # Create a numeric dataframe for the heatmap
                        heatmap_values = []
                        for _, row in heatmap_data.iterrows():
                            year = row['Year']
                            for country in selected_countries:
                                value = row[country]
                                heatmap_values.append({
                                    'Year': year,
                                    'Country': country,
                                    'Value': value
                                })
                        
                        # Create the heatmap dataframe
                        heatmap_df = pd.DataFrame(heatmap_values)
                        
                        # Create pivot table for heatmap
                        pivot_data = heatmap_df.pivot(index='Country', columns='Year', values='Value')
                        
                        # Create heatmap
                        fig3, ax3 = plt.subplots(figsize=(14, len(selected_countries) * 0.8))
                        
                        # Generate annotation text with formatted values
                        annotations = pivot_data.map(lambda x: f'{x:.2f}')
                        
                        # Create the heatmap
                        heatmap = sns.heatmap(pivot_data, annot=annotations, fmt='', cmap='YlGnBu', 
                                        linewidths=.5, ax=ax3)
                        plt.title('Military Expenditure by Country and Year (% of GDP)')
                        plt.tight_layout()
                        st.pyplot(fig3)

# Defense Companies page
elif page == "Defense Companies":
    st.header("Major Defense Companies (2005-2022)")

    # Add option for single company analysis or comparative analysis
    analysis_type = st.radio("Select Analysis Type:", ["Single Company Analysis", "Comparative Analysis"])
    
    if analysis_type == "Single Company Analysis":
        # Show data summary
        st.subheader("Data Overview")
        st.dataframe(make_arrow_compatible(defense_companies.head(10)))
        
        # Select company for visualization
        companies = sorted(defense_companies['Company'].unique().tolist())
        selected_company = st.selectbox("Select a company:", companies)
        
        # Filter data for the selected company
        company_data = defense_companies[defense_companies['Company'] == selected_company]
        
        # Display company info
        if not company_data.empty:
            st.subheader(f"Information for {selected_company}")
            
            # Get the country of the company
            company_country = company_data['Country'].iloc[0]
            st.write(f"**Country:** {company_country}")
            
            # Prepare data for time series visualization
            year_cols = [col for col in company_data.columns if col.startswith('20')]
            
            # Create time series plot for revenue
            st.subheader(f"Revenue Trend for {selected_company}")
            
            # Convert year columns to proper format
            revenue_data = []
            for year in year_cols:
                revenue = company_data[year].iloc[0] if year in company_data.columns else None
                if pd.notna(revenue):
                    revenue_data.append({
                        'Year': int(year),
                        'Revenue (USD millions)': revenue
                    })
            
            if revenue_data:
                revenue_df = pd.DataFrame(revenue_data)
                
                # Create line plot
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.lineplot(x='Year', y='Revenue (USD millions)', data=revenue_df, marker='o', linewidth=2, ax=ax)
                plt.grid(True, alpha=0.3)
                plt.title(f"Annual Revenue for {selected_company} (USD millions)")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Calculate growth rates
                if len(revenue_data) > 1:
                    st.subheader("Revenue Growth Analysis")
                    
                    # Calculate year-over-year growth
                    revenue_df['Previous Year Revenue'] = revenue_df['Revenue (USD millions)'].shift(1)
                    revenue_df['YoY Growth (%)'] = (revenue_df['Revenue (USD millions)'] - revenue_df['Previous Year Revenue']) / revenue_df['Previous Year Revenue'] * 100
                    revenue_df = revenue_df.dropna()
                    
                    # Display growth stats
                    avg_growth = revenue_df['YoY Growth (%)'].mean()
                    max_growth = revenue_df['YoY Growth (%)'].max()
                    min_growth = revenue_df['YoY Growth (%)'].min()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Average Annual Growth", f"{avg_growth:.2f}%")
                    col2.metric("Highest Annual Growth", f"{max_growth:.2f}%")
                    col3.metric("Lowest Annual Growth", f"{min_growth:.2f}%")
                    
                    # Create growth rate chart
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bars = sns.barplot(x='Year', y='YoY Growth (%)', data=revenue_df, ax=ax)
                    
                    # Add value labels on top of bars
                    for i, bar in enumerate(bars.patches):
                        ax.text(bar.get_x() + bar.get_width()/2., 
                                bar.get_height() + 1 if bar.get_height() >= 0 else bar.get_height() - 5,
                                f"{revenue_df['YoY Growth (%)'].iloc[i]:.1f}%",
                                ha='center', va='bottom' if bar.get_height() >= 0 else 'top')
                    
                    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    plt.grid(True, alpha=0.3)
                    plt.title(f"Year-over-Year Revenue Growth for {selected_company}")
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.write("No revenue data available for this company.")
    else:  # Comparative Analysis
        st.subheader("Compare Defense Companies")
        
        # Get list of all companies
        companies = sorted(defense_companies['Company'].unique().tolist())
        
        # Allow user to select number of companies to compare
        num_companies = st.slider("Number of companies to compare:", min_value=2, max_value=5, value=3)
        
        # Select default companies (largest ones for better comparison)
        default_companies = []
        
        # Try to find some major defense companies as defaults
        major_companies = [
            'Lockheed Martin', 'Boeing', 'Raytheon Company', 'BAE Systems', 'Northrop Grumman',
            'General Dynamics', 'Airbus', 'L3Harris Technologies', 'Leonardo', 'Thales'
        ]
        
        for company in major_companies:
            if company in companies and len(default_companies) < num_companies:
                default_companies.append(company)
        
        # Fill with other companies if needed
        while len(default_companies) < num_companies and len(default_companies) < len(companies):
            for company in companies:
                if company not in default_companies:
                    default_companies.append(company)
                    break
        
        # Create multiselect for companies
        selected_companies = st.multiselect(
            "Select companies to compare:",
            options=companies,
            default=default_companies[:num_companies]
        )
        
        # Check if we have the right number of companies selected
        if len(selected_companies) < 2:
            st.warning("Please select at least 2 companies for comparison.")
        elif len(selected_companies) > 5:
            st.warning("Please select at most 5 companies for comparison.")
        else:
            # Filter data for selected companies
            filtered_data = defense_companies[defense_companies['Company'].isin(selected_companies)]
            
            if not filtered_data.empty:
                st.subheader("Company Nationalities")
                
                # Display company countries
                country_data = {}
                for company in selected_companies:
                    company_info = filtered_data[filtered_data['Company'] == company]
                    if not company_info.empty:
                        country = company_info['Country'].iloc[0]
                        country_data[company] = country
                
                # Display as a table
                country_df = pd.DataFrame({
                    'Company': list(country_data.keys()),
                    'Country': list(country_data.values())
                })
                st.dataframe(make_arrow_compatible(country_df), use_container_width=True)
                
                # Get year columns (2005-2022)
                year_cols = [col for col in filtered_data.columns if col.startswith('20')]
                
                # Create revenue comparison data
                revenue_data = []
                for company in selected_companies:
                    company_info = filtered_data[filtered_data['Company'] == company]
                    if not company_info.empty:
                        for year in year_cols:
                            revenue = company_info[year].iloc[0] if year in company_info.columns else None
                            if pd.notna(revenue):
                                revenue_data.append({
                                    'Company': company,
                                    'Year': int(year),
                                    'Revenue (USD millions)': revenue
                                })
                
                if revenue_data:
                    revenue_df = pd.DataFrame(revenue_data)
                    
                    # Create comparison line chart
                    st.subheader("Revenue Comparison")
                    
                    # Use a color palette based on number of companies
                    if len(selected_companies) <= 3:
                        palette = "Set1"
                    else:
                        palette = "Set2"
                    
                    fig, ax = plt.subplots(figsize=(14, 7))
                    sns.lineplot(
                        x='Year', 
                        y='Revenue (USD millions)', 
                        hue='Company', 
                        data=revenue_df, 
                        marker='o',
                        palette=palette,
                        ax=ax
                    )
                    plt.grid(True, alpha=0.3)
                    plt.title("Annual Revenue Comparison (USD millions)")
                    plt.legend(title="Company", bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Calculate latest available year for each company
                    latest_revenues = {}
                    for company in selected_companies:
                        company_revenue = revenue_df[revenue_df['Company'] == company]
                        if not company_revenue.empty:
                            latest_year = company_revenue['Year'].max()
                            latest_revenue = company_revenue[company_revenue['Year'] == latest_year]['Revenue (USD millions)'].iloc[0]
                            latest_revenues[company] = {
                                'Year': latest_year,
                                'Revenue': latest_revenue
                            }
                    
                    # Display latest revenue comparison
                    st.subheader("Latest Available Revenue")
                    
                    # Create columns for metrics
                    metrics_cols = st.columns(len(selected_companies))
                    
                    # Display metrics
                    for i, (company, data) in enumerate(latest_revenues.items()):
                        with metrics_cols[i]:
                            st.metric(
                                f"{company}",
                                f"${data['Revenue']:,.0f}M",
                                f"({data['Year']})"
                            )
                    
                    # Calculate average annual growth rate for each company
                    st.subheader("Growth Rate Comparison")
                    
                    growth_data = []
                    for company in selected_companies:
                        company_revenue = revenue_df[revenue_df['Company'] == company].sort_values('Year')
                        if len(company_revenue) > 1:
                            company_revenue['Previous Year Revenue'] = company_revenue['Revenue (USD millions)'].shift(1)
                            company_revenue['YoY Growth (%)'] = (company_revenue['Revenue (USD millions)'] - company_revenue['Previous Year Revenue']) / company_revenue['Previous Year Revenue'] * 100
                            company_revenue = company_revenue.dropna()
                            
                            avg_growth = company_revenue['YoY Growth (%)'].mean()
                            
                            growth_data.append({
                                'Company': company,
                                'Average Annual Growth (%)': avg_growth
                            })
                    
                    if growth_data:
                        growth_df = pd.DataFrame(growth_data)
                        
                        # Create bar chart for growth comparison
                        fig, ax = plt.subplots(figsize=(12, 6))
                        bars = sns.barplot(
                            x='Company', 
                            y='Average Annual Growth (%)',
                            data=growth_df,
                            palette=palette,
                            ax=ax
                        )
                        
                        # Add value labels on top of bars
                        for i, bar in enumerate(bars.patches):
                            ax.text(
                                bar.get_x() + bar.get_width()/2.,
                                bar.get_height() + 0.5 if bar.get_height() >= 0 else bar.get_height() - 2,
                                f"{growth_df['Average Annual Growth (%)'].iloc[i]:.1f}%",
                                ha='center', 
                                va='bottom' if bar.get_height() >= 0 else 'top'
                            )
                        
                        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                        plt.grid(True, alpha=0.3)
                        plt.title("Average Annual Revenue Growth Rate")
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Create heat map for year-by-year comparison if there are enough companies
                        if len(selected_companies) >= 2:
                            st.subheader("Year-by-Year Revenue Comparison")
                            
                            # Create pivot table for the heatmap
                            pivot_data = revenue_df.pivot(index='Company', columns='Year', values='Revenue (USD millions)')
                            
                            # Create heatmap
                            fig, ax = plt.subplots(figsize=(14, len(selected_companies) * 0.8))
                            
                            # Format values for display
                            annotations = pivot_data.map(lambda x: f'${x:,.0f}M' if pd.notna(x) else '')
                            
                            # Create the heatmap
                            heatmap = sns.heatmap(
                                pivot_data, 
                                annot=annotations, 
                                fmt='', 
                                cmap='YlGnBu',
                                linewidths=.5, 
                                ax=ax
                            )
                            plt.title('Annual Revenue by Company (USD millions)')
                            plt.tight_layout()
                            st.pyplot(fig)
                else:
                    st.write("No revenue data available for the selected companies.")

# Trade Data page
elif page == "Trade Data":
    st.header("Military Trade Data")
    
    # Add option for single country analysis or comparative analysis
    analysis_type = st.radio("Select Analysis Type:", ["Single Country Analysis", "Comparative Analysis"])
    
    if analysis_type == "Single Country Analysis":
        # Show data summary
        st.subheader("Data Overview")
        st.dataframe(make_arrow_compatible(exports_imports.head(10)))
        
        # Select country for analysis
        countries = sorted(exports_imports['country'].unique().tolist())
        selected_country = st.selectbox("Select a country:", countries)
        
        # Filter data for the selected country
        country_data = exports_imports[exports_imports['country'] == selected_country]
        
        if not country_data.empty:
            st.subheader(f"Military Trade Analysis for {selected_country}")
            
            # Get time range for which we have data
            years = sorted(country_data['financial_year(end)'].unique().tolist())
            
            # Create time series data
            trade_data = []
            for _, row in country_data.iterrows():
                year = row['financial_year(end)']
                exports = row['export'] if pd.notna(row['export']) else 0
                imports = row['import'] if pd.notna(row['import']) else 0
                
                trade_data.append({
                    'Year': year,
                    'Exports (USD millions)': exports,
                    'Imports (USD millions)': imports,
                    'Trade Balance (USD millions)': exports - imports
                })
            
            trade_df = pd.DataFrame(trade_data)
            
            # Create multiline chart for exports and imports
            st.subheader(f"Military Exports and Imports Trend for {selected_country}")
            
            # Create figure with dual y-axis
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot exports
            ax1.plot(trade_df['Year'], trade_df['Exports (USD millions)'], 'b-', marker='o', linewidth=2, label='Exports')
            ax1.plot(trade_df['Year'], trade_df['Imports (USD millions)'], 'r-', marker='s', linewidth=2, label='Imports')
            
            # Set labels and title
            ax1.set_xlabel('Year')
            ax1.set_ylabel('USD Millions')
            ax1.tick_params(axis='y')
            ax1.grid(True, alpha=0.3)
            plt.xticks(trade_df['Year'], rotation=45)
            plt.title(f"Military Exports and Imports for {selected_country}")
            plt.legend(loc='upper left')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display trade balance chart
            st.subheader(f"Trade Balance for {selected_country}")
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            bars = sns.barplot(x='Year', y='Trade Balance (USD millions)', data=trade_df, ax=ax2)
            
            # Add value labels
            for i, bar in enumerate(bars.patches):
                ax2.text(
                    bar.get_x() + bar.get_width()/2.,
                    bar.get_height() + 0.5 if bar.get_height() >= 0 else bar.get_height() - 50,
                    f"${trade_df['Trade Balance (USD millions)'].iloc[i]:.1f}M",
                    ha='center',
                    va='bottom' if bar.get_height() >= 0 else 'top'
                )
            
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.title(f"Military Trade Balance for {selected_country}")
            plt.tight_layout()
            st.pyplot(fig2)
            
            # Display stats
            total_exports = trade_df['Exports (USD millions)'].sum()
            total_imports = trade_df['Imports (USD millions)'].sum()
            overall_balance = total_exports - total_imports
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Exports", f"${total_exports:,.0f}M")
            col2.metric("Total Imports", f"${total_imports:,.0f}M")
            col3.metric("Overall Balance", f"${overall_balance:,.0f}M", 
                      delta="Surplus" if overall_balance > 0 else "Deficit")
            
            # Display recent data table
            st.subheader("Year-by-Year Data")
            st.dataframe(make_arrow_compatible(trade_df), use_container_width=True)
    else:  # Comparative Analysis
        st.subheader("Compare Military Trade Data Between Countries")
        
        # Get list of all countries
        countries = sorted(exports_imports['country'].unique().tolist())
        
        # Allow user to select number of countries to compare
        num_countries = st.slider("Number of countries to compare:", min_value=2, max_value=5, value=2)
        
        # Create multiselect for countries with default values for major countries
        default_countries = []
        major_countries = ['United States', 'Russia', 'China', 'France', 'Germany', 'United Kingdom', 'Israel', 'Italy']
        
        for country in major_countries:
            if country in countries and len(default_countries) < num_countries:
                default_countries.append(country)
        
        # Fill with other countries if needed
        while len(default_countries) < num_countries and len(default_countries) < len(countries):
            for country in countries:
                if country not in default_countries:
                    default_countries.append(country)
                    break
        
        # Create multiselect for countries
        selected_countries = st.multiselect(
            "Select countries to compare:",
            options=countries,
            default=default_countries[:num_countries]
        )
        
        # Check if we have the right number of countries selected
        if len(selected_countries) < 2:
            st.warning("Please select at least 2 countries for comparison.")
        elif len(selected_countries) > 5:
            st.warning("Please select at most 5 countries for comparison.")
        else:
            # Prepare data for selected countries
            comparison_data = []
            
            for country in selected_countries:
                country_data = exports_imports[exports_imports['country'] == country]
                
                if not country_data.empty:
                    # Get export and import data for each year
                    for _, row in country_data.iterrows():
                        year = row['financial_year(end)']
                        exports = row['export'] if pd.notna(row['export']) else 0
                        imports = row['import'] if pd.notna(row['import']) else 0
                        
                        comparison_data.append({
                            'Country': country,
                            'Year': year,
                            'Exports (USD millions)': exports,
                            'Imports (USD millions)': imports,
                            'Trade Balance (USD millions)': exports - imports
                        })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Group the data for analysis
                total_by_country = comparison_df.groupby('Country').agg({
                    'Exports (USD millions)': 'sum',
                    'Imports (USD millions)': 'sum'
                }).reset_index()
                
                total_by_country['Trade Balance (USD millions)'] = total_by_country['Exports (USD millions)'] - total_by_country['Imports (USD millions)']
                total_by_country['Net Exporter'] = total_by_country['Trade Balance (USD millions)'] > 0
                
                # Use a color palette based on number of countries
                if len(selected_countries) <= 3:
                    palette = "Set1"
                else:
                    palette = "Set2"
                
                # Create export comparison chart
                st.subheader("Military Exports Comparison")
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                sns.lineplot(
                    data=comparison_df, 
                    x='Year', 
                    y='Exports (USD millions)', 
                    hue='Country',
                    marker='o',
                    palette=palette,
                    ax=ax1
                )
                plt.title('Military Exports Comparison')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.legend(title="Country", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig1)
                
                # Create import comparison chart
                st.subheader("Military Imports Comparison")
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                sns.lineplot(
                    data=comparison_df, 
                    x='Year', 
                    y='Imports (USD millions)', 
                    hue='Country',
                    marker='o',
                    palette=palette,
                    ax=ax2
                )
                plt.title('Military Imports Comparison')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.legend(title="Country", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig2)
                
                # Display total metrics
                st.subheader("Overall Trade Summary")
                
                # Create metrics for each country
                metrics_cols = st.columns(len(selected_countries))
                for i, country in enumerate(selected_countries):
                    country_summary = total_by_country[total_by_country['Country'] == country]
                    if not country_summary.empty:
                        with metrics_cols[i]:
                            exports = country_summary['Exports (USD millions)'].iloc[0]
                            imports = country_summary['Imports (USD millions)'].iloc[0]
                            balance = country_summary['Trade Balance (USD millions)'].iloc[0]
                            status = "Surplus" if balance > 0 else "Deficit"
                            
                            st.metric(f"{country}", f"${balance:,.0f}M", status)
                            st.write(f"Exports: ${exports:,.0f}M")
                            st.write(f"Imports: ${imports:,.0f}M")
                
                # Create comparison bar chart
                st.subheader("Total Military Trade Balance")
                
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                bars = sns.barplot(
                    x='Country', 
                    y='Trade Balance (USD millions)', 
                    data=total_by_country,
                    palette=palette,
                    ax=ax3
                )
                
                # Add value labels
                for i, bar in enumerate(bars.patches):
                    ax3.text(
                        bar.get_x() + bar.get_width()/2.,
                        bar.get_height() + 100 if bar.get_height() >= 0 else bar.get_height() - 100,
                        f"${total_by_country['Trade Balance (USD millions)'].iloc[i]:,.0f}M",
                        ha='center',
                        va='bottom' if bar.get_height() >= 0 else 'top'
                    )
                
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.grid(True, alpha=0.3)
                plt.title('Total Military Trade Balance by Country')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig3)
                
                # Create stacked bar chart for exports vs imports
                st.subheader("Exports vs Imports by Country")
                
                # Reshape data for stacked bar chart
                stacked_data = []
                for _, row in total_by_country.iterrows():
                    stacked_data.append({
                        'Country': row['Country'],
                        'Type': 'Exports',
                        'Amount (USD millions)': row['Exports (USD millions)']
                    })
                    stacked_data.append({
                        'Country': row['Country'],
                        'Type': 'Imports',
                        'Amount (USD millions)': row['Imports (USD millions)']
                    })
                
                stacked_df = pd.DataFrame(stacked_data)
                
                fig4, ax4 = plt.subplots(figsize=(12, 6))
                sns.barplot(
                    x='Country', 
                    y='Amount (USD millions)', 
                    hue='Type',
                    data=stacked_df,
                    palette=['forestgreen', 'indianred'],
                    ax=ax4
                )
                plt.title('Military Exports vs Imports by Country')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                plt.legend(title="")
                plt.tight_layout()
                st.pyplot(fig4)
                
                # Create heatmap for year-by-year exports
                if len(selected_countries) >= 2:
                    st.subheader("Year-by-Year Export Comparison")
                    
                    # Create pivot table for heatmap
                    export_pivot = comparison_df.pivot(index='Country', columns='Year', values='Exports (USD millions)')
                    
                    # Create heatmap
                    fig5, ax5 = plt.subplots(figsize=(14, len(selected_countries) * 0.8))
                    
                    # Format values for display
                    annotations = export_pivot.map(lambda x: f'${x:,.0f}M' if pd.notna(x) and x != 0 else 'No data')
                    
                    # Create the heatmap
                    heatmap = sns.heatmap(
                        export_pivot, 
                        annot=annotations, 
                        fmt='', 
                        cmap='YlGnBu',
                        linewidths=.5, 
                        ax=ax5
                    )
                    plt.title('Military Exports by Country and Year (USD millions)')
                    plt.tight_layout()
                    st.pyplot(fig5)
            else:
                st.write("No trade data available for the selected countries.")

# 2047 Predictions page
elif page == "2047 Predictions":
    st.header("Top Military Powers Prediction for 2047")
    
    st.write("""
    This page uses historical data to predict which countries might be among the top 10 military powers by 2047.
    The prediction is based on multiple factors including military expenditure trends, current military strength, 
    and economic indicators.
    """)
    
    # Create a composite score for current strength
    def create_strength_score(df):
        # Select the most relevant columns for military power assessment
        power_columns = [
            'total_national_populations', 
            'active_service_military_manpower',
            'total_military_aircraft_strength',
            'total_combat_tank_strength',
            'navy_strength',
            'national_annual_defense_budgets',
            'purchasing_power_parities'
        ]
        
        # Clean the data by converting to numeric and handling missing values
        for col in power_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter to include only rows with complete data for key metrics
        df_clean = df.dropna(subset=[col for col in power_columns if col in df.columns])
        
        # Standardize the values
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_clean[power_columns])
        
        # Create a DataFrame with scaled values
        scaled_df = pd.DataFrame(scaled_data, columns=power_columns)
        
        # Compute a composite score (simple average of scaled values)
        scaled_df['strength_score'] = scaled_df.mean(axis=1)
        
        # Add country information back
        scaled_df['country'] = df_clean['country'].values
        scaled_df['pwr_index'] = df_clean['pwr_index'].values  # Lower is better
        
        # Sort by the composite score
        scaled_df = scaled_df.sort_values('strength_score', ascending=False)
        
        return scaled_df
    
    # Create a growth trajectory based on defense budget trends
    def analyze_growth_trajectory(strength_df, budget_df):
        # Get the country codes to match between datasets
        country_mapping = dict(zip(military_strength['country'], military_strength['country_code']))
        
        # Create a growth score based on military expenditure trends
        growth_scores = []
        
        for country in strength_df['country'].values:
            try:
                # Match country to country code
                country_code = country_mapping.get(country)
                
                if country_code:
                    # Get budget data for this country
                    country_budget = budget_df[budget_df['Country Code'] == country_code]
                    
                    if not country_budget.empty:
                        # Get last 20 years of data if available
                        years = [str(year) for year in range(2000, 2021) if str(year) in country_budget.columns]
                        
                        if years:
                            # Extract values for these years
                            values = country_budget[years].values[0]
                            
                            # Calculate trend using linear regression
                            valid_indices = ~np.isnan(values)
                            if sum(valid_indices) > 5:  # Need at least 5 data points
                                X = np.array(range(len(values)))[valid_indices].reshape(-1, 1)
                                y = values[valid_indices]
                                
                                model = LinearRegression()
                                model.fit(X, y)
                                
                                # Use slope as growth indicator
                                growth_score = model.coef_[0]
                            else:
                                growth_score = 0
                        else:
                            growth_score = 0
                    else:
                        growth_score = 0
                else:
                    growth_score = 0
            except Exception as e:
                growth_score = 0
                
            growth_scores.append(growth_score)
        
        # Add growth scores to the strength dataframe
        strength_df['growth_score'] = growth_scores
        
        # Normalize growth scores
        min_growth = strength_df['growth_score'].min()
        max_growth = strength_df['growth_score'].max()
        if max_growth > min_growth:
            strength_df['growth_score_normalized'] = (strength_df['growth_score'] - min_growth) / (max_growth - min_growth)
        else:
            strength_df['growth_score_normalized'] = 0
            
        return strength_df
    
    # Predict future ranking based on current strength and growth trajectory
    def predict_future_ranking(df, target_year=2047):
        current_year = 2024
        years_projection = target_year - current_year
        
        # Create a projection score
        # Weight current strength more heavily for near-term, growth more heavily for long-term
        strength_weight = 0.7  # Starting weight for current strength
        growth_weight = 0.3    # Starting weight for growth trajectory
        
        # Adjust weights based on projection period
        if years_projection > 10:
            # For longer term projections, growth becomes more important
            strength_weight = 0.5
            growth_weight = 0.5
        
        # Calculate projection score
        df['projection_score'] = (
            strength_weight * df['strength_score'] + 
            growth_weight * df['growth_score_normalized'] -
            0.2 * df['pwr_index']  # Lower PWR index is better
        )
        
        # Sort countries by projection score
        df_sorted = df.sort_values('projection_score', ascending=False)
        
        return df_sorted
    
    # Execute the prediction
    try:
        with st.spinner("Generating predictions..."):
            # Step 1: Calculate current strength scores
            strength_df = create_strength_score(military_strength)
            
            # Step 2: Analyze growth trajectories
            projection_df = analyze_growth_trajectory(strength_df, defense_budget)
            
            # Step 3: Predict future rankings
            future_ranking = predict_future_ranking(projection_df)
            
            # Display current top 10 vs predicted top 10
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Current Top 10 Military Powers (2024)")
                current_top10 = strength_df[['country', 'strength_score']].head(10)
                current_top10.columns = ['Country', 'Strength Score']
                st.table(current_top10)
            
            with col2:
                st.subheader("Predicted Top 10 Military Powers (2047)")
                future_top10 = future_ranking[['country', 'projection_score']].head(10)
                future_top10.columns = ['Country', 'Projection Score']
                st.table(future_top10)
            
            # Visualization comparing current vs future rankings
            st.subheader("Current vs Predicted Top Military Powers")
            
            # Create a visualization showing movement in rankings
            current_ranks = {country: idx+1 for idx, country in enumerate(strength_df['country'].head(15))}
            future_ranks = {country: idx+1 for idx, country in enumerate(future_ranking['country'].head(15))}
            
            # Combine data for visualization
            vis_data = []
            all_countries = set(list(current_ranks.keys()) + list(future_ranks.keys()))
            
            for country in all_countries:
                current_rank = current_ranks.get(country, 20)  # Default to 20 if not in top 15
                future_rank = future_ranks.get(country, 20)    # Default to 20 if not in top 15
                
                # Only include if either current or future rank is within top 15
                if current_rank <= 15 or future_rank <= 15:
                    vis_data.append({
                        'Country': country,
                        'Current Rank (2024)': current_rank,
                        'Predicted Rank (2047)': future_rank,
                        'Rank Change': current_rank - future_rank  # Positive means improvement
                    })
            
            vis_df = pd.DataFrame(vis_data)
            vis_df = vis_df.sort_values('Predicted Rank (2047)')
            
            # Plot the rank changes
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot connecting lines between current and future rankings
            for _, row in vis_df.iterrows():
                ax.plot([1, 2], [row['Current Rank (2024)'], row['Predicted Rank (2047)']], 
                        'k-', alpha=0.3)
            
            # Plot current rankings
            ax.scatter([1] * len(vis_df), vis_df['Current Rank (2024)'], 
                      s=100, label='2024 Ranking')
            
            # Plot future rankings
            ax.scatter([2] * len(vis_df), vis_df['Predicted Rank (2047)'], 
                      s=100, label='2047 Ranking')
            
            # Add country labels
            for _, row in vis_df.iterrows():
                ax.text(0.9, row['Current Rank (2024)'], row['Country'], 
                       ha='right', va='center')
                ax.text(2.1, row['Predicted Rank (2047)'], row['Country'], 
                       ha='left', va='center')
            
            # Customize the plot
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['2024', '2047'])
            ax.set_ylabel('Rank')
            ax.set_ylim(16, 0)  # Reverse y-axis so rank 1 is at the top
            ax.set_title('Projected Changes in Military Power Rankings (2024-2047)')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
            
            # Add explanatory notes
            st.subheader("Key Factors Influencing the Predictions")
            st.markdown("""
            The predictions are based on several key factors:
            
            1. **Current Military Strength**: Based on military personnel, equipment, and resources.
            2. **Defense Budget Trends**: Countries with increasing defense spending are projected to rise.
            3. **Economic Indicators**: Economic strength supports military capability.
            4. **Technological Advancement**: Countries investing in military technology are likely to gain advantage.
            5. **Geopolitical Factors**: Regional influence and international alliances.
            
            **Note**: These predictions have limitations and should be interpreted with caution. 
            Unforeseen events, policy changes, and technological breakthroughs can significantly alter trajectories.
            """)
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Recommendation: Review the data for completeness and consistency to improve prediction accuracy.")
