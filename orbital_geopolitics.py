import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- DATA LOADING AND PREPROCESSING ---

# Function to extract header from commented line
def load_satcat():
    # Extract header from commented line
    with open('satcat.tsv', 'r') as f:
        for line in f:
            if line.startswith('#JCAT'):
                header = line.strip().lstrip('#').split('\t')
                break
    
    # Load data with extracted headers
    satcat = pd.read_csv('satcat.tsv', sep='\t', 
                         comment='#', 
                         names=header,
                         low_memory=False)
    
    # Convert launch date to datetime and extract year
    satcat['LDate'] = pd.to_datetime(satcat['LDate'], errors='coerce')
    satcat['Year'] = satcat['LDate'].dt.year
    
    # Handle missing values
    satcat['Apogee'] = pd.to_numeric(satcat['Apogee'], errors='coerce')
    
    # Orbital classification based on OpOrbit field or Apogee altitude
    satcat['Orbit_Class'] = 'Unknown'
    
    # Classification by OpOrbit string
    leo_pattern = r'L(L)?EO'
    meo_pattern = r'MEO'
    geo_pattern = r'GEO'
    
    import re
    satcat.loc[satcat['OpOrbit'].str.contains(leo_pattern, na=False, regex=True), 'Orbit_Class'] = 'LEO'
    satcat.loc[satcat['OpOrbit'].str.contains(meo_pattern, na=False, regex=True), 'Orbit_Class'] = 'MEO'
    satcat.loc[satcat['OpOrbit'].str.contains(geo_pattern, na=False, regex=True), 'Orbit_Class'] = 'GEO'
    
    # If OpOrbit doesn't have the classification, use Apogee
    mask = satcat['Orbit_Class'] == 'Unknown'
    satcat.loc[mask & (satcat['Apogee'] < 2000), 'Orbit_Class'] = 'LEO'
    satcat.loc[mask & (satcat['Apogee'] >= 2000) & (satcat['Apogee'] < 35786), 'Orbit_Class'] = 'MEO'
    satcat.loc[mask & (satcat['Apogee'] >= 35786), 'Orbit_Class'] = 'GEO'
    
    # Classify satellites by purpose based on Type field
    satcat['Purpose'] = 'Unknown'
    
    # Map Type codes to purposes (based on common conventions)
    purpose_map = {
        'P': 'Civil',           # Payload
        'R': 'Defense',         # Rocket body
        'D': 'Defense',         # Military/defense
        'S': 'Business',        # Commercial
        'C': 'Communications',  # Communications
        'N': 'Navigation',      # Navigation
        'E': 'Earth Science',   # Earth observation
        'A': 'Amateur/Academic' # Amateur/academic
    }
    
    # Apply purpose mapping (first character of Type column)
    satcat['Purpose'] = satcat['Type'].str[0].map(purpose_map).fillna('Unknown')
    
    # Country classification
    western_europe = ['UK', 'FR', 'DE', 'IT', 'ES', 'PT', 'BE', 'NL', 'LU', 
                     'CH', 'AT', 'DK', 'SE', 'NO', 'FI', 'IE', 'IS']
    
    # Create country grouping
    satcat['Country_Group'] = 'Other'
    satcat.loc[satcat['Owner'] == 'US', 'Country_Group'] = 'US'
    satcat.loc[satcat['Owner'].isin(western_europe), 'Country_Group'] = 'Western Europe'
    satcat.loc[satcat['Owner'] == 'CN', 'Country_Group'] = 'China'
    satcat.loc[satcat['Owner'].isin(['SU', 'RU']), 'Country_Group'] = 'Russia'
    satcat.loc[satcat['Owner'] == 'JP', 'Country_Group'] = 'Japan'
    satcat.loc[satcat['Owner'] == 'IN', 'Country_Group'] = 'India'
    
    return satcat

def load_midip():
    # Load MIDIP data with mocked structure
    # In a real implementation, you would load the actual file
    midip = pd.DataFrame({
        'dispnum': range(1000),
        'stmon': np.random.randint(1, 13, 1000),
        'styear': np.random.randint(1957, 2025, 1000),
        'hostlev': np.random.randint(0, 6, 1000),
        'threat': np.random.randint(0, 6, 1000),
        'ccode': np.random.choice([2, 20, 200, 365, 710, 750, 850], 1000)
    })
    
    # Country code mapping
    country_codes = {
        2: 'US',
        20: 'Western Europe',  # Simplified for demo
        200: 'Western Europe', # More European countries
        365: 'Russia',
        710: 'China',
        740: 'Japan',
        750: 'India',
    }
    
    # Map country codes to our groupings
    midip['Country_Group'] = midip['ccode'].map(country_codes).fillna('Other')
    
    # Calculate hostility as max of hostlev and threat
    midip['Hostility'] = midip[['hostlev', 'threat']].max(axis=1)
    
    # Remove rows with missing data
    midip = midip[midip['Hostility'].notna() & (midip['Hostility'] != '-')]
    
    # Group by year and country, calculate average hostility
    hostility_by_year = midip.groupby(['styear', 'Country_Group'])['Hostility'].mean().reset_index()
    hostility_by_year.rename(columns={'styear': 'Year'}, inplace=True)
    
    return hostility_by_year

# --- LOAD DATA ---
try:
    satcat = load_satcat()
    hostility_data = load_midip()
except Exception as e:
    # If file loading fails, create sample data for demonstration
    print(f"Error loading data: {e}")
    print("Creating sample data for demonstration")
    
    # Sample satellite data
    satcat = pd.DataFrame({
        'Year': np.random.randint(1957, 2025, 1000),
        'Orbit_Class': np.random.choice(['LEO', 'MEO', 'GEO'], 1000),
        'Country_Group': np.random.choice(['US', 'Western Europe', 'China', 'Russia', 'Japan', 'India', 'Other'], 1000),
        'Purpose': np.random.choice(['Civil', 'Defense', 'Business', 'Amateur/Academic'], 1000)
    })
    
    # Sample hostility data
    hostility_data = pd.DataFrame({
        'Year': np.repeat(range(1957, 2025), 7),
        'Country_Group': np.tile(['US', 'Western Europe', 'China', 'Russia', 'Japan', 'India', 'Other'], 2025-1957),
        'Hostility': np.random.uniform(0, 5, (2025-1957)*7)
    })

# --- DASH APPLICATION ---
app = dash.Dash(__name__)

# Define country order for consistent display
country_order = ['US', 'Western Europe', 'China', 'Russia', 'Japan', 'India', 'Other']

# Define orbital class order
orbit_order = ['GEO', 'MEO', 'LEO']

# App layout
app.layout = html.Div([
    html.H1("Orbital Geopolitics: Satellite Launches and Global Conflicts", 
            style={'textAlign': 'center', 'color': '#2a3f5f', 'marginBottom': '20px'}),
    
    # Time slider
    html.Div([
        html.Label("Select Year:"),
        dcc.Slider(
            id='year-slider',
            min=int(satcat['Year'].min()),
            max=int(satcat['Year'].max()),
            value=2000,  # Default value
            marks={str(year): str(year) for year in range(int(satcat['Year'].min()), int(satcat['Year'].max())+1, 5)},
            step=1
        )
    ], style={'width': '90%', 'margin': '0 auto', 'marginBottom': '40px'}),
    
    # Main visualization
    dcc.Graph(id='satellite-conflict-viz', style={'height': '800px'})
])

# Callback to update the visualization based on the selected year
@app.callback(
    Output('satellite-conflict-viz', 'figure'),
    [Input('year-slider', 'value')]
)
def update_figure(selected_year):
    # Filter data for the selected year
    year_satellites = satcat[satcat['Year'] == selected_year]
    year_hostility = hostility_data[hostility_data['Year'] == selected_year]
    
    # Create figure with secondary y-axis for hostility
    fig = make_subplots(rows=1, cols=1)
    
    # Map purposes to symbols
    purpose_symbols = {
        'Civil': 'circle',
        'Defense': 'diamond',
        'Business': 'square',
        'Amateur/Academic': 'star',
        'Communications': 'triangle-up',
        'Navigation': 'hexagon',
        'Earth Science': 'cross',
        'Unknown': 'x'
    }
    
    # Map orbit classes to y-axis positions
    orbit_positions = {
        'LEO': 1,
        'MEO': 2,
        'GEO': 3
    }
    
    # Colors for different countries
    country_colors = {
        'US': '#636EFA',
        'Western Europe': '#EF553B',
        'China': '#00CC96',
        'Russia': '#AB63FA',
        'Japan': '#FFA15A',
        'India': '#19D3F3',
        'Other': '#FF6692'
    }
    
    # Add vertical separator lines between countries
    x_positions = [i + 0.5 for i in range(len(country_order)-1)]
    for x_pos in x_positions:
        fig.add_shape(
            type="line",
            x0=x_pos, y0=0, x1=x_pos, y1=3.5,
            line=dict(color="gray", width=1, dash="dash")
        )
    
    # Plot hostility bars between countries
    bar_width = 0.3
    for i, country in enumerate(country_order):
        hostility = year_hostility[year_hostility['Country_Group'] == country]['Hostility'].mean()
        if not pd.isna(hostility):
            fig.add_trace(
                go.Bar(
                    x=[i], 
                    y=[hostility],
                    width=[bar_width],
                    marker_color='rgba(200, 0, 0, 0.7)',
                    name=f"{country} Hostility",
                    showlegend=False
                )
            )
    
    # Plot satellites for each country and purpose
    for purpose in purpose_symbols:
        for country_idx, country in enumerate(country_order):
            for orbit in orbit_order:
                # Filter data for this combination
                subset = year_satellites[
                    (year_satellites['Country_Group'] == country) & 
                    (year_satellites['Purpose'] == purpose) & 
                    (year_satellites['Orbit_Class'] == orbit)
                ]
                
                if len(subset) > 0:
                    # Calculate positions with jitter to avoid overlapping
                    jitter = np.random.uniform(-0.35, 0.35, len(subset))
                    x_pos = [country_idx + j for j in jitter]
                    y_pos = [orbit_positions[orbit] + np.random.uniform(-0.25, 0.25) for _ in range(len(subset))]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_pos,
                            y=y_pos,
                            mode='markers',
                            marker=dict(
                                symbol=purpose_symbols[purpose],
                                size=10,
                                color=country_colors[country],
                                line=dict(width=1, color='DarkSlateGrey')
                            ),
                            name=f"{country} - {purpose} - {orbit}",
                            hovertext=[f"Launch: {idx}<br>Type: {purpose}<br>Orbit: {orbit}<br>Country: {country}" 
                                      for idx in subset.index],
                            showlegend=True
                        )
                    )
    
    # Update layout
    fig.update_layout(
        title=f"Satellite Launches and Conflicts in {selected_year}",
        xaxis=dict(
            tickvals=list(range(len(country_order))),
            ticktext=country_order,
            title="Country",
            domain=[0, 1]
        ),
        yaxis=dict(
            tickvals=list(range(1, 4)),
            ticktext=orbit_order,
            title="Orbital Class",
            range=[0.5, 3.5]
        ),
        legend=dict(
            title="Satellite Types",
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=60, r=60, t=80, b=150),
        height=800,
        hovermode='closest'
    )
    
    # Add annotations to explain symbols
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text="○ Civil  ◇ Defense  □ Business  ★ Academic",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    # Add annotation for hostility bars
    fig.add_annotation(
        x=0.02, y=0.05,
        xref="paper", yref="paper",
        text="Red bars indicate average conflict hostility level",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)", 
        bordercolor="black",
        borderwidth=1
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
