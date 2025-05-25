import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# --- DATA LOADING ---
def load_satcat(file_path='output.csv'):
    """Load and preprocess actual satellite catalog data"""
    # Load the SATCAT data
    satcat = pd.read_csv(file_path, low_memory=False)
    
    # Convert launch date to datetime and extract year
    satcat['LDate'] = pd.to_datetime(satcat['LDate'], errors='coerce')
    satcat['Year'] = satcat['LDate'].dt.year
    
    # Ensure Apogee is numeric
    satcat['Apogee'] = pd.to_numeric(satcat['Apogee'], errors='coerce')
    
    # Orbital classification based on OpOrbit field and Apogee
    # First check OpOrbit string patterns
    satcat['Orbit_Class'] = 'Unknown'
    
    # Using string contains for classification
    satcat['Orbit_Class'] = np.where(satcat['OpOrbit'].str.contains('LEO', na=False, regex=False), 'LEO', satcat['Orbit_Class'])
    satcat['Orbit_Class'] = np.where(satcat['OpOrbit'].str.contains('MEO', na=False, regex=False), 'MEO', satcat['Orbit_Class'])
    satcat['Orbit_Class'] = np.where(satcat['OpOrbit'].str.contains('GEO', na=False, regex=False), 'GEO', satcat['Orbit_Class'])
    
    # For still Unknown orbits, classify based on Apogee values
    mask = satcat['Orbit_Class'] == 'Unknown'
    satcat['Orbit_Class'] = np.where(mask & (satcat['Apogee'] < 2000), 'LEO', satcat['Orbit_Class'])
    satcat['Orbit_Class'] = np.where(mask & (satcat['Apogee'] >= 2000) & (satcat['Apogee'] < 35786), 'MEO', satcat['Orbit_Class'])
    satcat['Orbit_Class'] = np.where(mask & (satcat['Apogee'] >= 35786), 'GEO', satcat['Orbit_Class'])
    
    # Define purpose mapping based on Type field
    purpose_map = {
        'P': 'Civil',
        'R': 'Defense',
        'D': 'Defense',
        'S': 'Business',
        'C': 'Communications',
        'N': 'Navigation',
        'E': 'Earth Science',
        'A': 'Amateur/Academic'
    }
    
    # Extract first character of Type for purpose classification
    satcat['Purpose'] = satcat['Type'].str[0].map(purpose_map).fillna('Unknown')
    
    # Country classification
    western_europe = ['UK', 'FR', 'DE', 'IT', 'ES', 'PT', 'BE', 'NL', 'LU', 
                      'CH', 'AT', 'DK', 'SE', 'NO', 'FI', 'IE', 'IS']
    
    # Create country grouping
    country_map = {**{c: 'Western Europe' for c in western_europe},
                  'US': 'US', 'CN': 'China', 'SU': 'Russia', 
                  'RU': 'Russia', 'JP': 'Japan', 'IN': 'India'}
    
    satcat['Country_Group'] = satcat['Owner'].map(country_map).fillna('Other')
    
    return satcat

def load_midip(file_path='MIDIP 5.0.csv'):
    """Load and preprocess actual MIDIP conflict dataset"""
    # Load MIDIP data
    midip = pd.read_csv(file_path, na_values=['-', 'nan', ''], low_memory=False)
    
    # Country code mapping according to CoW country codes
    country_codes = {
        2: 'US',        # USA
        200: 'Western Europe',  # UK
        220: 'Western Europe',  # France
        255: 'Western Europe',  # Germany
        325: 'Western Europe',  # Italy
        365: 'Russia',   # Russia/USSR
        710: 'China',    # China
        740: 'Japan',    # Japan
        750: 'India'     # India
    }
    
    # Apply country mapping
    midip['Country_Group'] = midip['ccode'].map(country_codes).fillna('Other')
    
    # Use hostlev as the hostility indicator
    midip['Hostility'] = pd.to_numeric(midip['hostlev'], errors='coerce')
    
    # Drop rows with missing hostility data
    midip_clean = midip.dropna(subset=['Hostility'])
    
    # Group by year and country, calculate mean hostility
    hostility_by_year = midip_clean.groupby(['styear', 'Country_Group'])['Hostility'].mean().reset_index()
    hostility_by_year.rename(columns={'styear': 'Year'}, inplace=True)
    
    return hostility_by_year

# --- LOAD DATA ---
satcat = load_satcat()
hostility_data = load_midip()

# --- DASH APPLICATION ---
app = dash.Dash(__name__)

# Define country order and colors
country_order = ['US', 'Western Europe', 'China', 'Russia', 'Japan', 'India', 'Other']
country_colors = {
    'US': '#636EFA',
    'Western Europe': '#EF553B',
    'China': '#00CC96',
    'Russia': '#AB63FA',
    'Japan': '#FFA15A',
    'India': '#19D3F3',
    'Other': '#FF6692'
}

# Define orbit order
orbit_order = ['GEO', 'MEO', 'LEO']

# App layout
app.layout = html.Div([
    html.H1("Orbital Geopolitics: Satellites & Global Conflicts", 
            style={'textAlign': 'center', 'fontFamily': 'Arial', 'color': '#2a3f5f'}),
    
    # Year selector
    html.Div([
        html.Label("Select Year:"),
        dcc.Slider(
            id='year-slider',
            min=int(satcat['Year'].min()),
            max=int(satcat['Year'].max()),
            value=2000,
            marks={i: str(i) for i in range(int(satcat['Year'].min()), int(satcat['Year'].max())+1, 5)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'width': '90%', 'margin': '20px auto'}),
    
    # Main visualization
    dcc.Graph(id='satellite-plot', style={'height': '800px'})
])

# Callback to update the visualization
@app.callback(
    Output('satellite-plot', 'figure'),
    Input('year-slider', 'value')
)
def update_figure(selected_year):
    # Filter data for selected year
    satellites = satcat[satcat['Year'] == selected_year]
    hostility = hostility_data[hostility_data['Year'] == selected_year]
    
    # Create figure
    fig = make_subplots(rows=1, cols=1)
    
    # Purpose symbol mapping
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
    
    # Orbit position mapping
    orbit_positions = {'LEO': 1, 'MEO': 2, 'GEO': 3}
    
    # Add country separators
    for i in range(len(country_order)-1):
        fig.add_shape(
            type="line",
            x0=i + 0.5, y0=0, x1=i + 0.5, y1=3.5,
            line=dict(color="gray", width=1, dash="dash")
        )
    
    # Add hostility bars
    for i, country in enumerate(country_order):
        country_hostility = hostility[hostility['Country_Group'] == country]['Hostility'].mean()
        if not pd.isna(country_hostility):
            fig.add_trace(
                go.Bar(
                    x=[i], 
                    y=[country_hostility],
                    width=0.3,
                    marker_color='rgba(200, 0, 0, 0.7)',
                    name=f"{country} Conflict Level",
                    showlegend=False
                )
            )
    
    # Add satellites as scatter points by purpose and orbit
    for purpose in purpose_symbols:
        for orbit in orbit_order:
            # Filter satellites for purpose and orbit
            subset = satellites[
                (satellites['Purpose'] == purpose) & 
                (satellites['Orbit_Class'] == orbit)
            ]
            
            if len(subset) > 0:
                # Process each country
                for i, country in enumerate(country_order):
                    country_subset = subset[subset['Country_Group'] == country]
                    
                    if len(country_subset) > 0:
                        # Add jitter for point separation
                        x_jitter = np.random.uniform(-0.35, 0.35, len(country_subset))
                        y_jitter = np.random.uniform(-0.2, 0.2, len(country_subset))
                        
                        # Add scatter points
                        fig.add_trace(
                            go.Scatter(
                                x=[i + j for j in x_jitter],
                                y=[orbit_positions[orbit] + j for j in y_jitter],
                                mode='markers',
                                marker=dict(
                                    symbol=purpose_symbols[purpose],
                                    size=10,
                                    color=country_colors[country],
                                    line=dict(width=1, color='DarkSlateGrey')
                                ),
                                name=f"{country} {purpose} {orbit}",
                                hovertemplate=(
                                    f"Country: {country}<br>" +
                                    f"Purpose: {purpose}<br>" +
                                    f"Orbit: {orbit}<br>" +
                                    "Count: %{marker.size}"
                                ),
                                legendgroup=purpose
                            )
                        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Satellite Launches and Global Conflicts in {selected_year}",
            font=dict(size=24)
        ),
        xaxis=dict(
            tickvals=list(range(len(country_order))),
            ticktext=country_order,
            title="Country/Region"
        ),
        yaxis=dict(
            tickvals=list(range(1, 4)),
            ticktext=orbit_order,
            title="Orbital Class",
            range=[0.5, 3.5]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=60, r=60, t=80, b=150),
        height=800,
        hovermode='closest',
        template="plotly_white"
    )
    
    # Add symbol legend
    fig.add_annotation(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        text="○ Civil  ◇ Defense  □ Business  ★ Academic",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    # Add hostility legend
    fig.add_annotation(
        x=0.01,
        y=0.05,
        xref="paper",
        yref="paper",
        text="Red bars indicate average conflict hostility level",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
