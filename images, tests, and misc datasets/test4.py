import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- DATA LOADING ---
def load_satcat(file_path='output.csv'):
    """Load and preprocess satellite catalog data"""
    satcat = pd.read_csv(file_path, low_memory=False)
    
    # Temporal filtering
    satcat['LDate'] = pd.to_datetime(satcat['LDate'], errors='coerce')
    satcat['Year'] = satcat['LDate'].dt.year
    satcat = satcat[(satcat['Year'] >= 1992) & (satcat['Year'] <= 2015)]

    # Orbital classification
    satcat['Apogee'] = pd.to_numeric(satcat['Apogee'], errors='coerce')
    satcat['Orbit_Class'] = 'Unknown'
    satcat['Orbit_Class'] = np.where(satcat['OpOrbit'].str.contains('LEO', na=False), 'LEO', satcat['Orbit_Class'])
    satcat['Orbit_Class'] = np.where(satcat['OpOrbit'].str.contains('MEO', na=False), 'MEO', satcat['Orbit_Class'])
    satcat['Orbit_Class'] = np.where(satcat['OpOrbit'].str.contains('GEO', na=False), 'GEO', satcat['Orbit_Class'])
    
    mask = satcat['Orbit_Class'] == 'Unknown'
    satcat['Orbit_Class'] = np.where(mask & (satcat['Apogee'] < 2000), 'LEO', satcat['Orbit_Class'])
    satcat['Orbit_Class'] = np.where(mask & (satcat['Apogee'] >= 2000) & (satcat['Apogee'] < 35786), 'MEO', satcat['Orbit_Class'])
    satcat['Orbit_Class'] = np.where(mask & (satcat['Apogee'] >= 35786), 'GEO', satcat['Orbit_Class'])

    # Purpose classification
    purpose_map = {
        'P': 'Civil', 'R': 'Defense', 'D': 'Defense',
        'S': 'Business', 'C': 'Communications', 
        'N': 'Navigation', 'E': 'Earth Science', 
        'A': 'Amateur/Academic'
    }
    satcat['Purpose'] = satcat['Type'].str[0].map(purpose_map).fillna('Unknown')

    # Country classification system
    country_map = {
        'US': 'US', 'CN': 'China', 'RU': 'Russia', 'SU': 'Russia',
        'J': 'Japan', 'IN': 'India', 'F': 'Western Europe', 
        'D': 'Western Europe', 'GB': 'Western Europe', 
        'I': 'Western Europe', 'E': 'Western Europe', 
        'NL': 'Western Europe', 'S': 'Western Europe', 
        'CH': 'Western Europe'
    }

    # Organization to country mapping
    org_mapping = {
        'NASA': 'US', 'ISRO': 'India', 'JAXA': 'Japan',
        'ESA': 'Western Europe', 'ROSCOSMOS': 'Russia',
        'CNSA': 'China', 'CNES': 'Western Europe',
        'DLR': 'Western Europe', 'UKSA': 'Western Europe'
    }

    # Clean and split State codes
    satcat['Primary_Code'] = (
        satcat['State']
        .str.split(r'[-/]', n=1).str[0]
        .str.strip()
        .replace('', np.nan)
    )

    # Apply mappings
    satcat['Country_Group'] = (
        satcat['Primary_Code']
        .map(org_mapping)
        .fillna(satcat['Primary_Code'].map(country_map))
        .fillna('Other')
    )

    return satcat

def load_midip(file_path='MIDIP 5.0.csv'):
    """Load conflict data"""
    midip = pd.read_csv(file_path, na_values=['-', 'nan', ''])
    country_codes = {
        2: 'US', 200: 'Western Europe', 220: 'Western Europe',
        255: 'Western Europe', 325: 'Western Europe', 365: 'Russia',
        710: 'China', 740: 'Japan', 750: 'India'
    }
    
    midip['Country_Group'] = midip['ccode'].map(country_codes).fillna('Other')
    midip['Hostility'] = pd.to_numeric(midip['hostlev'], errors='coerce')
    return midip.dropna(subset=['Hostility']).groupby(['styear', 'Country_Group'])['Hostility'].mean().reset_index().rename(columns={'styear': 'Year'})

# --- INITIALIZE DATA ---
satcat = load_satcat()
hostility_data = load_midip()

# --- DASH APP ---
app = dash.Dash(__name__)

country_order = ['US', 'China', 'Russia', 'Japan', 'India', 'Western Europe', 'Other']
country_colors = {
    'US': '#636EFA', 'China': '#00CC96', 'Russia': '#AB63FA',
    'Japan': '#FFA15A', 'India': '#19D3F3', 
    'Western Europe': '#EF553B', 'Other': '#FF6692'
}

app.layout = html.Div([
    html.H1("Orbital Geopolitics: Satellites & Global Conflicts", 
            style={'textAlign': 'center', 'color': '#2a3f5f'}),
    html.Div([
        dcc.Slider(
            id='year-slider',
            min=1992, max=2014, value=2000,
            marks={i: str(i) for i in range(1992, 2015, 5)},
            step=1,
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'width': '90%', 'margin': '20px auto'}),
    dcc.Graph(id='satellite-plot', style={'height': '800px'})
])

@app.callback(
    Output('satellite-plot', 'figure'),
    Input('year-slider', 'value')
)
def update_figure(selected_year):
    fig = make_subplots(specs=[[{"secondary_y": True}]],
                        shared_xaxes=True) # ensuring x axes are shared
    satellites = satcat[satcat['Year'] == selected_year]
    hostility = hostility_data[hostility_data['Year'] == selected_year]

    

    # Add hostility bars FIRST (background layer)
    for i, country in enumerate(country_order):
        hostility_level = hostility[hostility['Country_Group'] == country]['Hostility'].mean()
        if not pd.isna(hostility_level):
            fig.add_trace(go.Bar(
                x=[i], y=[hostility_level], width=0.3,
                marker_color='rgba(200, 0, 0, 0.2)', 
                showlegend=False,
            ), secondary_y=True)

    # Add country separators
    for i in range(len(country_order)-1):
        fig.add_shape(type="line", x0=i+0.5, y0=0, x1=i+0.5, y1=3.5,
                     line=dict(color="gray", width=1, dash="dash"))

    # Add satellite markers (foreground layer)
    purpose_symbols = {
        'Civil': 'circle',
        'Defense': 'diamond',
        'Communications': 'triangle-up'
    }
    orbit_positions = {'LEO': 1, 'MEO': 2, 'GEO': 3}
    
    for purpose, symbol in purpose_symbols.items():
        for orbit in ['LEO', 'MEO', 'GEO']:
            subset = satellites[
                (satellites['Purpose'] == purpose) & 
                (satellites['Orbit_Class'] == orbit)
            ]
            for i, country in enumerate(country_order):
                country_subset = subset[subset['Country_Group'] == country]
                if not country_subset.empty:
                    x_jitter = np.random.uniform(-0.35, 0.35, len(country_subset))
                    y_jitter = np.random.uniform(-0.2, 0.2, len(country_subset))
                    fig.add_trace(go.Scatter(
                        x=[i + j for j in x_jitter],
                        y=[orbit_positions[orbit] + j for j in y_jitter],
                        mode='markers',
                        marker_symbol=symbol,
                        marker=dict(
                            size=10,
                            color=country_colors[country],
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        name=purpose,
                        legendgroup=purpose,
                        showlegend=True if (i==0 and orbit=='LEO') else False,
                        hovertemplate=(
                            f"<b>{country}</b><br>Purpose: {purpose}<br>"
                            f"Orbit: {orbit}<br>Count: {len(country_subset)}"
                        )
                    ),
                    secondary_y=False) # Assign to primary y-axis

    # Layout configuration
    fig.update_layout(
        title=f"Satellite Launches and Global Conflicts in {selected_year}",

        yaxis2=dict(
            title="Hostility Level (1-5)",
            range=[0, 5],
            side='right',
            showgrid=False,
            overlaying='y',
            layer='below traces'
        ),

        xaxis={'tickvals': list(range(len(country_order))), 
              'ticktext': country_order, 'title': 'Country/Region'},
       
        yaxis={
            
            'tickvals': [1,2,3],
            'ticktext': ['LEO', 'MEO', 'GEO'],
            'title': 'Orbital Class',
            'range': [0.5, 3.5],
            'layer' : 'above traces',
        },
        template="plotly_white",
        margin=dict(b=150),
        annotations=[dict(
            x=0.5, y=1.07, xref='paper', yref='paper',
            text="<b>○ Civil ◇ Defense △ Comms</b>",
            showarrow=False,
            font=dict(size=16),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#2a3f5f"
        )]
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
