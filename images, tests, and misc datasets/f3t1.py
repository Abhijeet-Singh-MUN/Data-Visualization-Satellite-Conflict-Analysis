import pandas as pd
import plotly.express as px
from datetime import datetime

# Load dataset
df = pd.read_csv('output.csv')

# Country mapping
country_map = {
    'US': 'US', 'CN': 'China', 'RU': 'Russia', 'SU': 'Russia',
    'J': 'Japan', 'IN': 'India', 'F': 'Western Europe', 
    'D': 'Western Europe', 'GB': 'Western Europe', 
    'I': 'Western Europe', 'E': 'Western Europe', 
    'NL': 'Western Europe', 'S': 'Western Europe', 
    'CH': 'Western Europe'
}

def process_data(df):
    # Filter relevant columns and valid entries
    df = df[['State', 'SDate', 'DDate', 'OpOrbit']].dropna(subset=['SDate', 'DDate'])
    
    # Map countries using State column
    df['Country'] = df['State'].map(country_map).fillna('Other')
    
    # Calculate lifespan in years
    date_format = '%Y %b %d'
    today = datetime(2025, 4, 11)  # Current date from context
    
    def get_lifespan(row):
        try:
            # Handle different date formats
            launch_str = ' '.join(row['SDate'].split()[:3])  # Take first 3 elements
            decay_str = ' '.join(row['DDate'].split()[:3]) if pd.notna(row['DDate']) else today
            
            launch = datetime.strptime(launch_str, '%Y %b %d')
            decay = datetime.strptime(decay_str, '%Y %b %d') if isinstance(decay_str, str) else decay_str
            
            return (decay - launch).days / 365.25
        except Exception as e:
            return None
    
    df['Lifespan'] = df.apply(get_lifespan, axis=1)
    df = df.dropna(subset=['Lifespan'])
    
    # Classify orbits
    def classify_orbit(op):
        op = str(op).upper()
        if 'LEO' in op: return 'LEO'
        if 'MEO' in op: return 'MEO'
        if 'GEO' in op or 'HEO' in op: return 'GEO'
        return 'Other'
    
    df['Orbit'] = df['OpOrbit'].apply(classify_orbit)
    
    return df

processed_df = process_data(df)

# Create hierarchical data with average lifespans
agg_df = processed_df.groupby(['Country', 'Orbit'])['Lifespan'].mean().reset_index()

# Create sunburst chart
fig = px.sunburst(
    agg_df,
    path=['Country', 'Orbit'],
    values='Lifespan',
    color='Lifespan',
    color_continuous_scale='Viridis',
    title='Satellite Lifespan by Country and Orbit (1957-1963)',
    height=800,
    branchvalues='total'
)

# Customize layout
fig.update_layout(
    margin=dict(t=40, l=0, r=0, b=0),
    coloraxis_colorbar=dict(
        title='Avg Lifespan (Years)',
        thickness=15,
        len=0.6
    ),
    title_font_size=20
)

fig.update_traces(
    textinfo='label+value',
    insidetextorientation='radial'
)

fig.show()
