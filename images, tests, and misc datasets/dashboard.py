# main_dashboard.py
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from test4 import load_satcat, load_midip, country_order, country_colors
from f3t1 import process_data, country_map
from f2t8 import calculate_cumulative_mass, load_satcat as load_satcat_f2t8
import pandas as pd

# Initialize unified Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Load and transform data once
print("Loading datasets...")
satcat_base = load_satcat()
hostility_data = load_midip()
processed_f3t1 = process_data(pd.read_csv('output.csv'))
cumulative_mass = calculate_cumulative_mass(load_satcat_f2t8()).melt(
    id_vars='Year', var_name='Country', value_name='Mass'
)

# Custom CSS for scrollable layout
styles = {
    'container': {
        'height': '100vh',
        'overflowY': 'auto',
        'padding': '20px'
    },
    'section': {
        'minHeight': '100vh',
        'marginBottom': '50px',
        'borderBottom': '1px solid #eee'
    }
}

app.layout = html.Div(style=styles['container'], children=[
    html.H1("Space Program Analysis Dashboard", style={'textAlign': 'center'}),
    
    # Visualization 1: test4.py integration
    html.Div(style=styles['section'], children=[
        html.H2("Satellite Launches & Global Conflicts"),
        dcc.Slider(
            id='test4-year-slider',
            min=1992, max=2014, value=2000,
            marks={i: str(i) for i in range(1992, 2015, 5)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        dcc.Graph(id='test4-main-plot', style={'height': '80vh'})
    ]),
    
    # Visualization 2: f3t1.py integration
    html.Div(style=styles['section'], children=[
        html.H2("Satellite Lifespan Analysis"),
        dcc.Graph(id='f3t1-sunburst', style={'height': '90vh'})
    ]),
    
    # Visualization 3: f2t8.py integration
    html.Div(style=styles['section'], children=[
        html.H2("Satellite Mass Accumulation"),
        dcc.Graph(id='f2t8-mass-plot', style={'height': '90vh'}),
        html.Div(id='f2t8-event-info', style={'marginTop': '20px'})
    ])
])

# Register test4.py callbacks
@app.callback(
    Output('test4-main-plot', 'figure'),
    Input('test4-year-slider', 'value')
)
def update_test4_plot(selected_year):
    fig = go.Figure()
    
    # Original test4.py plotting logic
    satellites = satcat_base[satcat_base['Year'] == selected_year]
    hostility = hostility_data[hostility_data['Year'] == selected_year]

    # Add hostility background
    for i, country in enumerate(country_order):
        hostility_level = hostility[hostility['Country_Group'] == country]['Hostility'].mean()
        if not pd.isna(hostility_level):
            fig.add_trace(go.Bar(
                x=[i], y=[hostility_level], width=0.3,
                marker_color='rgba(200, 0, 0, 0.2)',
                showlegend=False
            ))

    # Add satellite markers
    purpose_symbols = {'Civil': 'circle', 'Defense': 'diamond', 'Communications': 'triangle-up'}
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
                        name=f"{purpose} - {country}",
                        showlegend=False
                    ))

    # Final layout adjustments
    fig.update_layout(
        title=f"Satellite Launches and Global Conflicts in {selected_year}",
        xaxis={'tickvals': list(range(len(country_order))),
               'ticktext': country_order},
        yaxis={'tickvals': [1,2,3], 'ticktext': ['LEO', 'MEO', 'GEO']},
        template="plotly_white"
    )
    return fig

# Register f3t1.py visualization
@app.callback(
    Output('f3t1-sunburst', 'figure'),
    [Input('test4-year-slider', 'value')]
)
def update_sunburst(selected_year):
    filtered = processed_f3t1[processed_f3t1['LDate'].dt.year <= selected_year]
    agg_df = filtered.groupby(['Country', 'Orbit'])['Lifespan'].mean().reset_index()
    
    return {
        'data': [{
            'type': 'sunburst',
            'parents': agg_df['Country'],
            'labels': agg_df['Orbit'],
            'values': agg_df['Lifespan'],
            'marker': {'colorscale': 'Viridis'}
        }],
        'layout': {
            'title': f'Satellite Lifespan Distribution up to {selected_year}',
            'margin': {'t': 40}
        }
    }

# Register f2t8.py visualization
@app.callback(
    [Output('f2t8-mass-plot', 'figure'),
     Output('f2t8-event-info', 'children')],
    [Input('f2t8-mass-plot', 'clickData')]
)
def update_mass_timeline(clickData):
    fig = go.Figure()
    
    # Plot mass accumulation lines
    for country in ['USA', 'China', 'Russia', 'Japan', 'India', 'Western Europe']:
        country_data = cumulative_mass[cumulative_mass['Country'] == country]
        fig.add_trace(go.Scatter(
            x=country_data['Year'],
            y=country_data['Mass'],
            name=country,
            mode='lines+markers'
        ))

    # Add historical events as invisible markers
    event_text = []
    for country, events in HISTORICAL_EVENTS.items():
        for year, title, desc in events:
            fig.add_trace(go.Scatter(
                x=[year],
                y=[cumulative_mass.loc[cumulative_mass['Year'] == year, 'Mass'].mean()],
                mode='markers',
                marker=dict(size=12, opacity=0),
                name=title,
                hoverinfo='none',
                customdata=[desc]
            ))

    # Handle click events
    info = ""
    if clickData:
        point = clickData['points'][0]
        if point['curveNumber'] >= 6:  # Event markers start at index 6
            info = html.Div([
                html.H4(point['trace']['name']),
                html.P(point['customdata'][0])
            ], style={'padding': '10px', 'border': '1px solid #ddd'})

    fig.update_layout(
        title='Satellite Mass Accumulation Timeline',
        hovermode='x unified'
    )
    
    return fig, info

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
