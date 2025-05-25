import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

output1 = widgets.Output()
with output1:
    # --- DATA LOADING FUNCTIONS ---
    def load_satcat(file_path='satcat.csv'):
        """Load and preprocess satellite catalog data."""
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
        
        # Country mapping
        country_map = {
            'US': 'US', 'CN': 'China', 'RU': 'Russia', 'SU': 'Russia',
            'J': 'Japan', 'IN': 'India', 'F': 'Western Europe', 
            'D': 'Western Europe', 'GB': 'Western Europe', 
            'I': 'Western Europe', 'E': 'Western Europe', 
            'NL': 'Western Europe', 'S': 'Western Europe', 
            'CH': 'Western Europe'
        }
        org_mapping = {
            'NASA': 'US', 'ISRO': 'India', 'JAXA': 'Japan',
            'ESA': 'Western Europe', 'ROSCOSMOS': 'Russia',
            'CNSA': 'China', 'CNES': 'Western Europe',
            'DLR': 'Western Europe', 'UKSA': 'Western Europe'
        }
        satcat['Primary_Code'] = (
            satcat['State']
            .str.split(r'[-/]', n=1).str[0]
            .str.strip()
            .replace('', np.nan)
        )
        satcat['Country_Group'] = (
            satcat['Primary_Code']
            .map(org_mapping)
            .fillna(satcat['Primary_Code'].map(country_map))
            .fillna('Other')
        )
        return satcat

    def load_midip(file_path='MIDIP 5.0.csv'):
        """Load conflict data."""
        midip = pd.read_csv(file_path, na_values=['-', 'nan', ''])
        country_codes = {
            2: 'US', 200: 'Western Europe', 220: 'Western Europe',
            255: 'Western Europe', 325: 'Western Europe', 365: 'Russia',
            710: 'China', 740: 'Japan', 750: 'India'
        }
        midip['Country_Group'] = midip['ccode'].map(country_codes).fillna('Other')
        midip['Hostility'] = pd.to_numeric(midip['hostlev'], errors='coerce')
        return midip.dropna(subset=['Hostility']) \
                    .groupby(['styear', 'Country_Group'])['Hostility'] \
                    .mean().reset_index().rename(columns={'styear': 'Year'})

    # --- DATA INITIALIZATION ---
    satcat = load_satcat()
    hostility_data = load_midip()

    country_order = ['US', 'China', 'Russia', 'Japan', 'India', 'Western Europe', 'Other']
    country_colors = {
        'US': '#636EFA', 'China': '#00CC96', 'Russia': '#AB63FA',
        'Japan': '#FFA15A', 'India': '#19D3F3', 
        'Western Europe': '#EF553B', 'Other': '#FF6692'
    }
    purpose_symbols = {
        'Civil': 'circle',
        'Defense': 'diamond',
        'Communications': 'triangle-up'
    }
    orbit_positions = {'LEO': 1, 'MEO': 2, 'GEO': 3}

    # --- FIGURE UPDATE FUNCTION ---
    def update_figure(selected_year):
        fig = make_subplots(specs=[[{"secondary_y": True}]], shared_xaxes=True)
        satellites = satcat[satcat['Year'] == selected_year]
        hostility = hostility_data[hostility_data['Year'] == selected_year]
        
        # Add hostility bars (background layer)
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
        for purpose, symbol in purpose_symbols.items():
            for orbit in ['LEO', 'MEO', 'GEO']:
                subset = satellites[(satellites['Purpose'] == purpose) & (satellites['Orbit_Class'] == orbit)]
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
                                f"<b>{country}</b><br>Purpose: {purpose}<br>Orbit: {orbit}<br>Count: {len(country_subset)}"
                            )
                        ), secondary_y=False)
        
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
            xaxis={'tickvals': list(range(len(country_order))), 'ticktext': country_order, 'title': 'Country/Region'},
            yaxis={
                'tickvals': [1,2,3],
                'ticktext': ['LEO', 'MEO', 'GEO'],
                'title': 'Orbital Class',
                'range': [0.5, 3.5],
                'layer': 'above traces',
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

    # --- IPYWIDGETS SLIDER SETUP ---
    year_slider = widgets.IntSlider(
        value=2000,
        min=1992,
        max=2014,
        step=1,
        description='Year:',
        continuous_update=False
    )

    output = widgets.Output()

    def on_value_change(change):
        with output:
            output.clear_output(wait=True)
            fig = update_figure(change['new'])
            fig.show()

    year_slider.observe(on_value_change, names='value')

    # --- INITIAL DISPLAY ---
    display(year_slider)
    with output:
        fig = update_figure(year_slider.value)
        fig.show()
    display(output)



# Enable interactive matplotlib backend
%matplotlib ipympl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

output2 = widgets.Output()
with output2:
    # --- HISTORICAL EVENTS DATA ---
    HISTORICAL_EVENTS = {
        'USA': [
            (1957, 'Sputnik Crisis', 'Space Race begins after Soviet satellite launch'),
            (1969, 'Apollo 11',  'First crewed Moon landing boosts tech investment'),
            (1984, 'Commercial Space Act', 'Opened space industry to private companies'),
            (1986, 'Challenger Disaster', 'NASA shuttle program halted for 32 months'),
            (1991, 'Soviet Collapse', 'Reduced defense spending, "Peace Dividend"'),
            (2000, 'Dot-com Peak', 'Commercial space investments surge'),
            (2008, 'Financial Crisis', 'Budget cuts delay constellation programs'),
            (2012, 'Commercial Crew', 'NASA partnerships with SpaceX/Boeing began'),
            (2014, 'Crimea Sanctions', 'RD-180 engine import restrictions'),
            (2023, 'Artemis Accords', 'New international lunar exploration framework')
        ],
        'Russia': [
            (1957, 'Sputnik 1',  "World's first artificial satellite"),
            (1961, 'Gagarin Orbit',  'First human in space'),
            (1971, 'Salyut 1',  'First space station launched'),
            (1991, 'USSR Dissolution', 'Space budget cut by 80% overnight'),
            (1996, 'Industry Consolidation', 'Formation of Roscosmos state corp'),
            (1998, 'ISS Contribution',  'Core module of International Space Station'),
            (2014, 'Ukraine Conflict', 'Loss of Ukrainian launch components'),
            (2022, 'Sanctions Impact', '70% reduction in commercial launches')
        ],
        'China': [
            (1970, 'Dong Fang Hong', 'First Chinese satellite'),
            (2003, 'First Taikonaut', 'Shenzhou 5 establishes crewed capability'),
            (2011, 'GPS Alternative', 'BeiDou-2 initial operational capability'),
            (2019, 'Lunar Sample', "Chang'e 5 moon mission success"),
            (2021, 'Tiangong Station', 'Permanent space station deployment'),
            (2022, "CSS Complete", "Tiangong space station finished"),
            (2025, "Lunar Base Start",  "Begin construction of joint Russia/China base")
        ],
        'Global': [
            (1967, "Outer Space Treaty", "UN treaty governing space activities"),
            (1993, "EU Integration", "Maastricht Treaty created European space agency"),
            (2008, "Financial Crisis", "Global economic downturn affected budgets"),
            (2016, "Brexit", "EU space funding reorganization"),
            (2020, "COVID Pandemic", "Global supply chain disruptions"),
            (2024, "Artemis Accords", "55 nations signed lunar exploration pact")
        ]
    }

    # Country-specific marker shapes
    COUNTRY_MARKERS = {
        'USA': 'o',        # Circle
        'Russia': 's',     # Square  
        'China': '^',      # Triangle
        'Japan': 'D',      # Diamond
        'India': 'p',      # Pentagon
        'Western Europe': 'h',  # Hexagon
        'Global': '*'      # Star
    }

    # --- ECONOMIC PERIODS ---
    RECESSION_PERIODS = [
        (1957.8, 1958.4, "Suez Crisis Recession"),
        (1973.11, 1975.3, "1973 Oil Crisis"),
        (1980.1, 1980.7, "Energy Crisis Recession"),
        (1990.7, 1991.3, "Gulf War Recession"),
        (2000, 2002.5, "Dot-com Bubble Burst"),
        (2007.12, 2009.6, "Global Financial Crisis"),
        (2020, 2023.5, "COVID-19 Pandemic"),
        (2021.5, 2022.5, "Post-Pandemic Inflation")
    ]

    BOOM_PERIODS = [
        (1955, 1973.10, "Post-WWII Economic Expansion"),
        (1982.12, 1990.6, "Reaganomics Boom"),
        (1991.4, 2001.2, "Globalization Boom"),
        (2009.7, 2020.1, "Quantitative Easing Era"),
        (2022.5, 2025.12, "AI/Green Tech Boom")
    ]

    # --- DATA LOADING WITH VALIDATION ---
    def load_satcat(file_path='satcat.csv'):
        """Load and preprocess satellite catalog data with mass validation"""
        try:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File '{file_path}' not found in directory {Path().resolve()}")

            satcat = pd.read_csv(file_path, low_memory=False)
            
            # Date and mass processing
            satcat['LDate'] = pd.to_datetime(satcat['LDate'], errors='coerce')
            satcat['Year'] = satcat['LDate'].dt.year
            satcat['Mass'] = pd.to_numeric(satcat['Mass'], errors='coerce').abs().fillna(0)
            
            # Country mapping
            country_map = {
                'US': 'USA', 'CN': 'China', 'RU': 'Russia', 'SU': 'Russia',
                'J': 'Japan', 'IN': 'India', 'F': 'Western Europe', 
                'D': 'Western Europe', 'GB': 'Western Europe', 
                'I': 'Western Europe', 'E': 'Western Europe', 
                'NL': 'Western Europe', 'S': 'Western Europe', 
                'CH': 'Western Europe'
            }
            
            satcat['Country_Group'] = (
                satcat['State']
                .str.split(r'[-/]').str[0]
                .map(country_map)
                .fillna('Other')
            )

            # Filter valid records
            valid_data = satcat[
                (satcat['Year'].between(1957, 2025)) &
                (satcat['Mass'] > 0)
            ].copy()

            print(f"Loaded {len(valid_data)} valid records")
            return valid_data

        except Exception as e:
            print(f"Data Loading Failed: {str(e)}")
            raise

    # --- CUMULATIVE MASS CALCULATION ---
    def calculate_cumulative_mass(satcat):
        """Calculate validated cumulative mass with gap filling"""
        try:
            full_years = pd.DataFrame({'Year': range(1956, 2026)})  # Start from 1956
            
            mass_data = (
                satcat.groupby(['Year', 'Country_Group'])['Mass']
                .sum()
                .unstack(fill_value=0)
                .reindex(full_years['Year'], fill_value=0)
            )
            
            cumulative = (mass_data / 1000).cumsum().apply(lambda x: x.cummax())
            
            print("Mass calculation completed")
            return cumulative.reset_index()

        except Exception as e:
            print(f"Mass Calculation Error: {str(e)}")
            raise

    # --- PLOTTING WITH CLICKABLE MARKERS ---
    def plot_interactive_mass(data):
        """Create interactive plot with clickable historical markers"""
        fig, ax = plt.subplots(figsize=(12,6))
        
        # Create annotation for popup (initially invisible)
        annot = ax.annotate("", xy=(0,0), xytext=(-100,-60), textcoords="offset points",
                        bbox=dict(boxstyle="round, pad = 0.5", fc="white", ec="black", lw=1, alpha=0.95),
                        arrowprops=dict(arrowstyle="->,head_width=0.4",
                                    connectionstyle="arc3,rad=0.2",
                                    color="black"))
        annot.set_visible(False)
        
        # Calculate global average across all country columns (skip 'Year')
        data['Global'] = data.iloc[:, 1:].mean(axis=1)
        
        # Styling configuration
        country_styles = {
            'USA': ('#1f77b4', 2.5),
            'China': ('#00CC96', 2.2),
            'Russia': ('#FF6B6B', 1.8),
            'Japan': ('#6B3FA0', 1.5),
            'India': ('#FFA500', 1.5),
            'Western Europe': ('#8B4513', 1.5),
            'Global': ('#6A0DAD', 2.2),  # Purple color for global average
            'Other': ('#708090', 1.0)
        }
        
        # Plot main trajectories for each country
        for country in data.columns[1:]:
            if country in country_styles:
                color, lw = country_styles[country]
                linestyle = '--' if country == 'Global' else '-'
                ax.plot(data['Year'], data[country], 
                    color=color, linewidth=lw, label=country, zorder=2, linestyle=linestyle)
        
        # Create event markers with consistent shapes per country
        event_artists = []
        for country in HISTORICAL_EVENTS:
            if country in data.columns:
                color = country_styles[country][0]
                marker = COUNTRY_MARKERS.get(country, 'o')
                for event in HISTORICAL_EVENTS[country]:
                    year, label, desc = event
                    if year in data['Year'].values:
                        idx = data[data['Year'] == year].index[0]
                        y_val = data.loc[idx, country]
                        artist = ax.plot(year, y_val, marker=marker, markersize=12,
                                    markeredgecolor='black', markerfacecolor=color,
                                    picker=5, zorder=5)[0]
                        artist.set_gid(f"{country}|{year}|{label}|{desc}")
                        event_artists.append(artist)
        
        # Click event handler for markers
        def on_pick(event):
            artist = event.artist
            if artist in event_artists and artist.get_gid():
                country, year, label, desc = artist.get_gid().split('|')
                x = artist.get_xdata()[0]
                y = artist.get_ydata()[0]
                # Determine relative position to adjust annotation offset
                x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                x_mid = ax.get_xlim()[0] + x_range/2
                y_mid = ax.get_ylim()[0] + y_range/2
                x_offset = -40 if x > x_mid else 40
                y_offset = -40 if y > y_mid else 40
                annot.xy = (x, y)
                annot.xytext = (x_offset, y_offset)
                annot.set_text(f"{label} ({year})\n{desc}")
                annot.get_bbox_patch().set_facecolor(country_styles[country][0])
                annot.set_visible(True)
                fig.canvas.draw_idle()
        
        def on_click(event):
            # Hide annotation when clicking outside markers
            if event.inaxes != ax:
                annot.set_visible(False)
                fig.canvas.draw_idle()
        
        # Connect event handlers
        fig.canvas.mpl_connect('pick_event', on_pick)
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        # Add economic periods (shading and labels)
        y_positions = {
            'Suez Crisis Recession': 0.90,
            '1973 Oil Crisis': 0.85,
            'Energy Crisis Recession': 0.90,
            'Gulf War Recession': 0.85,
            'Dot-com Bubble Burst': 0.90,
            'Global Financial Crisis': 0.85,
            'COVID-19 Pandemic': 0.90,
            'Post-Pandemic Inflation': 0.84,
            'Post-WWII Economic Expansion': 0.95,
            'Reaganomics Boom': 0.95,
            'Globalization Boom': 0.95,
            'Quantitative Easing Era': 0.95,
            'AI/Green Tech Boom': 0.7
        }
        manual_label_positions = {
            'Suez Crisis Recession': 1960,
            'AI/Green Tech Boom': 2021,
            'Post-Pandemic Inflation': 2021,
            'COVID-19 Pandemic' : 2020
        }
        
        for period_list, color, _ in [
            (RECESSION_PERIODS, 'red', 'Recession'),
            (BOOM_PERIODS, 'green', 'Economic Expansion')
        ]:
            for start, end, period_label in period_list:
                ax.axvspan(start, end, color=color, alpha=0.15, zorder=1)
                y_pos = y_positions.get(period_label, 0.5)
                x_pos = manual_label_positions.get(period_label, (start + end) / 2)
                ax.text(x_pos, ax.get_ylim()[1] * y_pos, period_label,
                        ha='center', rotation=0, fontsize=8,
                        color=color, alpha=0.8)
        
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Cumulative Mass (tonnes)', fontsize=14)
        ax.set_title('Satellite Mass Accumulation with Historical Context (1955-2025)', fontsize=16)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1955, 2025)
        ax.set_xticks(np.arange(1955, 2026, 5))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
        plt.subplots_adjust(right=0.85)
        plt.tight_layout()
        
        print("Interactive visualization created. Click markers to see event details.")
        plt.show()

    # --- MAIN EXECUTION ---
    try:
        satcat_data = load_satcat()  # Ensure 'output.csv' is in the working directory
        cumulative_data = calculate_cumulative_mass(satcat_data)
        plot_interactive_mass(cumulative_data)
        print("Analysis completed successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}\nPlease ensure 'output.csv' is present.")
    except KeyboardInterrupt:
        print("\nOperation canceled.")
    except Exception as e:
        print(f"Unexpected error: {e}")

import pandas as pd
import plotly.express as px
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display
import warnings

output3 = widgets.Output()
with output3:
    # --- Load Dataset ---

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
        df = pd.read_csv('satcat.csv', low_memory=False)


    # --- Country Mapping ---
    country_map = {
        'US': 'US', 'CN': 'China', 'RU': 'Russia', 'SU': 'Russia',
        'J': 'Japan', 'IN': 'India', 'F': 'Western Europe', 
        'D': 'Western Europe', 'GB': 'Western Europe', 
        'I': 'Western Europe', 'E': 'Western Europe', 
        'NL': 'Western Europe', 'S': 'Western Europe', 
        'CH': 'Western Europe'
    }

    # --- Data Processing Function ---
    def process_data(df, filter_years=None):
        df = df[['State', 'SDate', 'DDate', 'OpOrbit']].dropna(subset=['SDate', 'DDate'])
        df['Country'] = df['State'].map(country_map).fillna('Other')
        
        date_format = '%Y %b %d'
        today = datetime(2025, 4, 11)  # Fixed reference date
        
        def get_lifespan(row):
            try:
                launch_str = ' '.join(row['SDate'].split()[:3])
                decay_str = ' '.join(row['DDate'].split()[:3]) if pd.notna(row['DDate']) else today
                launch = datetime.strptime(launch_str, '%Y %b %d')
                decay = datetime.strptime(decay_str, '%Y %b %d') if isinstance(decay_str, str) else decay_str
                return (decay - launch).days / 365.25
            except:
                return None

        df['Lifespan'] = df.apply(get_lifespan, axis=1)
        df = df.dropna(subset=['Lifespan'])

        def classify_orbit(op):
            op = str(op).upper()
            if 'LEO' in op: return 'LEO'
            if 'MEO' in op: return 'MEO'
            if 'GEO' in op or 'HEO' in op: return 'GEO'
            return 'Other'
        
        df['Orbit'] = df['OpOrbit'].apply(classify_orbit)

        # Optional filter by launch year
        if filter_years:
            df['Launch_Year'] = df['SDate'].str.extract(r'(\d{4})').astype(float)
            df = df[df['Launch_Year'].between(*filter_years)]

        return df

    # --- Plot Function ---
    def plot_sunburst(df):
        agg_df = df.groupby(['Country', 'Orbit'])['Lifespan'].mean().reset_index()

        fig = px.sunburst(
            agg_df,
            path=['Country', 'Orbit'],
            values='Lifespan',
            color='Lifespan',
            color_continuous_scale='Viridis',
            title='Average Satellite Lifespan by Country and Orbit',
            height=700,
            branchvalues='total'
        )

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

    # --- Optional Year Slider ---
    year_slider = widgets.IntRangeSlider(
        value=[1957, 2025],
        min=1957,
        max=2025,
        step=1,
        description='Years:',
        continuous_update=False,
        layout=widgets.Layout(width='60%')
    )

    out = widgets.Output()

    def on_year_change(change):
        with out:
            out.clear_output(wait=True)
            filtered_df = process_data(df, filter_years=change['new'])
            plot_sunburst(filtered_df)

    year_slider.observe(on_year_change, names='value')

    # --- Display Widgets ---
    display(year_slider)
    with out:
        initial_df = process_data(df, filter_years=year_slider.value)
        plot_sunburst(initial_df)
    display(out)

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# --- Top section background styling ---
display(HTML("""
<style>
.dashboard-header {
    background-image: url('night sky.jpg');
    background-position: 20% 45%;  /* X% Y% coordinates */
    background-size: cover;
    background-repeat: no-repeat;
    height: 200px;
    width: 100%;
    margin: -10px -10px 20px -10px;
    padding: 20px;
    text-align: center;
}
</style>
<div class="dashboard-header">
    <h2 style='color:#F9C74F !important; font-weight: bold; margin-top: 30px; text-shadow: 
            -1px -1px 0 #0A2342,
            1px -1px 0 #0A2342,
            -1px 1px 0 #0A2342,
            1px 1px 0 #0A2342,
            0 0 8px rgba(10,35,66,0.5); 
        letter-spacing: 0.5px;'>🪐 Satellites, Space, & Conflicts Dashboard</h2>
    <p style='font-size: 14px;
        color: #E0E0E0; 
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
        margin-top: -10px;
        padding-bottom: 15px;'>Explore orbital history through interactive visuals</p>
</div>
"""))

# --- Styled output container ---
card_style = widgets.Layout(
    border='1px solid #555',
    padding='15px',
    margin='10px 0px',
    background_color='rgba(255,255,255,0.05)',
    box_shadow='2px 2px 10px rgba(0,0,0,0.5)'
)

# Apply to output boxes
output1.layout = card_style
output2.layout = card_style
output3.layout = card_style

# --- Captions (now placed ABOVE visuals) ---
caption1 = widgets.HTML(
    "<p style='font-size:15px; color:#666666; font-weight: bold;'>🛰️ Use the time slider to explore satellite launches by year. Click on legend items (Right Side) to toggle satellite types. Hostility bars are in the background.</p>"
)
caption2 = widgets.HTML(
    "<p style='font-size:15px; color:#666666; font-weight: bold;'>📌 Click on event markers in the timeline to view historical context.</p>"
)
caption3 = widgets.HTML(
    "<p style='font-size:15px; color:#666666; font-weight: bold;'>🌍 Click a country in the inner circle to expand its orbital breakdown. Hover for lifespan details. Use Time Range slider for the desired year range.</p>"
)

# --- Combine captions ABOVE visuals ---
view1 = widgets.VBox([caption1, output1])
view2 = widgets.VBox([caption2, output2])
view3 = widgets.VBox([caption3, output3])

# --- Dropdown ---
options_dict = {
    'Visualization 1 – Satellites vs. Hostility (Slider)': view1,
    'Visualization 2 – Cumulative Mass + Historical Events': view2,
    'Visualization 3 – Lifespan by Country & Orbit': view3
}

dropdown = widgets.Dropdown(
    options=list(options_dict.keys()),
    description='Select:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='70%', height='40px', padding='5px')
)

dropdown_box = widgets.HBox([dropdown], layout=widgets.Layout(justify_content='center'))

# --- Dashboard display area ---
dashboard_display = widgets.Output()

def on_dropdown_change(change):
    with dashboard_display:
        dashboard_display.clear_output()
        selected_view = options_dict[change['new']]
        display(selected_view)

dropdown.observe(on_dropdown_change, names='value')

# Show first view by default
with dashboard_display:
    display(options_dict[dropdown.value])

# --- Final dashboard layout ---
dashboard = widgets.VBox([dropdown_box, dashboard_display])
display(dashboard)


