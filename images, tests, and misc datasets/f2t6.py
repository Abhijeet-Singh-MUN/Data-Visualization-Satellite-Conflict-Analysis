import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import mplcursors  # Added for hover interactions

# --- HISTORICAL EVENTS DATA ---
HISTORICAL_EVENTS = {
    'USA': [
        (1957, 'Sputnik Crisis', 's', 'Space Race begins after Soviet satellite launch'),
        (1969, 'Apollo 11', '^', 'First crewed Moon landing boosts tech investment'),
        (1984, 'Commercial Space Act', 'D', 'Opened space industry to private companies'),
        (1997, 'Mars Pathfinder', 'o', 'First successful Mars rover demonstrated new tech'),
        (2012, 'Commercial Crew', '*', 'NASA partnerships with SpaceX/Boeing began'),
        (2023, 'Artemis Accords', 'v', 'New international lunar exploration framework')
    ],
    'Russia': [
        (1957, 'Sputnik 1', 's', "World's first artificial satellite"),
        (1961, 'Gagarin Orbit', '^', 'First human in space'),
        (1971, 'Salyut 1', 'D', 'First space station launched'),
        (1998, 'ISS Contribution', 'o', 'Core module of International Space Station'),
        (2014, 'RD-181 Export', '*', 'Began rocket engine exports to US'),
        (2024, 'Luna 25 Crash', 'v', 'Failed lunar landing attempt')
    ],
    'China': [
        (1970, 'Dong Fang Hong', 's', 'First Chinese satellite'),
        (2003, 'Shenzhou 5', '^', 'First Chinese astronaut in space'),
        (2011, 'Tiangong-1', 'D', 'First space lab module'),
        (2019, "Chang'e 4", 'o', "First lunar far side landing"),
        (2022, "CSS Complete", '*', "Tiangong space station finished"),
        (2025, "Lunar Base Start", "v", "Begin construction of joint Russia/China base")
    ],
    "Global": [
        (1967, "Outer Space Treaty", "✩", "UN treaty governing space activities"),
        (1993, "EU Integration", "■", "Maastricht Treaty created European space agency"),
        (2008, "Financial Crisis", "◊", "Global economic downturn affected budgets"),
        (2016, "Brexit", "⚑", "EU space funding reorganization"),
        (2020, "COVID Pandemic", "⚕", "Global supply chain disruptions"),
        (2024, "Artemis Accords", "⚖", "55 nations signed lunar exploration pact")
    ]
}

# --- Updated Economic Periods ---
RECESSION_PERIODS = [

    (1973.11, 1975.3, "1973 Oil Crisis"),
    (1980.1, 1980.7, "Energy Crisis Recession"),
    (1990.7, 1991.3, "Gulf War Recession"),
    (2001.3, 2001.11, "Dot-com Bubble Burst"),
    (2007.12, 2009.6, "Global Financial Crisis"),
    (2020.2, 2020.4, "COVID-19 Pandemic"),
    (2023.1, 2024.2, "Post-Pandemic Inflation")
]

BOOM_PERIODS = [
    (1957, 1973.10, "Continued Post-WWII Economic Expansion"),
    (1982.12, 1990.6, "Reaganomics Boom"),
    (1991.4, 2001.2, "Globalization Boom"),
    (2009.7, 2020.1, "Quantitative Easing Era"),
    (2024.3, 2025.12,"AI/Green Tech Investment Boom")
]
# --- DATA LOADING WITH VALIDATION ---
def load_satcat(file_path='output.csv'):
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
        full_years = pd.DataFrame({'Year': range(1957, 2026)})
        
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

# --- PLOTTING WITH HOVER TOOLTIPS ---
def plot_interactive_mass(data):
    """Create interactive plot with hover tooltips"""
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Styling configuration
    country_styles = {
        'USA': ('#1f77b4', 2.5),
        'China': ('#00CC96', 2.2),
        'Russia': ('#FF6B6B', 1.8),
        'Japan': ('#6B3FA0', 1.5),
        'India': ('#FFA500', 1.5),
        'Western Europe': ('#8B4513', 1.5),
        'Other': ('#708090', 1.0)
    }
    
    # Plot main trajectories
    for country in data.columns[1:]:
        if country in country_styles:
            color, lw = country_styles[country]
            ax.plot(data['Year'], data[country], 
                   color=color, linewidth=lw, label=country, zorder=2)
    
    # Create event markers
    event_artists = []
    for country in HISTORICAL_EVENTS:
        if country in data.columns:
            color = country_styles[country][0]
            for event in HISTORICAL_EVENTS[country]:
                year, label, marker, desc = event
                if year in data['Year'].values:
                    idx = data[data['Year'] == year].index[0]
                    y_val = data.loc[idx, country]
                    artist = ax.plot(year, y_val, marker=marker, markersize=12,
                                   markeredgecolor='black', markerfacecolor=color,
                                   zorder=3)[0]
                    artist.set_gid(f"{country}|{year}|{label}|{desc}")
                    event_artists.append(artist)

    # Configure hover tooltips
    cursor = mplcursors.cursor(event_artists, hover=True)
    
    @cursor.connect("add")
    def on_hover(sel):
        gid = sel.artist.get_gid()
        if gid:
            country, year, label, desc = gid.split('|')
            sel.annotation.set_text(f"{label} ({year})\n{desc}")
            sel.annotation.get_bbox_patch().set_facecolor(country_styles[country][0])
            sel.annotation.get_bbox_patch().set_alpha(0.9)
            sel.annotation.set_fontsize(10)
            sel.annotation.arrow_patch.set_color('black')

    # Add economic periods
    for period_list, color, label in [
        (RECESSION_PERIODS, 'red', 'Recession'), 
        (BOOM_PERIODS, 'green', 'Economic Expansion')
    ]:
        for start, end, period_label in period_list:
            ax.axvspan(start, end, color=color, alpha=0.1, zorder=1)
            mid_x = (start + end) / 2
            ax.text(mid_x, ax.get_ylim()[1] * 0.95, period_label,
                   ha='center', rotation=90, fontsize=8, 
                   color=color, alpha=0.7)

    # Axis formatting
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Cumulative Mass (tonnes)', fontsize=14)
    ax.set_title('Satellite Mass Accumulation with Historical Context (1957-2025)', fontsize=16)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1960, 2025)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    
    plt.tight_layout()
    plt.savefig('satellite_mass_timeline.png', dpi=300, bbox_inches='tight')
    print("Interactive visualization created. Hover over markers for details.")
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    print("Space Program Analysis Tool")
    print("---------------------------")
    
    try:
        satcat = load_satcat()
        cumulative_data = calculate_cumulative_mass(satcat)
        plot_interactive_mass(cumulative_data)
        print("Analysis completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}\nPlease ensure 'output.csv' is present.")
    except KeyboardInterrupt:
        print("\nOperation canceled.")
    except Exception as e:
        print(f"Unexpected error: {e}")
