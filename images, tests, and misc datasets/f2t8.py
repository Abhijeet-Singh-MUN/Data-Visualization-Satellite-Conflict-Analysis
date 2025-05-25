import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
# Removed mplcursors import as we're using click events now

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
        (2020, 'Starlink Deployment', 'SpaceX launches 800+ satellites annually'),
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
    (2001.3, 2001.11, "Dot-com Bubble Burst"),
    (2007.12, 2009.6, "Global Financial Crisis"),
    (2020.2, 2020.4, "COVID-19 Pandemic"),
    (2023.1, 2024.2, "Post-Pandemic Inflation")
]

BOOM_PERIODS = [
    (1955, 1973.10, "Post-WWII Economic Expansion"),
    (1982.12, 1990.6, "Reaganomics Boom"),
    (1991.4, 2001.2, "Globalization Boom"),
    (2009.7, 2020.1, "Quantitative Easing Era"),
    (2024.3, 2025.12, "AI/Green Tech Boom")
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
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Create annotation for popup (initially invisible)
    annot = ax.annotate("", xy=(0,0), xytext=(-120,50), textcoords="offset points",
                      bbox=dict(boxstyle="round, pad = 0.5", fc="white", ec="black", lw = 1,  alpha=0.95),
                      arrowprops=dict(arrowstyle="->,head_width=0.4",
                                connectionstyle="arc3,rad=0.2",
                                color="black"))
    annot.set_visible(False)
    
    # Calculate global average
    data['Global'] = data.iloc[:, 1:].mean(axis=1)  # Skip 'Year' column
    
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
    
    # Plot main trajectories
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
                                   picker=5, zorder=5)[0]  # Added picker parameter for click detection
                    artist.set_gid(f"{country}|{year}|{label}|{desc}")
                    event_artists.append(artist)

    # Click event handler
    def on_pick(event):
        artist = event.artist
        if artist in event_artists and artist.get_gid():
            country, year, label, desc = artist.get_gid().split('|')
            x = artist.get_xdata()[0]
            y = artist.get_ydata()[0]

            # Determine which quadrant of the plot the point is in and adjust accordingly
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        x_mid = ax.get_xlim()[0] + x_range/2
        y_mid = ax.get_ylim()[0] + y_range/2
        
        # Position popup away from the point based on location
          
        # Determine position relative to plot center
        if x > x_mid:
            x_offset = -40  # Left side
        else:
            x_offset = 40   # Right side
            
        if y > y_mid:
            y_offset = -40  # Below point
        else:
            y_offset = 40   # Above point
            
        annot.xytext = (x_offset, y_offset)
        annot.xy = (x, y)
        annot.set_text(f"{label} ({year})\n{desc}")
        annot.get_bbox_patch().set_facecolor(country_styles[country][0])
        annot.set_visible(True)
        fig.canvas.draw_idle()

    # Background click handler to hide annotation
    def on_click(event):
        if event.inaxes != ax:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    # Connect event handlers
    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Manual positioning for economic period labels
    y_positions = {
        # Recessions
        'Suez Crisis Recession': 0.90,
        '1973 Oil Crisis': 0.85,
        'Energy Crisis Recession': 0.90,
        'Gulf War Recession': 0.85,
        'Dot-com Bubble Burst': 0.90,
        'Global Financial Crisis': 0.85,
        'COVID-19 Pandemic': 0.90,
        'Post-Pandemic Inflation': 0.85,
        
        # Booms
        'Post-WWII Economic Expansion': 0.95,
        'Reaganomics Boom': 0.95,
        'Globalization Boom': 0.95,
        'Quantitative Easing Era': 0.95,
        'AI/Green Tech Boom': 0.95
    }

    # Manual label positions for specific labels
    manual_label_positions = {
        'Suez Crisis Recession': 1958.5,
        'AI/Green Tech Boom': 2022,
        'Post-Pandemic Inflation': 2021.5
    }

    # Add economic periods with manual label positioning
    for period_list, color, label in [
        (RECESSION_PERIODS, 'red', 'Recession'), 
        (BOOM_PERIODS, 'green', 'Economic Expansion')
    ]:
        for start, end, period_label in period_list:
            ax.axvspan(start, end, color=color, alpha=0.15, zorder=1)
            
            # Use predefined position or fall back to default
            y_pos = y_positions.get(period_label, 0.5)
            # Use manual position if defined, else calculate midpoint
            x_pos = manual_label_positions.get(period_label, (start + end)/2)
            
            ax.text(x_pos, ax.get_ylim()[1] * y_pos, period_label,
                   ha='center', rotation=0, fontsize=8, 
                   color=color, alpha=0.8)

    # Axis formatting
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Cumulative Mass (tonnes)', fontsize=14)
    ax.set_title('Satellite Mass Accumulation with Historical Context (1955-2025)', fontsize=16)
    
    # Place legend outside the plot to the right
    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1955, 2025)
    
    # Set x-axis ticks every 5 years
    x_ticks = np.arange(1955, 2026, 5)
    ax.set_xticks(x_ticks)
    
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    
    # Make room for the legend on the right
    plt.subplots_adjust(right=0.85)
    
    plt.tight_layout()
    plt.savefig('satellite_mass_timeline.png', dpi=300, bbox_inches='tight')
    print("Interactive visualization created. Click markers to see event details.")
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
