import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# --- HISTORICAL EVENTS DATA ---
HISTORICAL_EVENTS = {
    'USA': [
        (1986, 'Challenger Disaster', '^', 'NASA shuttle program halted for 32 months'),
        (1991, 'Soviet Collapse', 's', 'Reduced defense spending, "Peace Dividend"'),
        (2000, 'Dot-com Peak', 'o', 'Commercial space investments surge'),
        (2008, 'Financial Crisis', 'v', 'Budget cuts delay constellation programs'),
        (2014, 'Crimea Sanctions', 'D', 'RD-180 engine import restrictions'),
        (2020, 'Starlink Deployment', '*', 'SpaceX launches 800+ satellites annually')
    ],
    'Russia': [
        (1991, 'USSR Dissolution', 's', 'Space budget cut by 80% overnight'),
        (1996, 'Industry Consolidation', 'o', 'Formation of Roscosmos state corp'),
        (2014, 'Ukraine Conflict', 'D', 'Loss of Ukrainian launch components'),
        (2022, 'Sanctions Impact', 'v', '70% reduction in commercial launches')
    ],
    'China': [
        (2003, 'First Taikonaut', '^', 'Shenzhou 5 establishes crewed capability'),
        (2011, 'GPS Alternative', 's', 'BeiDou-2 initial operational capability'),
        (2019, 'Lunar Sample', 'o', "Chang'e 5 moon mission success"),
        (2021, 'Tiangong Station', '*', 'Permanent space station deployment')
    ]
}

# --- ECONOMIC PERIODS ---
RECESSION_PERIODS = [
    (1981.7, 1982.11, 'Early 1980s Recession'),
    (1990.7, 1991.3, 'Gulf War Recession'), 
    (2001.3, 2001.11, 'Dot-com Crash'),
    (2007.12, 2009.6, 'Global Financial Crisis'),
    (2020.2, 2020.4, 'COVID-19 Recession'),
    (2023.1, 2024.2, 'Post-Pandemic Inflation')
]

BOOM_PERIODS = [
    (1991.4, 2001.2, 'Long Expansion'),
    (2009.7, 2020.1, 'Quantitative Easing Era')
]

# --- DATA LOADING WITH VALIDATION ---
def load_satcat(file_path='output.csv'):
    """Load and preprocess satellite catalog data with mass validation"""
    try:
        # Verify file existence
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File '{file_path}' not found in directory {Path().resolve()}")

        # Load CSV with validation
        satcat = pd.read_csv(file_path, low_memory=False)
        
        # Process dates and mass
        satcat['LDate'] = pd.to_datetime(satcat['LDate'], errors='coerce')
        satcat['Year'] = satcat['LDate'].dt.year
        satcat['Mass'] = pd.to_numeric(satcat['Mass'], errors='coerce').abs().fillna(0)
        
        # Country mapping with validation
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

        print(f"Successfully loaded {len(valid_data)} valid records")
        return valid_data

    except Exception as e:
        print(f"Data Loading Failed: {str(e)}")
        raise

# --- CUMULATIVE MASS CALCULATION ---
def calculate_cumulative_mass(satcat):
    """Calculate validated cumulative mass with gap filling"""
    try:
        # Create complete year range
        full_years = pd.DataFrame({'Year': range(1957, 2026)})
        
        # Aggregate mass with gap filling
        mass_data = (
            satcat.groupby(['Year', 'Country_Group'])['Mass']
            .sum()
            .unstack(fill_value=0)
            .reindex(full_years['Year'], fill_value=0)
        )
        
        # Convert to tonnes and ensure monotonic increase
        cumulative = (mass_data / 1000).cumsum().apply(lambda x: x.cummax())
        
        print("Cumulative mass calculation completed successfully")
        return cumulative.reset_index()

    except Exception as e:
        print(f"Mass Calculation Error: {str(e)}")
        raise

# --- PLOTTING WITH INTERACTIVE MARKERS ---
def plot_interactive_mass(data):
    """Create interactive plot with clickable historical markers"""
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Color scheme and styling
    country_styles = {
        'USA': ('#1f77b4', 2.5),
        'China': ('#00CC96', 2.2),
        'Russia': ('#FF6B6B', 1.8),
        'Japan': ('#6B3FA0', 1.5),
        'India': ('#FFA500', 1.5),
        'Western Europe': ('#8B4513', 1.5),
        'Other': ('#708090', 1.0)
    }
    
    # Plot cumulative trajectories
    for country in data.columns[1:]:  # Skip 'Year' column
        if country in country_styles:
            color, lw = country_styles[country]
            ax.plot(data['Year'], data[country], 
                   color=color, linewidth=lw, label=country, zorder=2)
    
    # Add historical event markers
    for country in HISTORICAL_EVENTS:
        if country in data.columns:
            color = country_styles[country][0]
            for event in HISTORICAL_EVENTS[country]:
                year, label, marker, desc = event
                if year in data['Year'].values:
                    idx = data[data['Year'] == year].index[0]
                    y_val = data.loc[idx, country]
                    ax.plot(year, y_val, marker=marker, markersize=12,
                           markeredgecolor='black', markerfacecolor=color,
                           picker=5, zorder=3)
                    # Store event info in artist's custom data
                    ax.get_children()[-1].set_gid(f"{country}|{year}|{label}|{desc}")

    # Add recession periods
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

    # Setup annotation box
    annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                       bbox=dict(boxstyle="round", fc="w", ec="k", alpha=0.9),
                       arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    
    # Define pick event handler
    def on_pick(event):
        artist = event.artist
        if hasattr(artist, 'get_gid') and artist.get_gid():
            gid = artist.get_gid().split('|')
            if len(gid) == 4:
                country, year, label, desc = gid
                x = artist.get_xdata()[0]
                y = artist.get_ydata()[0]
                
                annot.xy = (x, y)
                annot.set_text(f"{label} ({year})\n{desc}")
                annot.get_bbox_patch().set_facecolor(country_styles[country][0])
                annot.set_visible(True)
                fig.canvas.draw_idle()
    
    # Connect the pick event
    fig.canvas.mpl_connect('pick_event', on_pick)
    
    # Hide annotation when clicking elsewhere
    def on_click(event):
        if event.inaxes != ax:
            annot.set_visible(False)
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Formatting
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Cumulative Mass (tonnes)', fontsize=14)
    ax.set_title('Satellite Mass Accumulation with Historical Context (1957-2025)', fontsize=16)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1960, 2025)
    
    # Format y-axis with comma separators
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    
    plt.tight_layout()
    plt.savefig('satellite_mass_timeline.png', dpi=300, bbox_inches='tight')
    print("Interactive visualization created. Click markers to see event details.")
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    print("Space Program Analysis Tool")
    print("---------------------------")
    
    try:
        # Load and process data
        satcat = load_satcat()
        cumulative_data = calculate_cumulative_mass(satcat)
        
        # Generate visualization
        plot_interactive_mass(cumulative_data)
        print("Analysis completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'output.csv' is in the current directory.")
    except KeyboardInterrupt:
        print("\nOperation canceled by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
