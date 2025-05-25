import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation
from matplotlib.backend_bases import PickEvent

# --- HISTORICAL EVENTS DATA ---
historical_events = {
    'USA': [
        (1986, 'Challenger Disaster', '▲', 'NASA shuttle program halted for 32 months'),
        (1991, 'Soviet Collapse', '■', 'Reduced defense spending, "Peace Dividend"'),
        (2000, 'Dot-com Peak', '●', 'Commercial space investments surge'),
        (2008, 'Financial Crisis', '▼', 'Budget cuts delay constellation programs'),
        (2014, 'Crimea Sanctions', '◆', 'RD-180 engine import restrictions'),
        (2020, 'Starlink Deployment', '★', 'SpaceX launches 800+ satellites annually')
    ],
    'Russia': [
        (1991, 'USSR Dissolution', '■', 'Space budget cut by 80% overnight'),
        (1996, 'Industry Consolidation', '●', 'Formation of Roscosmos state corp'),
        (2014, 'Ukraine Conflict', '◆', 'Loss of Ukrainian launch components'),
        (2022, 'Sanctions Impact', '▼', '70% reduction in commercial launches')
    ],
    'China': [
        (2003, 'First Taikonaut', '▲', 'Shenzhou 5 establishes crewed capability'),
        (2011, 'GPS Alternative', '■', 'BeiDou-2 initial operational capability'),
        (2019, 'Lunar Sample', '●', 'ChangE 5 moon mission success'),
        (2021, 'Tiangong Station', '★', 'Permanent space station deployment')
    ]
}

# --- ECONOMIC PERIODS (US-Centric) ---
recession_periods = [
    (1981.7, 1982.11, 'Early 1980s Recession'),
    (1990.7, 1991.3, 'Gulf War Recession'), 
    (2001.3, 2001.11, 'Dot-com Crash'),
    (2007.12, 2009.6, 'Global Financial Crisis'),
    (2020.2, 2020.4, 'COVID-19 Recession'),
    (2023.1, 2024.2, 'Post-Pandemic Inflation')
]

boom_periods = [
    (1991.4, 2001.2, 'Long Expansion'),
    (2009.7, 2020.1, 'Quantitative Easing Era')
]

# --- ENHANCED PLOT FUNCTION ---
def plot_cumulative_mass(data):
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Style configuration
    style_map = {
        'USA': ('#1f77b4', '-', 2.0),
        'Russia': ('#d62728', '--', 1.5),
        'China': ('#2ca02c', '-.', 2.2),
        'Other': ('#7f7f7f', ':', 1.0)
    }
    
    # Plot cumulative mass lines
    for country in data.columns[1:]:
        color, linestyle, lw = style_map.get(country, ('#333', '-', 1))
        line, = ax.plot(data['Year'], data[country], 
                       color=color, linestyle=linestyle, linewidth=lw,
                       marker='', markersize=8, label=country, picker=5)
        
        # Add historical markers
        if country in historical_events:
            for event in historical_events[country]:
                year, label, marker, desc = event
                idx = data['Year'].searchsorted(year)
                if idx < len(data):
                    ax.plot(data['Year'].iloc[idx], data[country].iloc[idx],
                           marker=marker, color=color, markersize=12,
                           markeredgewidth=1, markeredgecolor='k',
                           picker=5, gid=f"{country}|{year}|{label}|{desc}")

    # Add economic period shading
    for start, end, label in recession_periods:
        ax.add_patch(Rectangle((start, 0), end-start, 1e6,
                     facecolor='red', alpha=0.1, zorder=-1))
        
    for start, end, label in boom_periods:
        ax.add_patch(Rectangle((start, 0), end-start, 1e6,
                     facecolor='green', alpha=0.1, zorder=-1))

    # Annotation setup
    annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w", alpha=0.9),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    # Interactive event handling
    def on_pick(event):
        if isinstance(event.artist, plt.Line2D):
            return  # Ignore line clicks
        
        xdata, ydata = event.artist.get_data()
        ind = event.ind[0]
        year = xdata[ind]
        country = event.artist.get_gid().split('|')[0]
        label = event.artist.get_gid().split('|')[2]
        desc = event.artist.get_gid().split('|')[3]
        
        annot.xy = (year, ydata[ind])
        annot.set_text(f"{label} ({year})\n{desc}")
        annot.get_bbox_patch().set_facecolor(style_map[country][0])
        annot.get_bbox_patch().set_alpha(0.8)
        annot.set_visible(True)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)

    # Formatting
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Cumulative Mass (tonnes)', fontsize=12)
    ax.set_title('Satellite Mass Accumulation with Historical Context', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1960, 2025)
    ax.set_ylim(0, data.max().max()*1.1)
    
    plt.tight_layout()
    plt.show()

# --- DATA PROCESSING (Use previous load_satcat and calculate_cumulative_mass functions) ---
# ... [Keep existing data loading and processing functions] ...

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    satcat = load_satcat()
    cumulative_data = calculate_cumulative_mass(satcat)
    plot_cumulative_mass(cumulative_data)
