import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- DATA LOADING WITH VALIDATION ---
def load_satcat(file_path='output.csv'):
    """Load and preprocess satellite catalog data with mass validation"""
    satcat = pd.read_csv(file_path, low_memory=False)
    
    # Parse launch dates and extract years
    satcat['LDate'] = pd.to_datetime(satcat['LDate'], errors='coerce')
    satcat['Year'] = satcat['LDate'].dt.year
    
    # Convert and validate mass values
    satcat['Mass'] = pd.to_numeric(satcat['Mass'], errors='coerce')
    
    # Handle negative mass values (set to absolute value)
    satcat['Mass'] = satcat['Mass'].abs()
    
    # Define country mapping
    country_map = {
        'US': 'US', 'CN': 'China', 'RU': 'Russia', 'SU': 'Russia',
        'J': 'Japan', 'IN': 'India', 'F': 'Western Europe', 
        'D': 'Western Europe', 'GB': 'Western Europe', 
        'I': 'Western Europe', 'E': 'Western Europe', 
        'NL': 'Western Europe', 'S': 'Western Europe', 
        'CH': 'Western Europe'
    }
    
    # Map countries and filter valid years
    satcat['Country_Group'] = (
        satcat['State']
        .str.split(r'[-/]').str[0]
        .map(country_map)
        .fillna('Other')
    )
    
    return satcat[
        (satcat['Year'] >= 1957) & 
        (satcat['Year'] <= 2025) &
        (satcat['Mass'].notna())
    ]

# --- CUMULATIVE MASS CALCULATION WITH ERROR CHECKING ---
def calculate_cumulative_mass(satcat):
    """Calculate non-decreasing cumulative mass over time"""
    # Create full year range
    years = pd.DataFrame({'Year': range(1957, 2026)})
    
    # Group and sum mass by year and country
    grouped = satcat.groupby(['Year', 'Country_Group'])['Mass'].sum().reset_index()
    
    # Pivot to wide format with countries as columns
    pivoted = grouped.pivot(index='Year', columns='Country_Group', values='Mass').fillna(0)
    
    # Reindex to ensure all years are present
    pivoted = pivoted.reindex(range(1957, 2026), fill_value=0)
    
    # Convert kg to tonnes
    pivoted = pivoted / 1000
    
    # Calculate cumulative sum with non-decreasing enforcement
    cumulative = pivoted.cumsum()
    
    # Ensure cumulative values never decrease
    for country in cumulative.columns:
        prev = cumulative[country].iloc[0]
        for i in range(1, len(cumulative)):
            current = cumulative[country].iloc[i]
            if current < prev:
                cumulative.iloc[i, cumulative.columns.get_loc(country)] = prev
            prev = cumulative[country].iloc[i]
    
    return cumulative.reset_index()

# --- PLOT CUMULATIVE MASS OVER TIME ---
def plot_cumulative_mass(data):
    """Plot cumulative mass timeline with proper formatting"""
    plt.figure(figsize=(14, 8))
    
    # Color scheme and style mapping
    style_map = {
        'US': ('#2E86C1', '-'),         # Blue
        'China': ('#00CC96', '-'),      # Green
        'Russia': ('#FF6B6B', '-'),     # Red
        'Japan': ('#6B3FA0', '-'),      # Purple
        'India': ('#FFA500', '-'),      # Orange
        'Western Europe': ('#8B4513', '-'), # Brown
        'Other': ('#708090', '--')      # Gray dashed
    }
    
    # Plot each country's cumulative mass
    for country in data.columns[1:]:  # Skip 'Year' column
        color, linestyle = style_map.get(country, ('#333333', '-'))
        plt.plot(
            'Year', country, 
            data=data, 
            color=color, 
            linestyle=linestyle, 
            linewidth=2.5, 
            label=country
        )
    
    # Axis and grid configuration
    plt.xlabel('Year', fontsize=14, labelpad=10)
    plt.ylabel('Cumulative Mass (tonnes)', fontsize=14, labelpad=10)
    plt.xticks(np.arange(1960, 2026, 5), rotation=45)
    plt.gca().xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))
    plt.grid(True, alpha=0.3)
    
    # Legend and title
    plt.legend(
        title='Country/Group', 
        fontsize=10, 
        loc='upper left', 
        bbox_to_anchor=(1, 1)
    )
    plt.title(
        'Cumulative Satellite Mass in Orbit (1957-2025)\n'
        'Non-Decreasing Validation Applied',
        fontsize=16,
        pad=20
    )
    
    # Layout and save
    plt.tight_layout()
    plt.savefig('cumulative_mass_timeline.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    try:
        print("Loading and processing data...")
        satcat = load_satcat()
        
        print("Calculating cumulative mass...")
        cumulative_data = calculate_cumulative_mass(satcat)
        
        print("Generating visualization...")
        plot_cumulative_mass(cumulative_data)
        
        print("Process completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
