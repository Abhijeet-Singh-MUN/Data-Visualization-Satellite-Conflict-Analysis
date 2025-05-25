import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- DATA LOADING ---
def load_satcat(file_path='output.csv'):
    """Load and preprocess satellite catalog data"""
    satcat = pd.read_csv(file_path, low_memory=False)
    
    # Parse launch dates and extract years
    satcat['LDate'] = pd.to_datetime(satcat['LDate'], errors='coerce')
    satcat['Year'] = satcat['LDate'].dt.year
    
    # Convert mass to numeric, handling invalid values
    satcat['Mass'] = pd.to_numeric(satcat['Mass'], errors='coerce').fillna(0)
    
    # Define country mapping
    country_map = {
        'US': 'US', 'CN': 'China', 'RU': 'Russia', 'SU': 'Russia',
        'J': 'Japan', 'IN': 'India', 'F': 'Western Europe', 
        'D': 'Western Europe', 'GB': 'Western Europe', 
        'I': 'Western Europe', 'E': 'Western Europe', 
        'NL': 'Western Europe', 'S': 'Western Europe', 
        'CH': 'Western Europe'
    }
    
    # Map countries based on State column
    satcat['Country_Group'] = (
        satcat['State']
        .str.split(r'[-/]').str[0]
        .map(country_map)
        .fillna('Other')
    )
    
    # Filter valid years
    satcat = satcat[(satcat['Year'] >= 1957) & (satcat['Year'] <= 2025)]
    
    return satcat

# --- CALCULATE CUMULATIVE MASS ---
def calculate_cumulative_mass(satcat):
    """Calculate cumulative mass over time for each country"""
    # Full range of years
    years = range(1957, 2026)
    
    # Create a DataFrame with all years for each country
    yearly_data = satcat.groupby(['Year', 'Country_Group'])['Mass'].sum().reset_index()
    
    # Get unique countries
    countries = yearly_data['Country_Group'].unique()
    
    # Initialize result DataFrame with all years
    result = pd.DataFrame({'Year': list(years)})
    
    # For each country, calculate cumulative mass year by year
    for country in countries:
        country_yearly = yearly_data[yearly_data['Country_Group'] == country]
        
        # Create a DataFrame for this country with all years
        country_all_years = pd.DataFrame({'Year': list(years)})
        
        # Merge with actual data
        country_all_years = country_all_years.merge(
            country_yearly[['Year', 'Mass']], 
            on='Year', 
            how='left'
        ).fillna(0)
        
        # Calculate cumulative sum
        country_all_years['Cumulative_Mass'] = country_all_years['Mass'].cumsum()
        
        # Add to result
        result[country] = country_all_years['Cumulative_Mass']
    
    return result

# --- PLOTTING ---
def plot_cumulative_mass(data):
    """Plot cumulative mass over time by country"""
    plt.figure(figsize=(14, 8))
    
    # Line styles and colors for each country
    styles = {
        'US': ('blue', '-'),
        'Russia': ('green', '-'),
        'China': ('red', '-'),
        'Japan': ('purple', '-'),
        'India': ('orange', '-'),
        'Western Europe': ('brown', '-'),
        'Other': ('gray', '-')
    }
    
    # Plot each country's cumulative mass
    for country in data.columns:
        if country != 'Year':
            color, style = styles.get(country, ('black', '-'))
            plt.plot(data['Year'], data[country], 
                     label=country, color=color, linestyle=style, linewidth=2)
    
    # Set up the axes
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Cumulative Mass (kg)', fontsize=14)
    
    # Format y-axis with scientific notation
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    # Add grid, legend, and title
    plt.grid(True, alpha=0.3)
    plt.legend(title='Country', fontsize=12)
    plt.title('Cumulative Satellite Mass by Country (1957-2025)', fontsize=16)
    
    # Set x-axis ticks at 5-year intervals
    plt.xticks(np.arange(1957, 2026, 5))
    
    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    satcat = load_satcat()
    cumulative_data = calculate_cumulative_mass(satcat)
    plot_cumulative_mass(cumulative_data)
