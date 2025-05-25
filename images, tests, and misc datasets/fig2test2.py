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

# --- ANNUAL MASS CALCULATION ---
def calculate_annual_mass(satcat):
    """Calculate annual mass sent to space by country"""
    # Get years from 1957 to 2025
    years = range(1957, 2026)
    
    # Calculate annual mass by country
    annual_mass = satcat.groupby(['Year', 'Country_Group'])['Mass'].sum().unstack(fill_value=0)
    
    # Ensure all years are present
    annual_mass = annual_mass.reindex(years, fill_value=0)
    
    # Convert kg to tonnes
    annual_mass = annual_mass / 1000
    
    return annual_mass

# --- CUMULATIVE MASS CALCULATION ---
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
    
    # Convert from kg to tonnes
    for column in result.columns:
        if column != 'Year':
            result[column] = result[column] / 1000
    
    return result

# --- PLOT ANNUAL MASS ---
def plot_annual_mass(data):
    """Plot annual mass sent to space by country"""
    plt.figure(figsize=(14, 8))
    
    # Line styles and colors for each country
    styles = {
        'US': ('blue', '-'),
        'Russia': ('green', '-'),
        'China': ('#00CC96', '-'),
        'Japan': ('maroon', '-'),
        'India': ('orange', '-'),
        'Western Europe': ('goldenrod', '-'),
        'Other': ('gray', '-')
    }
    
    # Plot each country's annual mass
    for country in data.columns:
        color, style = styles.get(country, ('black', '-'))
        plt.plot(data.index, data[country], label=country, color=color, linestyle=style, linewidth=2)
    
    # Set up the axes
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Total Mass (tonnes)', fontsize=14)
    
    # Add grid, legend, and title
    plt.grid(True, alpha=0.3)
    plt.legend(title='Country', fontsize=10)
    plt.title('Total Mass Sent to Space by Country (1957-2025)', fontsize=16)
    
    # Set x-axis ticks at 10-year intervals
    plt.xticks(np.arange(1960, 2026, 10))
    
    plt.tight_layout()
    plt.savefig('annual_mass.png', dpi=300)
    plt.show()

# --- PLOT CUMULATIVE MASS VS YEARS SINCE 1957 ---
def plot_cumulative_mass_vs_years(data):
    """Plot years since 1957 vs cumulative mass by country"""
    plt.figure(figsize=(14, 8))
    
    # Line styles and colors for each country
    styles = {
        'US': ('blue', '-'),
        'Russia': ('green', '-'),
        'China': ('#00CC96', '-'),
        'Japan': ('maroon', '-'),
        'India': ('orange', '-'),
        'Western Europe': ('goldenrod', '-'),
        'Other': ('gray', '-')
    }
    
    # Calculate years since 1957
    years_since_1957 = data['Year'] - 1957
    
    # Plot each country's data
    for country in data.columns:
        if country != 'Year':
            color, style = styles.get(country, ('black', '-'))
            plt.plot(data[country], years_since_1957, 
                     label=country, color=color, linestyle=style, linewidth=2)
    
    # Set up the axes
    plt.xlabel('Cumulative Mass (tonnes)', fontsize=14)
    plt.ylabel('Years Since 1957', fontsize=14)
    
    # Format x-axis with scientific notation but show in tonnes
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    current_formatter = plt.gca().xaxis.get_major_formatter()
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, pos: f'{x:.0f}' if x < 1000 else f'{x/1000:.0f}k'))
    
    # Create tick marks every 5 years
    tick_positions = np.arange(0, 70, 5)  # 0, 5, 10, ..., 65
    tick_labels = [str(1957 + year) for year in tick_positions]
    plt.yticks(tick_positions, tick_labels)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(title='Country', fontsize=10, loc='lower right')
    plt.title('Cumulative Mass Sent to Space by Country (1957-2025)', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('cumulative_mass.png', dpi=300)
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    try:
        # Load and process data
        satcat = load_satcat()
        
        # Calculate annual mass by country
        annual_data = calculate_annual_mass(satcat)
        plot_annual_mass(annual_data)
        
        # Calculate cumulative mass by country
        cumulative_data = calculate_cumulative_mass(satcat)
        plot_cumulative_mass_vs_years(cumulative_data)
        
        print("Visualization completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
