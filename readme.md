# Satellite Conflict Analysis Dashboard  
**Interactive visualization of orbital deployments and geopolitical tensions (1992-2025)**  

## Sample Output(Screenshots)

![Inforgraphic](Satellite Infographic ss.png)
![Satellite launches and hostility index](Satellite Launches ss.png)
![Satellite Mass Accumulation](Satellite Mass Accumulation ss.png)
![Satellie Lifespan](Satellite Lifespan ss.png)


## 🚀 Installation & Dependencies
**Python 3.7+ Required**  

**Jupyter Extensions** (for interactive widgets):  

## 📂 Dataset Requirements
| File | Source | Critical Notes |
|------|--------|----------------|
| `satcat.csv` | [SatCat](https://planet4589.org/space/gcat/web/cat/cols.html) | Pre-converted from TSV [using this tool](https://onlinetsvtools.com/convert-tsv-to-csv) |
| `MIDIP 5.0.csv` | [MIDIP](https://correlatesofwar.org/data-sets/mids/) | Original conflict dataset |

**Preprocessing Already Done** in provided files:
- Date range: 1992-2015
- Country grouping (US/China/Russia/Japan/India/Western Europe/Other)
- Orbital class calculations (LEO/MEO/GEO)
- Mass validation & error handling

## 🛰️ Key Features
### 1. Temporal Analysis Dashboard
- **Year Slider**: Explore 1992-2014 with hostility level overlays
- **Satellite Types**: Toggle civil/defense/comm satellites on the right side
- **Orbit Layers**: Compare LEO/MEO/GEO distributions

### 2. Mass Accumulation Timeline
- **Clickable Historical Markers**: Sputnik Crisis, Challenger Disaster, COVID impacts
- **Economic Context**: Recession/boom period shading
- **Country Trajectories**: Compare US/China/Russia mass accumulation

### 3. Lifespan Sunburst Chart
- **Interactive Drill-Down**: Country → Orbit → Avg lifespan
- **Time Filter**: 1957-2025 range slider
- "Click Country" : clicking country expands its orbital satellite distribution
- **Color Coding**: Purple (short) → Yellow (long)

## 🔧 Troubleshooting
**If widgets don't load**:  
1. Restart Jupyter kernel  
2. Run all cells
2. Verify extensions:  


**Common Data Issues**:  
Ensure MIDIP 5.0.csv is unmodified  
Use provided satcat.csv (pre-converted from tsv)  

## ✅ Data Validation


## 📜 Attribution
- Conflict Data: [MIDIP 5.0](https://correlatesofwar.org/data-sets/mids/) (Correlates of War Project)  
- Satellite Catalog: [SatCat](https://planet4589.org/space/gcat/web/cat/cols.html) (Jonathan McDowell)  
- TSV Conversion: [OnlineTSVTools](https://onlinetsvtools.com/convert-tsv-to-csv)  
- Vector Graphics: [Freepik](https://www.freepik.com) (modified under premium license)  
- Analysis & Visualization: [Abhijeet Singh]  

---

**Launch Instructions**:  
1. Place both CSV files in project root  
2. Start Jupyter: `jupyter lab`  
3. Run all cells (Widgets may take 10-15s to initialize)  



