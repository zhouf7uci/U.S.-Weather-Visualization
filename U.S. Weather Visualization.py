# import the necessary libraries
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import math
import us # for converting state names to numeric geolocation
import plotly.graph_objects as go # to plot heatmaps
import gmaps # to plot heatmaps using Google Maps
# libraries for folium heatmaps
import folium
from folium import plugins

# # Data Selection

# Daily Weather in the U.S., 2017. This dataset contains daily U.S. weather measurements in 2017 provided by the NOAA Daily Global Historical Climatology Network.

# # The First Phase

# 1. What variables does the dataset contain?
#     1. Station
#     2. State
#     3. Latitude
#     4. Longitude
#     5. Elevation
#     6. Date
#     7. TMIN = Minimum temperature (F)
#     8. TMAX = Maximum temperature (F)
#     9. TAVG = Average temperature (F)
#     10. AWND = Average daily wind speed (miles / hour)
#     11. WDF5 = Direction of fastest 5-second wind (degrees)
#     12. WSF5 = Fastest 5-second wind speed (miles / hour)
#     13. SNOW = Snowfall (inch)
#     14. SNWD = Snow depth (inch)
#     15. PRCP = Precipitation (inch)
# 
# 2. What type of variable does the dataset contain? (e.g., nominal, ordinal, discrete).
#     1. Station  Type: nominal categorical value
#     2. State    Type: nominal categorical value
#     3. Latitude Type: continuous numerical value 
#     4. Longitude Type: continuous numerical value
#     5. Elevation Type: continuous numerical value
#     6. Date Type: discrete numerical value
#     7. TMIN = Minimum temperature (F) Type: continuous numerical value
#     8. TMAX = Maximum temperature (F) Type: continuous numerical value
#     9. TAVG = Average temperature (F) Type: continuous numerical value
#     10. AWND = Average daily wind speed (miles / hour) Type: continuous numerical value
#     11. WDF5 = Direction of fastest 5-second wind (degrees) Type: continuous numerical value
#     12. WSF5 = Fastest 5-second wind speed (miles / hour) Type: continuous numerical value
#     13. SNOW = Snowfall (inch) Type: continuous numerical value
#     14. SNWD = Snow depth (inch) Type: continuous numerical value
#     15. PRCP = Precipitation (inch) Type: continuous numerical value
# 
# 3. How are they distributed?
#     1. From the first impression of the dataset, there is no obvious distribution that can be observed.
# 
# 4. Are there any relationships among the variables?
#     1. The station with a higher latitude has a lower average temperature, and vice versa.
#     2. The station with the fastest wind has a lower average temperature, and vice versa.
#     3. The higher elevation station has a lower average temperature, and vice versa. 
# 
# 

# # The Second Phase

# In[2]:


#Dataset: Daily Weather in the U.S., 2017.
url_le = '/Users/heyahe/Downloads/weather.csv'
df1 = pd.read_csv(url_le)
df1.head(25)

# Drop all rows with NaN values
df2=df1.dropna()
df2.head(25)


# # 1. What is the distribution of temperature across the United States?

# # The Original Plot

# In[3]:


#The Original Plot
df3 = df2.groupby('state').TAVG.mean().reset_index().sort_values(by='TAVG', ascending=True)
fig, ax = plt.subplots(figsize=(7,10))
ax.scatter(df3['TAVG'], df3['state'])
# remove chart border
for spine in plt.gca().spines.values():
    spine.set_visible(False)
ax.grid()


# # Final Visualization

# In[4]:


BLACK = '#000000'
fig, ax = plt.subplots(figsize=(7,10))
colors = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100])
sc = ax.scatter(df3['TAVG'], df3['state'], c=colors, cmap='plasma')
plt.colorbar(sc)

ax.set_xlabel('Average temperature (F)')
ax.set_ylabel('State')
ax.set_title(
        "Average temperature in Americas in 2017: Majority above 50°F",
        fontsize=20, 
        color=BLACK, 
        fontweight='bold',
        loc='left',
        x=-0.29,
        y=1.05)

# remove chart border
for spine in plt.gca().spines.values():
    spine.set_visible(False)

ax.yaxis.grid()


# # Caption: 
# Most of the state has an average temperature above 50 °F. Alaska has the lowest average temperature in the country, at 35.76°F, while Guam has the highest, at 82.14°F. 

# # 2. Will the average daily wind speed have an impact on the local temperature?

# # The Original Plot

# In[5]:


#The Original Plot
fig, ax = plt.subplots(figsize=(7,10))
ax.scatter(df2['AWND'], df2['TAVG'])

# remove chart border
for spine in plt.gca().spines.values():
    spine.set_visible(False)
ax.grid()


# # Follow-up questions
# There are too many data points on the original plot for us to see a general relationship between average wind speed and average temperature. If we calculate the average wind speed and average temperature for each state, what will the plot look like?

# # Refine the Visualization

# In[6]:


df4 = df2.groupby('state')[["TAVG","AWND"]].mean().reset_index()

BLACK = '#000000'
fig, ax = plt.subplots(figsize=(7,10))
sc = ax.scatter(df4['AWND'], df4['TAVG'])

ax.set_xlabel('Average daily wind speed (miles / hour)')
ax.set_ylabel('Average temperature (F)')
ax.set_title(
        "Average wind speed versus average temperature for 50 States",  
        fontsize=20, 
        color=BLACK, 
        fontweight='bold',
        loc='left',
        x=-0.29,
        y=1.05)

#Create Linear Regression
X = df4.AWND.values.reshape(-1,1)
Y = df4.TAVG.values.reshape(-1,1)
linear_regressor = LinearRegression()
linear_regressor.fit(X,Y)
Y_pred = linear_regressor.predict(X)
plt.plot(X,Y_pred, color='red')

# remove chart border
for spine in plt.gca().spines.values():
    spine.set_visible(False)

ax.yaxis.grid()


# # Follow-up questions
# Will the outliers influence the relationship between average daily wind speed and average temperature? What will the graph look like after the outliers are removed?

# # Final Visualizaion

# In[7]:


df4_1 = df4.drop(labels=[0,10], axis=0)

BLACK = '#000000'
fig, ax = plt.subplots(figsize=(7,10))
sc = ax.scatter(df4_1['AWND'], df4_1['TAVG'])

ax.set_xlabel('Average daily wind speed (miles / hour)')
ax.set_ylabel('Average temperature (F)')
ax.set_title(
        "Average wind speed versus average temperature for 50 States",  
        fontsize=20, 
        color=BLACK, 
        fontweight='bold',
        loc='left',
        x=-0.29,
        y=1.05)

#Create Linear Regression
X = df4_1.AWND.values.reshape(-1,1)
Y = df4_1.TAVG.values.reshape(-1,1)
linear_regressor = LinearRegression()
linear_regressor.fit(X,Y)
Y_pred = linear_regressor.predict(X)
plt.plot(X,Y_pred, color='red')

# remove chart border
for spine in plt.gca().spines.values():
    spine.set_visible(False)

ax.yaxis.grid()


# # Caption:
# Average daily wind speed and average temperature have a strong negative relationship: states with higher average wind speed have lower average temperature, and vice versa.

# # 3. How do latitude and longitude influence the average temperature?

# # The Orginal Plot

# In[8]:


df5 = df2.groupby('state')[["latitude", "longitude","TAVG","elevation"]].mean()
m=folium.Map(
    location=[df5["latitude"].mean(), df5["longitude"].mean()],
    zoom_start=6)
df5.apply(
    lambda row: folium.Marker(
        location=[row['latitude'], row['longitude']]
        ).add_to(m),
    axis=1)
m


# # Refine the Visualization

# In[9]:


m=folium.Map(
    location=[df5["latitude"].mean(), df5["longitude"].mean()],
    zoom_start=4)

def get_icon():
    return folium.Icon(
                       color='black',
                       icon_color='#2ecc71'
                       )
df5.apply(
    lambda row: folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup="Average Temperature: %.2fF"%row['TAVG'],
        tooltip='<h5>Click here for more info</h5>',
        icon=get_icon(),
        ).add_to(m),
    axis=1)

title_html = '''
             <h3 align="center" style="font-size:20px"><b>Interactive Geospatial Map for average temperature in Americas</b></h3>
             '''
m.get_root().html.add_child(folium.Element(title_html))
m


# # Follow-up questions
# Only an open street map cannot reflect the real geographical location. Will using a terrain map help to better understand the real geographical location?

# # Final Visualization

# In[10]:


m=folium.Map(
    location=[df5["latitude"].mean(), df5["longitude"].mean()],
    zoom_start=4)

#Add terrain layer to the map
folium.TileLayer('Stamen Terrain').add_to(m)
folium.LayerControl().add_to(m)

def get_icon():
    return folium.Icon(
                       color='black',
                       icon_color='#2ecc71'
                       )
df5.apply(
    lambda row: folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup="Average Temperature: %.2fF"%row['TAVG'],
        tooltip='<h5>Click here for more info</h5>',
        icon=get_icon(),
        ).add_to(m),
    axis=1)

title_html = '''
             <h3 align="center" style="font-size:20px"><b>Interactive Geospatial Map for average temperature in Americas</b></h3>
             '''
m.get_root().html.add_child(folium.Element(title_html))
m


# # Caption:
# From left to right, there is no significant difference in average temperature for different states, implying that longitude has no significant influence on average temperature.
# From bottom to top, the states with the higher latitude have the lower average temperature, and vice versa. This implies that there is a strong negative correlation between latitude and average temperature.

# # 4. How are snowfall and snow depth distributed across the United States?

# # The Original Plot

# In[11]:


df6 = df2.groupby('state')[["SNOW", "SNWD"]].max().reset_index()
def plot_heatmap(df, var): 
    # plotting the heatmap by states
    fig = go.Figure(data=go.Choropleth(
        locations=df['state'], # Spatial coordinates
        locationmode = 'USA-states', # set of locations match entries in `locations`
        colorscale = 'Reds',
        colorbar_title = var,
    ))
    fig.update_layout(
        title_text = var + ' by state<br>',
        geo_scope='usa', # limit map scope to USA
    )
    fig.show()
plot_heatmap(df6, 'SNOW')
plot_heatmap(df6, 'SNWD')


# # Final Visualization

# In[12]:


def plot_heatmap(df, var): 
    # plotting the heatmap by states
    fig = go.Figure(data=go.Choropleth(
        locations=df['state'], # Spatial coordinates
        z = df[var].astype(float), # Data to be color-coded
        locationmode = 'USA-states', # set of locations match entries in `locations`
        colorscale = 'Reds',
        colorbar_title = var,
        text = df['state']
    ))
    fig.update_layout(
        title_text = var + ' by state<br>(Hover over the states for details)',
        geo_scope='usa', # limit map scope to USA
    )
    fig.show()
plot_heatmap(df6, 'SNOW')
plot_heatmap(df6, 'SNWD')


# # Caption:
# New York had the greatest snowfall in 2017, which reached 31.2 inches, while the southern states like Arizona, Georgia, and Florida had no snowfall in 2017.Also, Alaska has the greatest snow depth due to its location as the northernmost state in America.

# # Follow-up question
# Since we created several visualizations to investigate the impact of factors such as wind speed, latitude, longitude, and location on average temperature, could we create a correlation heatmap that shows all of the relationships between each factor?

# # Final Visualization

# In[13]:


df7 = df2.groupby('state')[["latitude","longitude","elevation","TMIN","TMAX","TAVG","AWND","WSF5","SNOW","SNWD","PRCP"]].mean().reset_index()
df7_1 = df7.drop(labels="state", axis=1)
plt.figure(figsize=(16, 6))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(df7_1.corr(), dtype=np.bool))
heatmap = sns.heatmap(df7_1.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16);


# # Caption:
# 1. There is a strong negative correlation between temperature and latitude; the higher the latitude, the lower the temperature.
# 2. The elevation has a strong positive correlation with the fastest 5-second wind speed; the higher the elevation, the faster the 5-second wind speed. Also, it has a strong negative correlation with precipitation, with the higher elevation expected to have lower precipitation. The elevation has no significant influence on the temperature.
# 3. There is a negative correlation between temperature and snowfall; the greater the snowfall, the lower the temperature.

