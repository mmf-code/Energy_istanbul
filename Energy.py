#!/usr/bin/env python
# coding: utf-8

# In[12]:


import netCDF4 as nc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


# In[13]:


# data2'nin yüklenmesi
data2 = xr.open_dataset('D:/ruzgarh.nc')

# İstanbul'un latitude ve longitude koordinatları
istanbul_latitude = 41.0082
istanbul_longitude = 28.9784

# İstanbul'un yakın çevresindeki veri sınırlaması
data_istanbul = data2.sel(latitude=istanbul_latitude, longitude=istanbul_longitude, method='nearest')

# u10 ve v10 bileşenlerinin alınması
u10 = data_istanbul['u10']
v10 = data_istanbul['v10']

# Ortalama rüzgar hızının hesaplanması
wind_speed_mean = ((u10**2 + v10**2)**0.5).mean(dim='time')

# Çıktının alınması
print(wind_speed_mean)



# In[14]:


# Load data
data_january = xr.open_dataset('D:/ruzgarh.nc')
data2 = xr.open_dataset('D:/ruzgarh.nc')

# Extract u10 and v10 components
u10_january = data_january['u10']
v10_january = data_january['v10']

# Calculate wind speed
wind_speed_january = (u10_january**2 + v10_january**2)**0.5

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot wind vectors
q = ax.quiver(data_january['longitude'], data_january['latitude'],
              u10_january.mean(dim='time'), v10_january.mean(dim='time'), wind_speed_january.mean(dim='time'),
              transform=ccrs.PlateCarree(), cmap='jet', width=0.07, scale=800)

plt.colorbar(q, label='Wind Speed (m/s)')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.title('Average Wind Speed in January (Near Istanbul)')

# Save the figure as a PNG file
plt.savefig('D:/energy/wind/1.png', dpi=300, bbox_inches='tight')

plt.show()

# Load data
data_february = xr.open_dataset('D:/ruzgarh2.nc')

# Extract u10 and v10 components
u10_february = data_february['u10']
v10_february = data_february['v10']

# Calculate wind speed
wind_speed_february = (u10_february**2 + v10_february**2)**0.5

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot wind vectors
q = ax.quiver(data2['longitude'], data2['latitude'],
              u10_february.mean(dim='time'), v10_february.mean(dim='time'), wind_speed_february.mean(dim='time'),
              transform=ccrs.PlateCarree(), cmap='jet', width=0.07, scale=800)

plt.colorbar(q, label='Wind Speed (m/s)')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.title('Average Wind Speed in February (Near Istanbul)')
# Save the figure as a PNG file
plt.savefig('D:/energy/wind/2.png', dpi=300, bbox_inches='tight')

plt.show()

# Load data
data_march = xr.open_dataset('D:/ruzgarh3.nc')

# Extract u10 and v10 components
u10_march = data_march['u10']
v10_march = data_march['v10']

# Calculate wind speed
wind_speed_march = (u10_march**2 + v10_march**2)**0.5

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot wind vectors
q = ax.quiver(data2['longitude'], data2['latitude'],
              u10_march.mean(dim='time'), v10_march.mean(dim='time'), wind_speed_march.mean(dim='time'),
              transform=ccrs.PlateCarree(), cmap='jet', width=0.07, scale=800)

plt.colorbar(q, label='Wind Speed (m/s)')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.title('Average Wind Speed in March (Near Istanbul)')
# Save the figure as a PNG file
plt.savefig('D:/energy/wind/3.png', dpi=300, bbox_inches='tight')

plt.show()

# Load data
data_april = xr.open_dataset('D:/ruzgarh4.nc')

# Extract u10 and v10 components
u10_april = data_april['u10']
v10_april = data_april['v10']

# Calculate wind speed
wind_speed_april = (u10_april**2 + v10_april**2)**0.5

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot wind vectors
q = ax.quiver(data2['longitude'], data2['latitude'],
              u10_april.mean(dim='time'), v10_april.mean(dim='time'), wind_speed_april.mean(dim='time'),
              transform=ccrs.PlateCarree(), cmap='jet', width=0.07, scale=800)

plt.colorbar(q, label='Wind Speed (m/s)')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.title('Average Wind Speed in April (Near Istanbul)')
# Save the figure as a PNG file
plt.savefig('D:/energy/wind/4.png', dpi=300, bbox_inches='tight')

plt.show()

# Load data
data_may = xr.open_dataset('D:/ruzgarh5.nc')

# Extract u10 and v10 components
u10_may = data_may['u10']
v10_may = data_may['v10']

# Calculate wind speed
wind_speed_may = (u10_may**2 + v10_may**2)**0.5

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot wind vectors
q = ax.quiver(data2['longitude'], data2['latitude'],
              u10_may.mean(dim='time'), v10_may.mean(dim='time'), wind_speed_may.mean(dim='time'),
              transform=ccrs.PlateCarree(), cmap='jet', width=0.07, scale=800)

plt.colorbar(q, label='Wind Speed (m/s)')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.title('Average Wind Speed in May (Near Istanbul)')
# Save the figure as a PNG file
plt.savefig('D:/energy/wind/5.png', dpi=300, bbox_inches='tight')

plt.show()

# Load data
data_june = xr.open_dataset('D:/ruzgarh6.nc')

# Extract u10 and v10 components
u10_june = data_june['u10']
v10_june = data_june['v10']

# Calculate wind speed
wind_speed_june = (u10_june**2 + v10_june**2)**0.5

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot wind vectors
q = ax.quiver(data2['longitude'], data2['latitude'],
              u10_june.mean(dim='time'), v10_june.mean(dim='time'), wind_speed_june.mean(dim='time'),
              transform=ccrs.PlateCarree(), cmap='jet', width=0.07, scale=800)

plt.colorbar(q, label='Wind Speed (m/s)')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.title('Average Wind Speed in June (Near Istanbul)')
# Save the figure as a PNG file
plt.savefig('D:/energy/wind/6.png', dpi=300, bbox_inches='tight')
plt.show()

# Load data
data_july = xr.open_dataset('D:/ruzgarh7.nc')

# Extract u10 and v10 components
u10_july = data_july['u10']
v10_july = data_july['v10']

# Calculate wind speed
wind_speed_july = (u10_july**2 + v10_july**2)**0.5

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot wind vectors
q = ax.quiver(data2['longitude'], data2['latitude'],
              u10_july.mean(dim='time'), v10_july.mean(dim='time'), wind_speed_july.mean(dim='time'),
              transform=ccrs.PlateCarree(), cmap='jet', width=0.07, scale=800)

plt.colorbar(q, label='Wind Speed (m/s)')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.title('Average Wind Speed in July (Near Istanbul)')
# Save the figure as a PNG file
plt.savefig('D:/energy/wind/7.png', dpi=300, bbox_inches='tight')
plt.show()

# Load data
data_august = xr.open_dataset('D:/ruzgarh8.nc')

# Extract u10 and v10 components
u10_august = data_august['u10']
v10_august = data_august['v10']

# Calculate wind speed
wind_speed_august = (u10_august**2 + v10_august**2)**0.5

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot wind vectors
q = ax.quiver(data2['longitude'], data2['latitude'],
              u10_august.mean(dim='time'), v10_august.mean(dim='time'), wind_speed_august.mean(dim='time'),
              transform=ccrs.PlateCarree(), cmap='jet', width=0.07, scale=800)

plt.colorbar(q, label='Wind Speed (m/s)')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.title('Average Wind Speed in August (Near Istanbul)')
# Save the figure as a PNG file
plt.savefig('D:/energy/wind/8.png', dpi=300, bbox_inches='tight')
plt.show()

# Load data
data_sep = xr.open_dataset('D:/ruzgarh9.nc')

# Extract u10 and v10 components
u10_sep = data_sep['u10']
v10_sep = data_sep['v10']

# Calculate wind speed
wind_speed_sep = (u10_sep**2 + v10_sep**2)**0.5

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot wind vectors
q = ax.quiver(data2['longitude'], data2['latitude'],
              u10_sep.mean(dim='time'), v10_sep.mean(dim='time'), wind_speed_sep.mean(dim='time'),
              transform=ccrs.PlateCarree(), cmap='jet', width=0.07, scale=800)

plt.colorbar(q, label='Wind Speed (m/s)')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.title('Average Wind Speed in September (Near Istanbul)')
# Save the figure as a PNG file
plt.savefig('D:/energy/wind/9.png', dpi=300, bbox_inches='tight')
plt.show()

# Load data
data_oct = xr.open_dataset('D:/ruzgarh10.nc')

# Extract u10 and v10 components
u10_oct = data_oct['u10']
v10_oct = data_oct['v10']

# Calculate wind speed
wind_speed_oct = (u10_oct**2 + v10_oct**2)**0.5

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot wind vectors
q = ax.quiver(data2['longitude'], data2['latitude'],
              u10_oct.mean(dim='time'), v10_oct.mean(dim='time'), wind_speed_oct.mean(dim='time'),
              transform=ccrs.PlateCarree(), cmap='jet', width=0.07, scale=800)

plt.colorbar(q, label='Wind Speed (m/s)')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.title('Average Wind Speed in October (Near Istanbul)')
# Save the figure as a PNG file
plt.savefig('D:/energy/wind/10.png', dpi=300, bbox_inches='tight')
plt.show()

# Load data
data_nov = xr.open_dataset('D:/ruzgarh11.nc')

# Extract u10 and v10 components
u10_nov = data_nov['u10']
v10_nov = data_nov['v10']

# Calculate wind speed
wind_speed_nov = (u10_nov**2 + v10_nov**2)**0.5

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot wind vectors
q = ax.quiver(data2['longitude'], data2['latitude'],
              u10_nov.mean(dim='time'), v10_nov.mean(dim='time'), wind_speed_nov.mean(dim='time'),
              transform=ccrs.PlateCarree(), cmap='jet', width=0.07, scale=800)

plt.colorbar(q, label='Wind Speed (m/s)')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.title('Average Wind Speed in November (Near Istanbul)')
# Save the figure as a PNG file
plt.savefig('D:/energy/wind/11.png', dpi=300, bbox_inches='tight')
plt.show()

# Load data
data_dec = xr.open_dataset('D:/ruzgarh12.nc')

# Extract u10 and v10 components
u10_dec = data_dec['u10']
v10_dec = data_dec['v10']

# Calculate wind speed
wind_speed_dec = (u10_dec**2 + v10_dec**2)**0.5

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot wind vectors
q = ax.quiver(data2['longitude'], data2['latitude'],
              u10_dec.mean(dim='time'), v10_dec.mean(dim='time'), wind_speed_dec.mean(dim='time'),
              transform=ccrs.PlateCarree(), cmap='jet', width=0.07, scale=800)

plt.colorbar(q, label='Wind Speed (m/s)')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.title('Average Wind Speed in December (Near Istanbul)')
# Save the figure as a PNG file
plt.savefig('D:/energy/wind/12.png', dpi=300, bbox_inches='tight')
plt.show()


# In[15]:


from PIL import Image

# Create GIF from the images
image_frames = []
days = np.arange(1, 13)  # 12 images from 1 to 12

for day in days:
    new_frame = Image.open(f'D:/energy/wind/{day}.png')
    image_frames.append(new_frame)

# Save as GIF
image_frames[0].save('D:/energy/wind/wind.gif', format='GIF',
                     append_images=image_frames[1:],
                     save_all=True,
                     duration=400, loop=0)


# In[16]:


# İstanbul'un latitude ve longitude koordinatları
istanbul_latitude = 41.35
istanbul_longitude = 28.42

# İstanbul'un yakın çevresindeki veri sınırlaması
data_istanbul = data2.sel(latitude=istanbul_latitude, longitude=istanbul_longitude, method='nearest')


# In[17]:


# Calculate wind speed
wind_speed_january = (u10_january**2 + v10_january**2)**0.5

# İstenilen koordinatlara en yakın noktanın indeksini bulma
lat_index = np.abs(data_january['latitude'] - istanbul_latitude).argmin()
lon_index = np.abs(data_january['longitude'] - istanbul_longitude).argmin()

# İstenilen koordinatlardaki rüzgar hızını alma
wind_speed_value = wind_speed_january[:, lat_index, lon_index]
avg_january = wind_speed_value.mean()

# Zaman serisi grafiğini çizdirme
plt.plot(data_january['time'], wind_speed_value, label='January')

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
plt.title('Wind Speed Time Series at lat={}, lon={}'.format(istanbul_latitude, istanbul_longitude))
plt.xticks(rotation=45) 

# Ortalamayı hesaplama
mean_value = wind_speed_value.mean()
mean_label = f'Mean: {mean_value:.2f} m/s'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Grafik gösterme
avg_january = mean_value
avg_january_value = avg_january.item()
print(avg_january_value)

plt.legend()
plt.show()


# In[18]:


# Calculate wind speed
wind_speed_february = (u10_february**2 + v10_february**2)**0.5

# İstenilen koordinatlara en yakın noktanın indeksini bulma
lat_index = np.abs(data_february['latitude'] - lat).argmin()
lon_index = np.abs(data_february['longitude'] - lon).argmin()

# İstenilen koordinatlardaki rüzgar hızını alma
wind_speed_value = wind_speed_february[:, lat_index, lon_index]
avg_february = wind_speed_value.mean()

# Zaman serisi grafiğini çizdirme
plt.plot(data_february['time'], wind_speed_value, label='February')

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
plt.title('Wind Speed Time Series at lat=41.35, lon=28.42')
plt.xticks(rotation=45) 

# Ortalamayı hesaplama
mean_value = wind_speed_value.mean()
mean_label = f'Mean: {mean_value:.2f} m/s'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Grafik gösterme
avg_february = mean_value
avg_february_value = avg_february.item()
print(avg_february_value)

plt.legend()
plt.show()


# In[ ]:


# Calculate wind speed
wind_speed_march = (u10_march**2 + v10_march**2)**0.5

# İstenilen koordinatlara en yakın noktanın indeksini bulma
lat_index = np.abs(data_march['latitude'] - lat).argmin()
lon_index = np.abs(data_march['longitude'] - lon).argmin()

# İstenilen koordinatlardaki rüzgar hızını alma
wind_speed_value = wind_speed_march[:, lat_index, lon_index]
avg_march = wind_speed_value.mean()

# Zaman serisi grafiğini çizdirme
plt.plot(data_march['time'], wind_speed_value, label='March')

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
plt.title('Wind Speed Time Series at lat=41.35, lon=28.42')
plt.xticks(rotation=45) 

# Ortalamayı hesaplama
mean_value = wind_speed_value.mean()
mean_label = f'Mean: {mean_value:.2f} m/s'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Grafik gösterme
avg_march = mean_value
avg_march_value = avg_march.item()
print(avg_march_value)

plt.legend()
plt.show()


# In[ ]:


# Calculate wind speed
wind_speed_april = (u10_april**2 + v10_april**2)**0.5

# İstenilen koordinatlara en yakın noktanın indeksini bulma
lat_index = np.abs(data_april['latitude'] - lat).argmin()
lon_index = np.abs(data_april['longitude'] - lon).argmin()

# İstenilen koordinatlardaki rüzgar hızını alma
wind_speed_value = wind_speed_april[:, lat_index, lon_index]
avg_april = wind_speed_value.mean()

# Zaman serisi grafiğini çizdirme
plt.plot(data_april['time'], wind_speed_value, label='April')

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
plt.title('Wind Speed Time Series at lat=41.35, lon=28.42')
plt.xticks(rotation=45) 

# Ortalamayı hesaplama
mean_value = wind_speed_value.mean()
mean_label = f'Mean: {mean_value:.2f} m/s'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Grafik gösterme
avg_april = mean_value
avg_april_value = avg_april.item()
print(avg_april_value)

plt.legend()
plt.show()


# In[ ]:


# Calculate wind speed
wind_speed_may = (u10_may**2 + v10_may**2)**0.5

# İstenilen koordinatlara en yakın noktanın indeksini bulma
lat_index = np.abs(data_may['latitude'] - lat).argmin()
lon_index = np.abs(data_may['longitude'] - lon).argmin()

# İstenilen koordinatlardaki rüzgar hızını alma
wind_speed_value = wind_speed_may[:, lat_index, lon_index]
avg_may = wind_speed_value.mean()

# Zaman serisi grafiğini çizdirme
plt.plot(data_may['time'], wind_speed_value, label='May')

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
plt.title('Wind Speed Time Series at lat=41.35, lon=28.42')
plt.xticks(rotation=45) 

# Ortalamayı hesaplama
mean_value = wind_speed_value.mean()
mean_label = f'Mean: {mean_value:.2f} m/s'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Grafik gösterme
avg_may = mean_value
avg_may_value = avg_may.item()
print(avg_may_value)

plt.legend()
plt.show()


# In[ ]:


# Calculate wind speed
wind_speed_june = (u10_june**2 + v10_june**2)**0.5

# İstenilen koordinatlara en yakın noktanın indeksini bulma
lat_index = np.abs(data_june['latitude'] - lat).argmin()
lon_index = np.abs(data_june['longitude'] - lon).argmin()

# İstenilen koordinatlardaki rüzgar hızını alma
wind_speed_value = wind_speed_june[:, lat_index, lon_index]
avg_june = wind_speed_value.mean()

# Zaman serisi grafiğini çizdirme
plt.plot(data_june['time'], wind_speed_value, label='June')

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
plt.title('Wind Speed Time Series at lat=41.35, lon=28.42')
plt.xticks(rotation=45) 

# Ortalamayı hesaplama
mean_value = wind_speed_value.mean()
mean_label = f'Mean: {mean_value:.2f} m/s'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Grafik gösterme
avg_june = mean_value
avg_june_value = avg_june.item()
print(avg_june_value)

plt.legend()
plt.show()


# In[ ]:


# Calculate wind speed
wind_speed_july = (u10_july**2 + v10_july**2)**0.5

# İstenilen koordinatlara en yakın noktanın indeksini bulma
lat_index = np.abs(data_july['latitude'] - lat).argmin()
lon_index = np.abs(data_july['longitude'] - lon).argmin()

# İstenilen koordinatlardaki rüzgar hızını alma
wind_speed_value = wind_speed_july[:, lat_index, lon_index]
avg_july = wind_speed_value.mean()

# Zaman serisi grafiğini çizdirme
plt.plot(data_july['time'], wind_speed_value, label='July')

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
plt.title('Wind Speed Time Series at lat=41.35, lon=28.42')
plt.xticks(rotation=45) 

# Ortalamayı hesaplama
mean_value = wind_speed_value.mean()
mean_label = f'Mean: {mean_value:.2f} m/s'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Grafik gösterme
avg_july = mean_value
avg_july_value = avg_july.item()
print(avg_july_value)

plt.legend()
plt.show()


# In[ ]:


# Calculate wind speed
wind_speed_august = (u10_august**2 + v10_august**2)**0.5

# İstenilen koordinatlara en yakın noktanın indeksini bulma
lat_index = np.abs(data_august['latitude'] - lat).argmin()
lon_index = np.abs(data_august['longitude'] - lon).argmin()

# İstenilen koordinatlardaki rüzgar hızını alma
wind_speed_value = wind_speed_august[:, lat_index, lon_index]
avg_august = wind_speed_value.mean()

# Zaman serisi grafiğini çizdirme
plt.plot(data_august['time'], wind_speed_value, label='August')

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
plt.title('Wind Speed Time Series at lat=41.35, lon=28.42')
plt.xticks(rotation=45) 

# Ortalamayı hesaplama
mean_value = wind_speed_value.mean()
mean_label = f'Mean: {mean_value:.2f} m/s'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Grafik gösterme
avg_august = mean_value
avg_august_value = avg_august.item()
print(avg_august_value)

plt.legend()
plt.show()


# In[ ]:


# Calculate wind speed
wind_speed_sep = (u10_sep**2 + v10_sep**2)**0.5

# İstenilen koordinatlara en yakın noktanın indeksini bulma
lat_index = np.abs(data_sep['latitude'] - lat).argmin()
lon_index = np.abs(data_sep['longitude'] - lon).argmin()

# İstenilen koordinatlardaki rüzgar hızını alma
wind_speed_value = wind_speed_sep[:, lat_index, lon_index]
avg_sep = wind_speed_value.mean()

# Zaman serisi grafiğini çizdirme
plt.plot(data_sep['time'], wind_speed_value, label='September')

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
plt.title('Wind Speed Time Series at lat=41.35, lon=28.42')
plt.xticks(rotation=45) 

# Ortalamayı hesaplama
mean_value = wind_speed_value.mean()
mean_label = f'Mean: {mean_value:.2f} m/s'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Grafik gösterme
avg_sep = mean_value
avg_sep_value = avg_sep.item()
print(avg_sep_value)

plt.legend()
plt.show()


# In[ ]:


# Calculate wind speed
wind_speed_oct = (u10_oct**2 + v10_oct**2)**0.5

# İstenilen koordinatlara en yakın noktanın indeksini bulma
lat_index = np.abs(data_oct['latitude'] - lat).argmin()
lon_index = np.abs(data_oct['longitude'] - lon).argmin()

# İstenilen koordinatlardaki rüzgar hızını alma
wind_speed_value = wind_speed_oct[:, lat_index, lon_index]
avg_oct = wind_speed_value.mean()

# Zaman serisi grafiğini çizdirme
plt.plot(data_oct['time'], wind_speed_value, label='October')

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
plt.title('Wind Speed Time Series at lat=41.35, lon=28.42')
plt.xticks(rotation=45) 

# Ortalamayı hesaplama
mean_value = wind_speed_value.mean()
mean_label = f'Mean: {mean_value:.2f} m/s'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Grafik gösterme
avg_oct = mean_value
avg_oct_value = avg_oct.item()
print(avg_oct_value)

plt.legend()
plt.show()


# In[ ]:


# Calculate wind speed
wind_speed_nov = (u10_nov**2 + v10_nov**2)**0.5

# İstenilen koordinatlara en yakın noktanın indeksini bulma
lat_index = np.abs(data_nov['latitude'] - lat).argmin()
lon_index = np.abs(data_nov['longitude'] - lon).argmin()

# İstenilen koordinatlardaki rüzgar hızını alma
wind_speed_value = wind_speed_nov[:, lat_index, lon_index]
avg_nov = wind_speed_avg

# Zaman serisi grafiğini çizdirme
plt.plot(data_nov['time'], wind_speed_value, label='November')

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
plt.title('Wind Speed Time Series at lat=41.35, lon=28.42')
plt.xticks(rotation=45) 

# Ortalamayı hesaplama
mean_value = wind_speed_value.mean()
mean_label = f'Mean: {mean_value:.2f} m/s'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Grafik gösterme
avg_nov = mean_value
avg_nov_value = avg_nov.item()
print(avg_nov_value)

plt.legend()
plt.show()


# In[ ]:


# Calculate wind speed
wind_speed_dec = (u10_dec**2 + v10_dec**2)**0.5

# İstenilen koordinatlara en yakın noktanın indeksini bulma
lat_index = np.abs(data_dec['latitude'] - lat).argmin()
lon_index = np.abs(data_dec['longitude'] - lon).argmin()

# İstenilen koordinatlardaki rüzgar hızını alma
wind_speed_value = wind_speed_dec[:, lat_index, lon_index]
avg_dec = wind_speed_value.mean()

# Zaman serisi grafiğini çizdirme
plt.plot(data_dec['time'], wind_speed_value, label='December')

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
plt.title('Wind Speed Time Series at lat=41.35, lon=28.42')
plt.xticks(rotation=45) 

# Ortalamayı hesaplama
mean_value = wind_speed_value.mean()
mean_label = f'Mean: {mean_value:.2f} m/s'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Grafik gösterme
avg_dec = mean_value
avg_dec_value = avg_dec.item()
print(avg_dec_value)

plt.legend()
plt.show()


# In[ ]:


print('January:', "{:.2f}".format(avg_january_value))
print('February:', "{:.2f}".format(avg_february_value))
print('March:', "{:.2f}".format(avg_march_value))
print('April:', "{:.2f}".format(avg_april_value))
print('May:', "{:.2f}".format(avg_may_value))
print('June:', "{:.2f}".format(avg_june_value))
print('July:', "{:.2f}".format(avg_july_value))
print('August:', "{:.2f}".format(avg_august_value))
print('September:', "{:.2f}".format(avg_sep_value))
print('October:', "{:.2f}".format(avg_oct_value))
print('November:', "{:.2f}".format(avg_nov_value))
print('December:', "{:.2f}".format(avg_dec_value))


# In[ ]:


avg_year = (avg_january_value + avg_february_value + avg_march_value + avg_april_value + avg_may_value + avg_june_value + avg_july_value + avg_august_value + avg_sep_value + avg_oct_value + avg_nov_value + avg_dec_value) / 12
print('Avg_year:',avg_year,'m/s')


# In[ ]:


# data has 10 meter wind components
# we should calculate 90 meter wind speed
# z0 = 0.5
import math

x = 10  # Hesaplanacak değer
ln_x = math.log(x)

avg_jan_90 = avg_jan*math.log(90/0.5)/math.log(10/0.5)
print('avg_jan_90:', "{:.2f}".format(avg_jan_90), 'm/s')
avg_feb_90 = avg_feb*math.log(90/0.5)/math.log(10/0.5)
print('avg_feb_90:', "{:.2f}".format(avg_feb_90), 'm/s')
avg_march_90 = avg_march*math.log(90/0.5)/math.log(10/0.5)
print('avg_march_90:', "{:.2f}".format(avg_march_90), 'm/s')
avg_april_90 = avg_april*math.log(90/0.5)/math.log(10/0.5)
print('avg_april_90:', "{:.2f}".format(avg_april_90), 'm/s')
avg_may_90 = avg_may*math.log(90/0.5)/math.log(10/0.5)
print('avg_may_90:', "{:.2f}".format(avg_may_90), 'm/s')
avg_june_90 = avg_june*math.log(90/0.5)/math.log(10/0.5)
print('avg_june_90:', "{:.2f}".format(avg_june_90), 'm/s')
avg_july_90 = avg_july*math.log(90/0.5)/math.log(10/0.5)
print('avg_july_90:', "{:.2f}".format(avg_july_90), 'm/s')
avg_august_90 = avg_august*math.log(90/0.5)/math.log(10/0.5)
print('avg_august_90:', "{:.2f}".format(avg_august_90), 'm/s')
avg_sep_90 = avg_sep*math.log(90/0.5)/math.log(10/0.5)
print('avg_sep_90:', "{:.2f}".format(avg_sep_90), 'm/s')
avg_oct_90 = avg_oct*math.log(90/0.5)/math.log(10/0.5)
print('avg_oct_90:', "{:.2f}".format(avg_oct_90), 'm/s')
avg_nov_90 = avg_nov*math.log(90/0.5)/math.log(10/0.5)
print('avg_nov_90:', "{:.2f}".format(avg_nov_90)), 'm/s'
avg_dec_90 = avg_dec*math.log(90/0.5)/math.log(10/0.5)
print('avg_dec_90:', "{:.2f}".format(avg_dec_90), 'm/s')


# In[ ]:


z0 = 0.5  # Zemin şartı

# Hesaplanacak değerleri içeren bir liste
avg_90_values = []

# Ay isimleri listesi
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Hesaplamaları yapma
for i, avg_value in enumerate([avg_jan, avg_feb, avg_march, avg_april, avg_may, avg_june, avg_july, avg_august, avg_sep, avg_oct, avg_nov, avg_dec]):
    avg_90 = avg_value * math.log(90/z0) / math.log(10/z0)
    avg_90_values.append(avg_90)
    print('avg_{}: {:.2f} m/s'.format(months[i], avg_90))


# In[ ]:


import numpy as np

avg_90_values = [avg_jan_90, avg_feb_90, avg_march_90, avg_april_90, avg_may_90, avg_june_90, avg_july_90, avg_august_90, avg_sep_90, avg_oct_90, avg_nov_90, avg_dec_90]

avg_90_mean = np.mean(avg_90_values)
print("12 Aylık Ortalama: {:.2f} m/s".format(avg_90_mean))


# In[ ]:


E50 = avg_90_mean  # avg_90_mean değeri, 12 aylık ortalama değeri olarak kabul ediliyor

density = 1.225  # Hava yoğunluğu (kg/m^3)
power_potential = (1/2) * density * E50**3

print("Enerji Potansiyeli: {:.2f} watt/m^2".format(power_potential))


# In[ ]:


data_january_solar = xr.open_dataset('D:/solar.nc')
ssr_january = data_january_solar['ssr']

print(ssr_january)


# In[ ]:


import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# İstanbul koordinatları
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]

# İstanbul sınırlarını çizme
polygon = Polygon(zip(istanbul_lon, istanbul_lat))

# SSR değerlerini al
ssr_values = ssr_january.mean(dim='time')

# Haritayı oluşturma
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# SSR değerlerini çizdirme
ssr_plot = ax.imshow(ssr_values, extent=(data_january_solar['longitude'].values.min(),
                                         data_january_solar['longitude'].values.max(),
                                         data_january_solar['latitude'].values.min(),
                                         data_january_solar['latitude'].values.max()),
                     cmap='jet')

# İstanbul sınırlarını çizdirme
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Renk skalası ekleme
cbar = fig.colorbar(ssr_plot, ax=ax, label='Net Surface Solar Radiation (J/m^2)')

# Başlık ekleme
plt.title('Net Surface Solar Radiation in January (Near Istanbul)')

# Haritayı gösterme
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import Polygon

# Open the solar radiation dataset
data_january_solar = xr.open_dataset('D:/solar.nc')
ssr_january = data_january_solar['ssr']

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot solar radiation values
plt.pcolormesh(data_january_solar['longitude'], data_january_solar['latitude'], ssr_january.mean(dim='time'),
               transform=ccrs.PlateCarree(), cmap='jet')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.colorbar(label='Surface Net Solar Radiation (J/m^2)')

plt.title('Average Surface Net Solar Radiation in January (Near Istanbul)')

# Save the figure as a PNG file
plt.savefig('D:/energy/solar/1.png', dpi=300, bbox_inches='tight')

plt.show()

# Open the solar radiation dataset
data_january_solar = xr.open_dataset('D:/solar2.nc')
ssr_january = data_january_solar['ssr']

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot solar radiation values
plt.pcolormesh(data_january_solar['longitude'], data_january_solar['latitude'], ssr_january.mean(dim='time'),
               transform=ccrs.PlateCarree(), cmap='jet')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.colorbar(label='Surface Net Solar Radiation (J/m^2)')

plt.title('Average Surface Net Solar Radiation in February (Near Istanbul)')

# Save the figure as a PNG file
plt.savefig('D:/energy/solar/2.png', dpi=300, bbox_inches='tight')

plt.show()

# Open the solar radiation dataset
data_january_solar = xr.open_dataset('D:/solar3.nc')
ssr_january = data_january_solar['ssr']

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot solar radiation values
plt.pcolormesh(data_january_solar['longitude'], data_january_solar['latitude'], ssr_january.mean(dim='time'),
               transform=ccrs.PlateCarree(), cmap='jet')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.colorbar(label='Surface Net Solar Radiation (J/m^2)')

plt.title('Average Surface Net Solar Radiation in March (Near Istanbul)')

# Save the figure as a PNG file
plt.savefig('D:/energy/solar/3.png', dpi=300, bbox_inches='tight')

plt.show()

# Open the solar radiation dataset
data_january_solar = xr.open_dataset('D:/solar4.nc')
ssr_january = data_january_solar['ssr']

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot solar radiation values
plt.pcolormesh(data_january_solar['longitude'], data_january_solar['latitude'], ssr_january.mean(dim='time'),
               transform=ccrs.PlateCarree(), cmap='jet')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.colorbar(label='Surface Net Solar Radiation (J/m^2)')

plt.title('Average Surface Net Solar Radiation in April (Near Istanbul)')

# Save the figure as a PNG file
plt.savefig('D:/energy/solar/4.png', dpi=300, bbox_inches='tight')

plt.show()

# Open the solar radiation dataset
data_january_solar = xr.open_dataset('D:/solar5.nc')
ssr_january = data_january_solar['ssr']

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot solar radiation values
plt.pcolormesh(data_january_solar['longitude'], data_january_solar['latitude'], ssr_january.mean(dim='time'),
               transform=ccrs.PlateCarree(), cmap='jet')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.colorbar(label='Surface Net Solar Radiation (J/m^2)')

plt.title('Average Surface Net Solar Radiation in May (Near Istanbul)')

# Save the figure as a PNG file
plt.savefig('D:/energy/solar/5.png', dpi=300, bbox_inches='tight')

plt.show()

# Open the solar radiation dataset
data_january_solar = xr.open_dataset('D:/solar6.nc')
ssr_january = data_january_solar['ssr']

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot solar radiation values
plt.pcolormesh(data_january_solar['longitude'], data_january_solar['latitude'], ssr_january.mean(dim='time'),
               transform=ccrs.PlateCarree(), cmap='jet')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.colorbar(label='Surface Net Solar Radiation (J/m^2)')

plt.title('Average Surface Net Solar Radiation in June (Near Istanbul)')

# Save the figure as a PNG file
plt.savefig('D:/energy/solar/6.png', dpi=300, bbox_inches='tight')

plt.show()

# Open the solar radiation dataset
data_january_solar = xr.open_dataset('D:/solar7.nc')
ssr_january = data_january_solar['ssr']

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot solar radiation values
plt.pcolormesh(data_january_solar['longitude'], data_january_solar['latitude'], ssr_january.mean(dim='time'),
               transform=ccrs.PlateCarree(), cmap='jet')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.colorbar(label='Surface Net Solar Radiation (J/m^2)')

plt.title('Average Surface Net Solar Radiation in July (Near Istanbul)')

# Save the figure as a PNG file
plt.savefig('D:/energy/solar/7.png', dpi=300, bbox_inches='tight')

plt.show()

# Open the solar radiation dataset
data_january_solar = xr.open_dataset('D:/solar8.nc')
ssr_january = data_january_solar['ssr']

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot solar radiation values
plt.pcolormesh(data_january_solar['longitude'], data_january_solar['latitude'], ssr_january.mean(dim='time'),
               transform=ccrs.PlateCarree(), cmap='jet')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.colorbar(label='Surface Net Solar Radiation (J/m^2)')

plt.title('Average Surface Net Solar Radiation in August (Near Istanbul)')

# Save the figure as a PNG file
plt.savefig('D:/energy/solar/8.png', dpi=300, bbox_inches='tight')

plt.show()

# Open the solar radiation dataset
data_january_solar = xr.open_dataset('D:/solar9.nc')
ssr_january = data_january_solar['ssr']

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot solar radiation values
plt.pcolormesh(data_january_solar['longitude'], data_january_solar['latitude'], ssr_january.mean(dim='time'),
               transform=ccrs.PlateCarree(), cmap='jet')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.colorbar(label='Surface Net Solar Radiation (J/m^2)')

plt.title('Average Surface Net Solar Radiation in September (Near Istanbul)')

# Save the figure as a PNG file
plt.savefig('D:/energy/solar/9.png', dpi=300, bbox_inches='tight')

plt.show()

# Open the solar radiation dataset
data_january_solar = xr.open_dataset('D:/solar10.nc')
ssr_january = data_january_solar['ssr']

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot solar radiation values
plt.pcolormesh(data_january_solar['longitude'], data_january_solar['latitude'], ssr_january.mean(dim='time'),
               transform=ccrs.PlateCarree(), cmap='jet')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.colorbar(label='Surface Net Solar Radiation (J/m^2)')

plt.title('Average Surface Net Solar Radiation in October (Near Istanbul)')

# Save the figure as a PNG file
plt.savefig('D:/energy/solar/10.png', dpi=300, bbox_inches='tight')

plt.show()

# Open the solar radiation dataset
data_january_solar = xr.open_dataset('D:/solar11.nc')
ssr_january = data_january_solar['ssr']

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot solar radiation values
plt.pcolormesh(data_january_solar['longitude'], data_january_solar['latitude'], ssr_january.mean(dim='time'),
               transform=ccrs.PlateCarree(), cmap='jet')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.colorbar(label='Surface Net Solar Radiation (J/m^2)')

plt.title('Average Surface Net Solar Radiation in November (Near Istanbul)')

# Save the figure as a PNG file
plt.savefig('D:/energy/solar/11.png', dpi=300, bbox_inches='tight')

plt.show()

# Open the solar radiation dataset
data_january_solar = xr.open_dataset('D:/solar12.nc')
ssr_january = data_january_solar['ssr']

# Create the map figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines(resolution='10m')
ax.gridlines(draw_labels=True)

# Define Istanbul boundaries
istanbul_lon = [27.7, 29.8, 29.8, 27.7, 27.7]
istanbul_lat = [40.8, 40.8, 41.8, 41.8, 40.8]
polygon = Polygon(zip(istanbul_lon, istanbul_lat))
ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=2)

# Plot solar radiation values
plt.pcolormesh(data_january_solar['longitude'], data_january_solar['latitude'], ssr_january.mean(dim='time'),
               transform=ccrs.PlateCarree(), cmap='jet')

# Zoom in to the Istanbul region
ax.set_extent([27.5, 30, 40, 42])

plt.colorbar(label='Surface Net Solar Radiation (J/m^2)')

plt.title('Average Surface Net Solar Radiation in December (Near Istanbul)')

# Save the figure as a PNG file
plt.savefig('D:/energy/solar/12.png', dpi=300, bbox_inches='tight')

plt.show()


# In[ ]:


from PIL import Image

# Create GIF from the images
image_frames = []
days = np.arange(1, 13)  # 12 images from 1 to 12

for day in days:
    new_frame = Image.open(f'D:/energy/solar/{day}.png')
    image_frames.append(new_frame)

# Save as GIF
image_frames[0].save('D:/energy/solar/solar.gif', format='GIF',
                     append_images=image_frames[1:],
                     save_all=True,
                     duration=400, loop=0)


# In[ ]:


import matplotlib.pyplot as plt

# İstanbul'un latitude ve longitude koordinatları
istanbul_latitude = 41.27
istanbul_longitude = 28.77

data_january_solar = xr.open_dataset('D:/solar.nc')
ssr_january = data_january_solar['ssr']

# İstanbul'un yakın çevresindeki veri sınırlaması
data_january = data_january_solar.sel(latitude=istanbul_latitude, longitude=istanbul_longitude, method='nearest')

# SSR değerlerini al
ssr_values = data_january['ssr']

# Zaman serisi grafiğini çizdirme
plt.plot(data_january['time'], ssr_values)

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Net Surface Solar Radiation (J/m^2)')
plt.title('Net Surface Solar Radiation Time Series in January (Near Istanbul)')

# Ortalama değeri hesaplama
mean_value = ssr_values.mean()
mean_label = f'Mean: {mean_value:.2f} J/m^2'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Ortalama değeri bir değişkene atama
avg_january = mean_value.item()

# Grafik gösterme
plt.legend()
plt.xticks(rotation=45) 
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# İstanbul'un latitude ve longitude koordinatları
istanbul_latitude = 41.27
istanbul_longitude = 28.77

data_february_solar = xr.open_dataset('D:/solar2.nc')
ssr_february = data_february_solar['ssr']

# İstanbul'un yakın çevresindeki veri sınırlaması
data_february = data_february_solar.sel(latitude=istanbul_latitude, longitude=istanbul_longitude, method='nearest')

# SSR değerlerini al
ssr_values = data_february['ssr']

# Zaman serisi grafiğini çizdirme
plt.plot(data_february['time'], ssr_values)

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Net Surface Solar Radiation (J/m^2)')
plt.title('Net Surface Solar Radiation Time Series in February (Near Istanbul)')

# Ortalama değeri hesaplama
mean_value = ssr_values.mean()
mean_label = f'Mean: {mean_value:.2f} J/m^2'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Ortalama değeri bir değişkene atama
avg_february = mean_value.item()

# Grafik gösterme
plt.legend()
plt.xticks(rotation=45) 
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# İstanbul'un latitude ve longitude koordinatları
istanbul_latitude = 41.27
istanbul_longitude = 28.77

data_march_solar = xr.open_dataset('D:/solar3.nc')
ssr_march = data_march_solar['ssr']

# İstanbul'un yakın çevresindeki veri sınırlaması
data_march = data_march_solar.sel(latitude=istanbul_latitude, longitude=istanbul_longitude, method='nearest')

# SSR değerlerini al
ssr_values = data_march['ssr']

# Zaman serisi grafiğini çizdirme
plt.plot(data_march['time'], ssr_values)

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Net Surface Solar Radiation (J/m^2)')
plt.title('Net Surface Solar Radiation Time Series in March (Near Istanbul)')

# Ortalama değeri hesaplama
mean_value = ssr_values.mean()
mean_label = f'Mean: {mean_value:.2f} J/m^2'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Ortalama değeri bir değişkene atama
avg_march = mean_value.item()

# Grafik gösterme
plt.legend()
plt.xticks(rotation=45) 
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# İstanbul'un latitude ve longitude koordinatları
istanbul_latitude = 41.27
istanbul_longitude = 28.77

data_april_solar = xr.open_dataset('D:/solar4.nc')
ssr_april = data_april_solar['ssr']

# İstanbul'un yakın çevresindeki veri sınırlaması
data_april = data_april_solar.sel(latitude=istanbul_latitude, longitude=istanbul_longitude, method='nearest')

# SSR değerlerini al
ssr_values = data_april['ssr']

# Zaman serisi grafiğini çizdirme
plt.plot(data_april['time'], ssr_values)

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Net Surface Solar Radiation (J/m^2)')
plt.title('Net Surface Solar Radiation Time Series in April (Near Istanbul)')

# Ortalama değeri hesaplama
mean_value = ssr_values.mean()
mean_label = f'Mean: {mean_value:.2f} J/m^2'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Ortalama değeri bir değişkene atama
avg_april = mean_value.item()

# Grafik gösterme
plt.legend()
plt.xticks(rotation=45) 
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# İstanbul'un latitude ve longitude koordinatları
istanbul_latitude = 41.27
istanbul_longitude = 28.77

data_may_solar = xr.open_dataset('D:/solar5.nc')
ssr_may = data_may_solar['ssr']

# İstanbul'un yakın çevresindeki veri sınırlaması
data_may = data_may_solar.sel(latitude=istanbul_latitude, longitude=istanbul_longitude, method='nearest')

# SSR değerlerini al
ssr_values = data_may['ssr']

# Zaman serisi grafiğini çizdirme
plt.plot(data_may['time'], ssr_values)

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Net Surface Solar Radiation (J/m^2)')
plt.title('Net Surface Solar Radiation Time Series in May (Near Istanbul)')

# Ortalama değeri hesaplama
mean_value = ssr_values.mean()
mean_label = f'Mean: {mean_value:.2f} J/m^2'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Ortalama değeri bir değişkene atama
avg_may = mean_value.item()

# Grafik gösterme
plt.legend()
plt.xticks(rotation=45) 
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# İstanbul'un latitude ve longitude koordinatları
istanbul_latitude = 41.27
istanbul_longitude = 28.77

data_june_solar = xr.open_dataset('D:/solar6.nc')
ssr_june = data_june_solar['ssr']

# İstanbul'un yakın çevresindeki veri sınırlaması
data_june = data_june_solar.sel(latitude=istanbul_latitude, longitude=istanbul_longitude, method='nearest')

# SSR değerlerini al
ssr_values = data_june['ssr']

# Zaman serisi grafiğini çizdirme
plt.plot(data_june['time'], ssr_values)

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Net Surface Solar Radiation (J/m^2)')
plt.title('Net Surface Solar Radiation Time Series in June (Near Istanbul)')

# Ortalama değeri hesaplama
mean_value = ssr_values.mean()
mean_label = f'Mean: {mean_value:.2f} J/m^2'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Ortalama değeri bir değişkene atama
avg_june = mean_value.item()

# Grafik gösterme
plt.legend()
plt.xticks(rotation=45) 
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# İstanbul'un latitude ve longitude koordinatları
istanbul_latitude = 41.27
istanbul_longitude = 28.77

data_july_solar = xr.open_dataset('D:/solar7.nc')
ssr_july = data_july_solar['ssr']

# İstanbul'un yakın çevresindeki veri sınırlaması
data_july = data_july_solar.sel(latitude=istanbul_latitude, longitude=istanbul_longitude, method='nearest')

# SSR değerlerini al
ssr_values = data_july['ssr']

# Zaman serisi grafiğini çizdirme
plt.plot(data_july['time'], ssr_values)

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Net Surface Solar Radiation (J/m^2)')
plt.title('Net Surface Solar Radiation Time Series in July (Near Istanbul)')

# Ortalama değeri hesaplama
mean_value = ssr_values.mean()
mean_label = f'Mean: {mean_value:.2f} J/m^2'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Ortalama değeri bir değişkene atama
avg_july = mean_value.item()

# Grafik gösterme
plt.legend()
plt.xticks(rotation=45) 
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# İstanbul'un latitude ve longitude koordinatları
istanbul_latitude = 41.27
istanbul_longitude = 28.77

data_august_solar = xr.open_dataset('D:/solar8.nc')
ssr_august = data_august_solar['ssr']

# İstanbul'un yakın çevresindeki veri sınırlaması
data_august = data_august_solar.sel(latitude=istanbul_latitude, longitude=istanbul_longitude, method='nearest')

# SSR değerlerini al
ssr_values = data_august['ssr']

# Zaman serisi grafiğini çizdirme
plt.plot(data_august['time'], ssr_values)

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Net Surface Solar Radiation (J/m^2)')
plt.title('Net Surface Solar Radiation Time Series in August (Near Istanbul)')

# Ortalama değeri hesaplama
mean_value = ssr_values.mean()
mean_label = f'Mean: {mean_value:.2f} J/m^2'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Ortalama değeri bir değişkene atama
avg_august = mean_value.item()

# Grafik gösterme
plt.legend()
plt.xticks(rotation=45) 
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# İstanbul'un latitude ve longitude koordinatları
istanbul_latitude = 41.27
istanbul_longitude = 28.77

data_september_solar = xr.open_dataset('D:/solar9.nc')
ssr_september = data_september_solar['ssr']

# İstanbul'un yakın çevresindeki veri sınırlaması
data_september = data_september_solar.sel(latitude=istanbul_latitude, longitude=istanbul_longitude, method='nearest')

# SSR değerlerini al
ssr_values = data_september['ssr']

# Zaman serisi grafiğini çizdirme
plt.plot(data_september['time'], ssr_values)

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Net Surface Solar Radiation (J/m^2)')
plt.title('Net Surface Solar Radiation Time Series in September (Near Istanbul)')

# Ortalama değeri hesaplama
mean_value = ssr_values.mean()
mean_label = f'Mean: {mean_value:.2f} J/m^2'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Ortalama değeri bir değişkene atama
avg_september = mean_value.item()

# Grafik gösterme
plt.legend()
plt.xticks(rotation=45) 
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# İstanbul'un latitude ve longitude koordinatları
istanbul_latitude = 41.27
istanbul_longitude = 28.77

data_october_solar = xr.open_dataset('D:/solar10.nc')
ssr_october = data_october_solar['ssr']

# İstanbul'un yakın çevresindeki veri sınırlaması
data_october = data_october_solar.sel(latitude=istanbul_latitude, longitude=istanbul_longitude, method='nearest')

# SSR değerlerini al
ssr_values = data_october['ssr']

# Zaman serisi grafiğini çizdirme
plt.plot(data_october['time'], ssr_values)

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Net Surface Solar Radiation (J/m^2)')
plt.title('Net Surface Solar Radiation Time Series in October (Near Istanbul)')

# Ortalama değeri hesaplama
mean_value = ssr_values.mean()
mean_label = f'Mean: {mean_value:.2f} J/m^2'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Ortalama değeri bir değişkene atama
avg_october = mean_value.item()

# Grafik gösterme
plt.legend()
plt.xticks(rotation=45) 
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# İstanbul'un latitude ve longitude koordinatları
istanbul_latitude = 41.27
istanbul_longitude = 28.77

data_november_solar = xr.open_dataset('D:/solar11.nc')
ssr_november = data_november_solar['ssr']

# İstanbul'un yakın çevresindeki veri sınırlaması
data_november = data_november_solar.sel(latitude=istanbul_latitude, longitude=istanbul_longitude, method='nearest')

# SSR değerlerini al
ssr_values = data_november['ssr']

# Zaman serisi grafiğini çizdirme
plt.plot(data_november['time'], ssr_values)

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Net Surface Solar Radiation (J/m^2)')
plt.title('Net Surface Solar Radiation Time Series in November (Near Istanbul)')

# Ortalama değeri hesaplama
mean_value = ssr_values.mean()
mean_label = f'Mean: {mean_value:.2f} J/m^2'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Ortalama değeri bir değişkene atama
avg_november = mean_value.item()

# Grafik gösterme
plt.legend()
plt.xticks(rotation=45) 
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# İstanbul'un latitude ve longitude koordinatları
istanbul_latitude = 41.27
istanbul_longitude = 28.77

data_december_solar = xr.open_dataset('D:/solar12.nc')
ssr_december = data_december_solar['ssr']

# İstanbul'un yakın çevresindeki veri sınırlaması
data_december = data_december_solar.sel(latitude=istanbul_latitude, longitude=istanbul_longitude, method='nearest')

# SSR değerlerini al
ssr_values = data_december['ssr']

# Zaman serisi grafiğini çizdirme
plt.plot(data_december['time'], ssr_values)

# Eksen ve başlık ayarları
plt.xlabel('Time')
plt.ylabel('Net Surface Solar Radiation (J/m^2)')
plt.title('Net Surface Solar Radiation Time Series in December (Near Istanbul)')

# Ortalama değeri hesaplama
mean_value = ssr_values.mean()
mean_label = f'Mean: {mean_value:.2f} J/m^2'
plt.axhline(mean_value, color='red', linestyle='--', label=mean_label)

# Ortalama değeri bir değişkene atama
avg_december = mean_value.item()
print(avg_december)

# Grafik gösterme
plt.legend()
plt.xticks(rotation=45) 
plt.show()


# In[ ]:


# Dönüşüm faktörü (Joule'den kilovatsaate)
conversion_factor = 3.6 * 10**6

# Değerleri kWh/m^2 birimine dönüştürme
avg_january_kWh_per_m2 = avg_january / conversion_factor
avg_february_kWh_per_m2 = avg_february / conversion_factor
avg_march_kWh_per_m2 = avg_march / conversion_factor
avg_april_kWh_per_m2 = avg_april / conversion_factor
avg_may_kWh_per_m2 = avg_may / conversion_factor
avg_june_kWh_per_m2 = avg_june / conversion_factor
avg_july_kWh_per_m2 = avg_july / conversion_factor
avg_august_kWh_per_m2 = avg_august / conversion_factor
avg_september_kWh_per_m2 = avg_september / conversion_factor
avg_october_kWh_per_m2 = avg_october / conversion_factor
avg_november_kWh_per_m2 = avg_november / conversion_factor
avg_december_kWh_per_m2 = avg_december / conversion_factor

# Sonuçları ekrana yazdırma
print(f"January: {avg_january_kWh_per_m2:.2f} kWh/m^2")
print(f"February: {avg_february_kWh_per_m2:.2f} kWh/m^2")
print(f"March: {avg_march_kWh_per_m2:.2f} kWh/m^2")
print(f"April: {avg_april_kWh_per_m2:.2f} kWh/m^2")
print(f"May: {avg_may_kWh_per_m2:.2f} kWh/m^2")
print(f"June: {avg_june_kWh_per_m2:.2f} kWh/m^2")
print(f"July: {avg_july_kWh_per_m2:.2f} kWh/m^2")
print(f"August: {avg_august_kWh_per_m2:.2f} kWh/m^2")
print(f"September: {avg_september_kWh_per_m2:.2f} kWh/m^2")
print(f"October: {avg_october_kWh_per_m2:.2f} kWh/m^2")
print(f"November: {avg_november_kWh_per_m2:.2f} kWh/m^2")
print(f"December: {avg_december_kWh_per_m2:.2f} kWh/m^2")


# In[ ]:


# Toplamı alınacak ayların değişkenlerini listeleyin
monthly_averages = [avg_january, avg_february, avg_march, avg_april, avg_may, avg_june,
                    avg_july, avg_august, avg_september, avg_october, avg_november, avg_december]

# Yıllık ortalama hesaplaması
annual_average = sum(monthly_averages) / len(monthly_averages)

# Sonucu yazdırma
print(f"Yıllık Ortalama: {annual_average:.2f} J/m^2")

# 1 kWh = 3.6 × 10^6 J
# kWh/m^2 = (J/m^2) / (3.6 × 10^6)
# Verilen değer (J/m^2)
joules_per_square_meter = 8679123.71

# Dönüşüm faktörü (Joule'den kilovatsaate)
conversion_factor = 3.6 * 10**6

# Dönüştürülen değer (kWh/m^2)
kilowatt_hours_per_square_meter = joules_per_square_meter / conversion_factor

# Sonucu ekrana yazdırma
print(f"Değer: {kilowatt_hours_per_square_meter:.2f} kWh/m^2")


# In[ ]:


import numpy as np

# Veri kümesindeki enlem ve boylam koordinatlarına erişin
lats = data_january_solar['latitude']
lons = data_january_solar['longitude']

# İstenilen latitude ve longitude değerlerini belirleyin
target_lat = 41.27
target_lon = 28.77

# En yakın grid noktalarını bulun
nearest_lat = np.abs(lats - target_lat).argmin()
nearest_lon = np.abs(lons - target_lon).argmin()

# Kare derece alanını hesaplayın
lat_diff = lats[nearest_lat + 1] - lats[nearest_lat]
lon_diff = lons[nearest_lon + 1] - lons[nearest_lon]
grid_area_deg = lat_diff * lon_diff

# Kare derece alanını kilometrekareye dönüştürün (örneğin, 1 kare derece yaklaşık olarak 111 km x 111 km)
# Bu değer, kullanılan projeksiyon ve bölgeye bağlı olarak değişebilir
grid_area_km2 = abs(grid_area_deg) * (111 * 111)

print(f"Latitude: {lats[nearest_lat]:.2f}, Longitude: {lons[nearest_lon]:.2f}")
print(f"Kapsadığı Alan: {grid_area_km2:.2f} kilometrekare")

# Kare derece alanını kilometrekareye dönüştürün (örneğin, 1 kare derece yaklaşık olarak 111 km x 111 km)
# Bu değer, kullanılan projeksiyon ve bölgeye bağlı olarak değişebilir
grid_area_km2 = abs(grid_area_deg) * (111 * 111)

# Kare kilometre alanını metrekareye dönüştürün
grid_area_m2 = grid_area_km2 * 1e6

print(f"Latitude: {lats[nearest_lat]:.2f}, Longitude: {lons[nearest_lon]:.2f}")
print(f"Kapsadığı Alan: {grid_area_m2:.2f} metrekare")


# In[ ]:


# a. Basit Formül:
# Solar Energy Potential = Rn * A

Solar_Energy_Potential = kilowatt_hours_per_square_meter*(123208585.84)
print(f"Solar Energy Potential: {Solar_Energy_Potential:.2f} Watt")


# In[ ]:


Solar_Energy_Potential_MW = Solar_Energy_Potential / 1000000
print(f"Solar Energy Potential: {Solar_Energy_Potential_MW:.2f} MW")

