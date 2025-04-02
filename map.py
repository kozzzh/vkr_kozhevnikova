import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from streamlit.components.v1 import html
import psycopg2

#подключение к постгре
def get_data(query):
    #подключение
    conn = None
    #установление подключения
    conn = psycopg2.connect(host="localhost", database="postgres", user="postgres", password="1602", port="5432")
    #курсор для хранения последнего оператора скл
    cur = conn.cursor()
    #выполнение подключения
    cur.execute(query)
    #получаем все строки
    data = cur.fetchall()
    cur.close()
    return data
    #если соединение установлено (не none), то завершаем подключение
    if conn is not None:
        conn.close()

# точки входа
# определение запроса
entry_query = "select entry_id, entry_name, latitude, longitude, type from points_of_entry"
#данные
entry_data = get_data(entry_query)
df_entry = pd.DataFrame(entry_data, columns=['entry_id', 'entry_name', 'latitude', 'longitude', 'type'])

#точки хранения
storage_query = "select storage_id, storage_name, latitude, longitude from points_of_storage"
storage_data = get_data(storage_query)
df_storage = pd.DataFrame(storage_data, columns=['storage_id', 'storage_name', 'latitude', 'longitude'])

#точки потребления
consumption_query = "select consumption_id, consumption_name, latitude, longitude from points_of_consumption"
consumption_data = get_data(consumption_query)
df_consumption = pd.DataFrame(consumption_data, columns=['consumption_id', 'consumption_name', 'latitude', 'longitude'])

#транспортные пути
routes_query = "select route_id, start_entry_id, start_storage_id, end_storage_id, end_consumption_id, route_type from transport_routes"
routes_data = get_data(routes_query)
df_routes = pd.DataFrame(routes_data, columns=['route_id', 'start_entry_id', 'start_storage_id', 'end_storage_id', 'end_consumption_id', 'route_type'])

#транспортные пути расширенные
query = """
select
tr.route_id,
e.entry_name as start_point_name,
c.consumption_name as end_point_name,
tr.route_type,
e.latitude as start_latitude,
e.longitude as start_longitude,
c.latitude as end_latitude,
c.longitude as end_longitude
from transport_routes tr
left join points_of_entry e on tr.start_entry_id = e.entry_id
left join points_of_consumption c on tr.end_consumption_id = c.consumption_id;
"""
data = get_data(query)

#датасет с наименованиями точек
df_names = pd.DataFrame(data, columns=['route_id', 'start_point_name', 'end_point_name', 'route_type', 'start_latitude', 'start_longitude', 'end_latitude', 'end_longitude'])

#объединение df_routes и df_names по route_id
df_routes = pd.merge(df_routes, df_names, on='route_id', how='left')

# получение координат по ID и типу точки
def get_coordinates(row):
    if pd.notna(row['start_entry_id']):
        point = df_entry[df_entry['entry_id'] == row['start_entry_id']]
        if not point.empty:
            return point['latitude'].iloc[0], point['longitude'].iloc[0]
    if pd.notna(row['start_storage_id']):
        point = df_storage[df_storage['storage_id'] == row['start_storage_id']]
        if not point.empty:
            return point['latitude'].iloc[0], point['longitude'].iloc[0]
    if pd.notna(row['end_storage_id']):
        point = df_storage[df_storage['storage_id'] == row['end_storage_id']]
        if not point.empty:
            return point['latitude'].iloc[0], point['longitude'].iloc[0]
    if pd.notna(row['end_consumption_id']):
        point = df_consumption[df_consumption['consumption_id'] == row['end_consumption_id']]
        if not point.empty:
            return point['latitude'].iloc[0], point['longitude'].iloc[0]
    return None, None

#Streamlit
st.title("Интерактивная карта ХКГМ")

#центр карты
center_latitude = 71.0
center_longitude = 67.0

#карта
m = folium.Map(location=[center_latitude, center_longitude], zoom_start=6)

#маркеры ТВ
for index, row in df_entry.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"<b>{row['entry_name']} (Точка входа)</b><br>Долгота:{row['latitude']}<br>Широта:{row['longitude']}",
        icon=folium.Icon(color="green", icon="home"),
    ).add_to(m)

#маркеры ТХ
for index, row in df_storage.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"<b>{row['storage_name']} (Точка хранения)</b><br>Долгота:{row['latitude']}<br>Широта:{row['longitude']}",
        icon=folium.Icon(color="blue", icon="home"),
    ).add_to(m)

#маркеры ТП
for index, row in df_consumption.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"<b>{row['consumption_name']} (Точка потребления)</b><br>Долгота:{row['latitude']}<br>Широта:{row['longitude']}",
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(m)

# дуги
routes =[]
for index, row in df_routes.iterrows():
    #координаты и наименование начточки
    start_lat = None
    start_lon = None
    start_name = None
    if pd.notna(row['start_entry_id']):
        start_point = df_entry[df_entry['entry_id'] == row['start_entry_id']]
        if not start_point.empty:
            start_lat = start_point['latitude'].iloc[0]
            start_lon = start_point['longitude'].iloc[0]
            start_name = start_point['entry_name'].iloc[0]
    elif pd.notna(row['start_storage_id']):
        start_point = df_storage[df_storage['storage_id'] == row['start_storage_id']]
        if not start_point.empty:
            start_lat = start_point['latitude'].iloc[0]
            start_lon = start_point['longitude'].iloc[0]
            start_name = start_point['storage_name'].iloc[0]

    #координаты и наименование кон точки
    end_lat = None
    end_lon = None
    end_name = None

    if pd.notna(row['end_storage_id']):
        end_point = df_storage[df_storage['storage_id'] == row['end_storage_id']]
        if not end_point.empty:
            end_lat = end_point['latitude'].iloc[0]
            end_lon = end_point['longitude'].iloc[0]
            end_name = end_point['storage_name'].iloc[0]
    elif pd.notna(row['end_consumption_id']):
        end_point = df_consumption[df_consumption['consumption_id'] == row['end_consumption_id']]
        if not end_point.empty:
            end_lat = end_point['latitude'].iloc[0]
            end_lon = end_point['longitude'].iloc[0]
            end_name = end_point['consumption_name'].iloc[0]

    # рисуем пути
    if start_lat is not None and start_lon is not None and end_lat is not None and end_lon is not None:
        #вспомогательный словарь для комментариев в тултипах
        route = {
            'route_id': row['route_id'],
            'start_lat': start_lat,
            'start_lon': start_lon,
            'start_name': start_name,
            'end_lat': end_lat,
            'end_lon': end_lon,
            'end_name': end_name
        }
        routes.append(route)

        folium.PolyLine(
            locations=[[start_lat, start_lon], [end_lat, end_lon]],
            color="gray",
            weight=2,
            opacity=1,
            smooth_factor=0,
            tooltip=f"<b>Номер маршрута:</b> {route['route_id']}<br><b>Начальная точка:</b> {route['start_name']}<br><b>Конечная точка:</b> {route['end_name']}"
        ).add_to(m)

# Отображение карты в Streamlit
folium_static(m, width=1500, height=800)