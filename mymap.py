import pandas as pd
import numpy as np
import psycopg2
import urllib.parse
import os
import pulp
from pulp import *
from datetime import datetime, date
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import folium
from streamlit_folium import folium_static
from shapely.geometry import Point, LineString

#подключение к постгре
@st.cache_data
def get_data(query):
    conn = None
    try:
        # Получите DATABASE_URL из переменных окружения
        database_url = os.environ.get("DATABASE_URL")

        # Подключитесь к базе данных
        conn = psycopg2.connect(database_url, sslmode='require')  # Прямое подключение

        cur = conn.cursor()
        cur.execute(query)
        data = cur.fetchall()
        cur.close()
        return data
    except psycopg2.Error as e:
        st.error(f"Ошибка при подключении к базе данных: {e}")  # Используйте st.error для отображения ошибок в Streamlit
        return None
    finally:
        if conn is not None:
            conn.close()


# точки входа
# определение запроса
entry_query = "select entry_id, entry_name, latitude, longitude, type from points_of_entry"
#данные
entry_data = get_data(entry_query)
df_entry = pd.DataFrame(entry_data, columns=['entry_id', 'entry_name', 'latitude', 'longitude', 'type'])
df_entry['capacity'] = [500000, 2050000, 200000, 30000, 210000, 1500000, 185000, 300000, 2050000]
df_entry = df_entry[~df_entry['entry_id'].isin([4, 5])]

#точки хранения
storage_query = "select storage_id, storage_name, latitude, longitude from points_of_storage"
storage_data = get_data(storage_query)
df_storage = pd.DataFrame(storage_data, columns=['storage_id', 'storage_name', 'latitude', 'longitude'])
df_storage['capacity'] = [2050000, 1500000, 300000]

#точки потребления
consumption_query = "select consumption_id, consumption_name, latitude, longitude from points_of_consumption"
consumption_data = get_data(consumption_query)
df_consumption = pd.DataFrame(consumption_data, columns=['consumption_id', 'consumption_name', 'latitude', 'longitude'])
df_consumption['capacity'] = [500000]

#типы грузов
cargo_query = "select cargo_id, cargo_type, unit from cargo_types"
cargo_data = get_data(cargo_query)
df_cargo = pd.DataFrame(cargo_data, columns=['cargo_id', 'cargo_type', 'unit'])

#транспортные пути
# Функция для получения данных о маршрутах с учетом даты
def get_transport_routes(current_date: date):
    query = f"""
        SELECT route_id, start_entry_id, start_storage_id, end_storage_id, end_consumption_id,
               route_type, cargo_volume, capacity, transportation_cost, allowed_transport_vehicles,
               start_date, end_date
        FROM transport_routes
        WHERE start_date <= '{current_date}' AND (end_date IS NULL OR end_date >= '{current_date}')
    """
    transport_data = get_data(query)
    df_routes = pd.DataFrame(transport_data, columns=[
        'route_id', 'start_entry_id', 'start_storage_id', 'end_storage_id', 'end_consumption_id',
        'route_type', 'cargo_volume', 'capacity', 'transportation_cost', 'allowed_transport_vehicles',
        'start_date', 'end_date'
    ])
    return df_routes

# Получаем текущую дату из Streamlit (можно использовать слайдер или выбор даты)
current_date = st.date_input("Выберите дату", value=datetime.today())

# Получаем данные о маршрутах на выбранную дату
df_routes = get_transport_routes(current_date)

# Переименовываем столбцы для удобства
df_entry = df_entry.rename(columns={'entry_id': 'node_id', 'entry_name': 'node_name'})
df_storage = df_storage.rename(columns={'storage_id': 'node_id', 'storage_name': 'node_name'})
df_consumption = df_consumption.rename(columns={'consumption_id': 'node_id', 'consumption_name': 'node_name'})

# Объединяем данные о узлах
frames = [df_entry, df_storage, df_consumption]
df_nodes = pd.concat(frames, ignore_index=True)

# Создаем списки для дуг (с учетом выбранной даты и df_routes)
A1 = []  # Прямые маршруты от ТВ до ТХ
A2 = []  # Маршруты от ТХ до ТП

# Проверяем, что df_routes не пуст
if not df_routes.empty:
    for index, row in df_routes.iterrows():
        start_node = row['start_entry_id'] if pd.notna(row['start_entry_id']) else row['start_storage_id']
        end_node = row['end_consumption_id'] if pd.notna(row['end_consumption_id']) else row['end_storage_id']

        # Проверка на None (или np.nan) before converting to int
        if not pd.isna(start_node) and not pd.isna(end_node):
            start_node = int(start_node)
            end_node = int(end_node)

            if not pd.isna(row['start_entry_id']):
                A1.append((start_node, end_node))
            else:
                A2.append((start_node, end_node))

# Типы грузов
K = df_cargo['cargo_type'].tolist()

# Определяем тип узла по приоритету: потребление > хранение > вход
if df_nodes is not None:
    # Сначала все узлы считаем точками входа
    df_nodes['node_type'] = 'Точка входа'

    # Если есть точка потребления, перезаписываем тип
    if df_consumption is not None:
        consumption_nodes = df_consumption['node_id'].tolist()
        df_nodes.loc[df_nodes['node_id'].isin(consumption_nodes), 'node_type'] = 'Точка потребления'

    # Если есть точка хранения, и она не точка потребления, перезаписываем тип
    if df_storage is not None:
        storage_nodes = df_storage['node_id'].tolist()
        df_nodes.loc[(df_nodes['node_id'].isin(storage_nodes)) & (df_nodes['node_type'] != 'Точка потребления'), 'node_type'] = 'Точка хранения'

# словарь со стоимостями доставки
costs = {}
# Use df_routes here
for index, row in df_routes.iterrows():
    start_node = row['start_entry_id'] if pd.notna(row['start_entry_id']) else row['start_storage_id']
    end_node = row['end_consumption_id'] if pd.notna(row['end_consumption_id']) else row['end_storage_id']
    transportation_cost = row['transportation_cost']
    cargo_volume = row['cargo_volume']

    if not pd.isna(start_node) and not pd.isna(end_node):
        start_node = int(start_node)
        end_node = int(end_node)

        for k in K:
            costs[(start_node, end_node, k)] = transportation_cost / cargo_volume if cargo_volume > 0 else 0

#истоки (точки входа) и стоки (точки потребления)
sources = {}
sinks = {}
for k in K:
    # пусть каждый тип груза может быть доставлен с любого ТВ !!!(ДОБАВИТЬ РАСПРЕДЕЛЕНИЕ ГРУЗОВ ПО ТВ)
    sources[k] = df_entry['node_id'].tolist()
    sinks[k] = df_consumption['node_id'].tolist()

# узлы с пропускные способности узлов
node_capacities = {}
for index, row in df_nodes.iterrows():
    node_id = row['node_id']
    capacity = row['capacity']
    node_capacities[node_id] = capacity

#пропусные способности маршрутов от точки входа
route_capacities1 = {}
# Use df_routes here
for index, row in df_routes.iterrows():
    start_entry_id = row['start_entry_id']
    end_storage_id = row['end_storage_id']
    end_consumption_id = row['end_consumption_id']
    capacity = row['capacity']

    # добавляем маршрут, если он существует (от точки входа до точки хранения)
    if not pd.isna(start_entry_id) and not pd.isna(end_storage_id):
        route_capacities1[(start_entry_id, end_storage_id)] = capacity

    # добавляем маршрут, если он существует (от точки входа до точки потребления)
    elif not pd.isna(start_entry_id) and not pd.isna(end_consumption_id):
        route_capacities1[(start_entry_id, end_consumption_id)] = capacity

#маршруты от точки хранения
route_capacities2 = {}
# Use df_routes here
for index, row in df_routes.iterrows():
    start_storage_id = row['start_storage_id']
    end_consumption_id = row['end_consumption_id']
    capacity = row['capacity']

    if not pd.isna(start_storage_id) and not pd.isna(end_consumption_id):
        route_capacities2[(start_storage_id, end_consumption_id)] = capacity

# Все уникальные узлы в сети из А1 и А2
nodes = set()
for (i, j) in A1 + A2:
    nodes.add(i)
    nodes.add(j)

# ------------------------------------------------------------------------------
# Модель максимизации потока
# ------------------------------------------------------------------------------

prob_max_flow = LpProblem("MaxFlow", LpMaximize)

# Переменные объема потока
f1_mf = LpVariable.dicts("f1_mf", [(i, j, k) for (i, j) in A1 for k in K], lowBound=0, cat='Continuous')
f2_mf = LpVariable.dicts("f2_mf", [(i, j, k) for (i, j) in A2 for k in K], lowBound=0, cat='Continuous')

# Целевая функция
prob_max_flow += lpSum([f1_mf[(i, j, k)] for (i, j) in A1 for k in K]) + lpSum([f2_mf[(i, j, k)] for (i, j) in A2 for k in K]), "Total Flow"

# ------------------------------------------------------------------------------
# Ограничения модели максимизации потока
# ------------------------------------------------------------------------------

# Ограничение 1: Величина потока по дуге не превосходит пропускную способность этой дуги

constraint_counter1 = 0 #счетчик ограничений
# Для маршрутов от ТВ
for (i, j) in A1:
    is_storage = j in df_storage['node_id'].values
    is_consumption = j in df_consumption['node_id'].values

    #если конечная точка - ТХ, то пишем ограничение о том, что суммарный поток по дуге не выше пропускной сопособности этой дуги
    if is_storage:
        #<= route_capacities1.get((i, j) - возвращаем ПС дуги, если она есть, иначе - 0
        prob_max_flow += lpSum([f1_mf[(i, j, k)] for k in K]) <= route_capacities1.get((i, j), 0), f"C_A1_ES_{i}_{j}_{constraint_counter1}"
        constraint_counter1 += 1 #увеличиваем счетчик
    #если конечная точка - ТП, то пишем ограничение о том, что суммарный поток по дуге не выше пропускной сопособности этой дуги
    elif is_consumption:
        prob_max_flow += lpSum([f1_mf[(i, j, k)] for k in K]) <= route_capacities1.get((i, j), 0), f"C_A1_EC_{i}_{j}_{constraint_counter1}"
        constraint_counter1 += 1 #увеличиваем счетчик

# Также для маршрутов от тх
for (i, j) in A2:
    prob_max_flow += lpSum([f2_mf[(i, j, k)] for k in K]) <= route_capacities2.get((i, j), 0), f"C_A2_{i}_{j}_{constraint_counter1}"
    constraint_counter1 += 1 #увеличиваем счетчик

# Ограничение 2: Поток, входящий в точку хранения (не в sources или sinks) больше или равен исходящему из этой точки потоку

constraint_counter2 = 0 #счетчик ограничений
for node in nodes:
    if node not in [item for sublist in sources.values() for item in sublist] and node not in [item for sublist in sinks.values() for item in sublist]:
        for k in K:
            prob_max_flow += lpSum([f1_mf[(i, node, k)] for (i, node) in A1 if (i, node, k) in f1_mf]) + lpSum([f2_mf[(i, node, k)] for (i, node) in A2 if (i, node, k) in f2_mf]) >= \
                            lpSum([f1_mf[(node, j, k)] for (node, j) in A1 if (node, j, k) in f1_mf]) + lpSum([f2_mf[(node, j, k)] for (node, j) in A2 if (node, j, k) in f2_mf]), f"Flow_Balance_{node}_{k}_{constraint_counter2}"
            constraint_counter2 += 1

# Ограничение 3: Суммарный поток, проходящий через узел, не превосходит пропускную способность этого узла

constraint_counter3 = 0 #счетчик ограничений
for node in df_nodes['node_id']:
    for k in K:
        if node in node_capacities:
            prob_max_flow += lpSum([f1_mf[(i, node, k)] for (i, node) in A1 if (i, node, k) in f1_mf]) + lpSum([f2_mf[(i, node, k)] for (i, node) in A2 if (i, node, k) in f2_mf]) + \
                            lpSum([f1_mf[(node, j, k)] for (node, j) in A1 if (node, j, k) in f1_mf]) + lpSum([f2_mf[(node, j, k)] for (node, j) in A2 if (node, j, k) in f2_mf]) <= node_capacities[node], f"Node_Capacity_{node}_{k}_{constraint_counter3}"
            constraint_counter3 += 1
# Ограничение 4: Сумма всего потока, исходящего из источника равно пропускной способности этого источника

for k in K:
    for s in sources[k]:
        total_outgoing_flow = lpSum([f1_mf[(s, j, k)] for (s, j) in A1 if s == s and (s, j, k) in f1_mf]) + \
                              lpSum([f2_mf[(s, j, k)] for (s, j) in A2 if s == s and (s, j, k) in f2_mf])

        if s in node_capacities:
            prob_max_flow += total_outgoing_flow <= node_capacities[s], f"Source_Capacity_{s}_{k}"

# ------------------------------------------------------------------------------
# Решение задачи максимизации потока
# ------------------------------------------------------------------------------

prob_max_flow.solve()
optimal_flow = value(prob_max_flow.objective)

# ------------------------------------------------------------------------------
# Модель минимизации затрат
# ------------------------------------------------------------------------------

prob_min_cost = LpProblem("MinCostGivenMaxFlow", LpMinimize)

# Переменные затрат
f1_mc = LpVariable.dicts("f1_mc", [(i, j, k) for (i, j) in A1 for k in K], lowBound=0, cat='Continuous')
f2_mc = LpVariable.dicts("f2_mc", [(i, j, k) for (i, j) in A2 for k in K], lowBound=0, cat='Continuous')

# Целевая функция
prob_min_cost += lpSum([costs[(i, j, k)] * f1_mc[(i, j, k)] for (i, j) in A1 for k in K]) + lpSum([costs[(i, j, k)] * f2_mc[(i, j, k)] for (i, j) in A2 for k in K]), "Total Cost"

# ------------------------------------------------------------------------------
# Ограничения модели минимизации затрат
# ------------------------------------------------------------------------------

# Ограничение 1: Величина потока по дуге не превосходит пропускную способность этой дуги

constraint_counter4 = 0 # счетчик ограничений
# Для маршрутов от ТВ
for (i, j) in A1:
    is_storage = j in df_storage['node_id'].values
    is_consumption = j in df_consumption['node_id'].values

    #если конечная точка - ТХ, то пишем ограничение о том, что суммарный поток по дуге не выше пропускной сопособности этой дуги
    if is_storage:
        #<= route_capacities1.get((i, j) - возвращаем ПС дуги, если она есть, иначе - 0
        prob_min_cost += lpSum([f1_mc[(i, j, k)] for k in K]) <= route_capacities1.get((i, j), 0), f"C_A1_ES_{i}_{j}_{constraint_counter4}"
        constraint_counter4 += 1
    #если конечная точка - ТП, то пишем ограничение о том, что суммарный поток по дуге не выше пропускной сопособности этой дуги
    elif is_consumption:
        prob_min_cost += lpSum([f1_mc[(i, j, k)] for k in K]) <= route_capacities1.get((i, j), 0), f"C_A1_EC_{i}_{j}_{constraint_counter4}"
        constraint_counter4 += 1

# Также для маршрутов от ТХ
for (i, j) in A2:
    prob_min_cost += lpSum([f2_mc[(i, j, k)] for k in K]) <= route_capacities2.get((i, j), 0), f"C_A2_{i}_{j}_{constraint_counter4}"
    constraint_counter4 += 1

# Ограничение 2: Поток, входящий в точку хранения (не в sources или sinks) больше или равен исходящему из этой точки потоку

constraint_counter4 = 0 #счетчик ограничений
nodes = df_nodes['node_id'].tolist()  # Все узлы
# Теперь источники и стоки должны быть определены на основе df_routes
sources = {}
sinks = {}
for k in K:  # Для каждого типа груза
        sources[k] = [row['start_entry_id'] for index, row in df_routes.iterrows() if row['start_entry_id'] is not None]
        sinks[k] = [row['end_consumption_id'] for index, row in df_routes.iterrows() if row['end_consumption_id'] is not None]

for node in nodes:
    if node not in [item for sublist in sources.values() for item in sublist] and node not in [item for sublist in sinks.values() for item in sublist]:
        for k in K:
            # Ограничение баланса потока
            prob_min_cost += lpSum([f1_mc[(i, node, k)] for (i, node) in A1 if (i, node, k) in f1_mc]) + lpSum([f2_mc[(i, node, k)] for (i, node) in A2 if (i, node, k) in f2_mc]) >= \
                            lpSum([f1_mc[(node, j, k)] for (node, j) in A1 if (node, j, k) in f1_mc]) + lpSum([f2_mc[(node, j, k)] for (node, j) in A2 if (node, j, k) in f2_mc]), f"Flow_Balance_{node}_{k}_{constraint_counter4}"
            constraint_counter4 += 1

# Ограничение 3: Суммарный поток, проходящий через узел, не превосходит пропускную способность этого узла
constraint_counter5 = 0 # счетчик ограничения
for node in df_nodes['node_id']:
    for k in K:
        if node in df_nodes['capacity'].values:
            #суммарный вход поток + суммарный исход поток <= пропускная способность узла
            prob_min_cost += lpSum([f1_mc[(i, node, k)] for (i, node) in A1 if (i, node, k) in f1_mc]) + lpSum([f2_mc[(i, node, k)] for (i, node) in A2 if (i, node, k) in f2_mc]) + \
                            lpSum([f1_mc[(node, j, k)] for (node, j) in A1 if (node, j, k) in f1_mc]) + lpSum([f2_mc[(node, j, k)] for (node, j) in A2 if (node, j, k) in f2_mc]) <= df_nodes[df_nodes['node_id']==node]['capacity'].values[0], f"Node_Capacity_{node}_{k}_{constraint_counter5}"
            constraint_counter5 += 1

# Ограничение 4: Сумма всего потока, исходящего из источника равно пропускной способности этого источника

constraint_counter6 = 0
node_capacities = df_nodes.set_index('node_id')['capacity'].to_dict()
for k in K:
    for s in sources[k]:
        total_outgoing_flow = lpSum([f1_mc[(s, j, k)] for (s, j) in A1 if (s, j, k) in f1_mc]) + \
                              lpSum([f2_mc[(s, j, k)] for (s, j) in A2 if (s, j, k) in f2_mc])

        # суммарный поток, выходящий из источника s, не превышал его пропускную способность
        if s in node_capacities:
            prob_min_cost += total_outgoing_flow <= node_capacities[s], f"Source_Capacity_{s}_{k}_{constraint_counter6}"
            constraint_counter6 += 1

        # Добавляем ограничение на минимальный поток из каждой точки входа
        prob_min_cost += total_outgoing_flow >= 1, f"Min_Source_Flow_{s}_{k}_{constraint_counter6}"
        constraint_counter6 += 1

# ------------------------------------------------------------------------------
# Решение задачи максимизации потока
# ------------------------------------------------------------------------------

# Здесь optimal_flow, A1 и A2 должны быть определены на основе текущей даты.

# Оптимальный поток задачи минимизации затрат не превосходит вычисленный оптимальный поток
prob_min_cost += lpSum([f1_mc[(i, j, k)] for (i, j) in A1 for k in K]) + lpSum([f2_mc[(i, j, k)] for (i, j) in A2 for k in K]) >= optimal_flow, "Maintain_Max_Flow"
prob_min_cost.solve()


#ГРАФ 
#использование маршрутов 
def calculate_route_usage(A1, A2, K, f1_mc, f2_mc):
    route_usage = {}
    for (i, j) in A1:
        route_usage[(i, j)] = 0
        for k in K:
            # добавляем переменную затрат
            if (i, j, k) in f1_mc and f1_mc[(i, j, k)].varValue is not None:
                route_usage[(i, j)] += f1_mc[(i, j, k)].varValue

    for (i, j) in A2:
        route_usage[(i, j)] = 0
        for k in K:
            # добавляем переменную затрат
            if (i, j, k) in f2_mc and f2_mc[(i, j, k)].varValue is not None:
                route_usage[(i, j)] += f2_mc[(i, j, k)].varValue
    return route_usage

# ПС узла
def calculate_node_capacities(df_nodes):
    node_capacities = {}
    for index, row in df_nodes.iterrows():
        node_id = row['node_id']
        capacity = row['capacity']
        node_capacities[node_id] = capacity
    return node_capacities

#использование злов
def calculate_node_usage(df_nodes, A1, A2, K, f1_mc, f2_mc):
  node_usage = {node: 0 for node in df_nodes['node_id']} 
  for node in df_nodes['node_id']:
      for k in K:
          # входящий поток в узел
          inflow = sum(
              [f1_mc[(i, node, k)].varValue for (i, node) in A1 if (i, node, k) in f1_mc and f1_mc[(i, node, k)].varValue is not None] +
              [f2_mc[(i, node, k)].varValue for (i, node) in A2 if (i, node, k) in f2_mc and f2_mc[(i, node, k)].varValue is not None]
          )

          # исходящий поток из узла
          outflow = sum(
              [f1_mc[(node, j, k)].varValue for (node, j) in A1 if (node, j, k) in f1_mc and f1_mc[(node, j, k)].varValue is not None] +
              [f2_mc[(node, j, k)].varValue for (node, j) in A2 if (node, j, k) in f2_mc and f2_mc[(node, j, k)].varValue is not None]
          )

          node_usage[node] += inflow + outflow  # суммируем входящий и исходящий поток
  return node_usage

route_usage = calculate_route_usage(A1, A2, K, f1_mc, f2_mc)
node_capacities = calculate_node_capacities(df_nodes)
node_usage = calculate_node_usage(df_nodes, A1, A2, K, f1_mc, f2_mc)

#значения потока
def extract_graph_data_max_flow(f1_mf, f2_mf, A1, A2, K):
    flow_values = {}
    for (i, j) in A1:
        for k in K:
            if (i, j, k) in f1_mf:
                flow_values[(i, j, k)] = f1_mf[(i, j, k)].varValue
    for (i, j) in A2:
        for k in K:
            if (i, j, k) in f2_mf:
                flow_values[(i, j, k)] = f2_mf[(i, j, k)].varValue
    return flow_values

def build_flow_network(nodes, edges, route_capacities1, route_capacities2, df_nodes, flow_values):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for u, v in edges:
        capacity = route_capacities1.get((u, v), 0) or route_capacities2.get((u, v), 0)
        total_flow = 0
        for k in K:
             if (u, v, k) in flow_values:
                 total_flow += flow_values[(u, v, k)]
        graph.add_edge(u, v, capacity=capacity, flow=total_flow)

    for node in graph.nodes():
        node_type = df_nodes.loc[df_nodes['node_id'] == node, 'node_type'].iloc[0]
        if node_type == 'Точка потребления':
            node_color = 'red'
        elif node_type == 'Точка хранения':
            node_color = 'orange'
        else:
            node_color = 'green'
        graph.nodes[node]['node_color'] = node_color 
    return graph


def draw_flow_network(graph, df_nodes, route_usage, node_capacities, node_usage):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph)

    # Определяем расположение узлов
    pos = {}
    x_entry = 0.1
    x_storage = 0.5  
    x_consumption = 0.9 
    y_spacing = 0.1 
    y_entry = 0
    y_storage = 0
    y_consumption = 0.1

    for node in graph.nodes():
        node_info = df_nodes[df_nodes['node_id'] == node].iloc[0]
        node_type = node_info['node_type']

        if node_type == 'Точка входа':
            pos[node] = (x_entry, y_entry)
            y_entry += y_spacing
        elif node_type == 'Точка хранения':
            pos[node] = (x_storage, y_storage)
            y_storage += y_spacing
        elif node_type == 'Точка потребления':
            pos[node] = (x_consumption, y_consumption)
            y_consumption += y_spacing

    # цвета узлов
    node_colors = [graph.nodes[node]['node_color'] for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=500)

    # смещаем подписи узлов
    pos_labels = {node: (x + 0.01, y + 0.01) for node, (x, y) in pos.items()}

    # рисуем номера узлов внутри кружочков
    nx.draw_networkx_labels(graph, pos,
                           labels={node: node for node in graph.nodes()},  # Отображаем только номер узла
                           font_size=12, font_family='sans-serif', font_color='black')  # Черный цвет

    # подписи узлов (с неиспользованной пропускной способностью)
    node_labels = {}
    for node in graph.nodes():
        node_info = df_nodes[df_nodes['node_id'] == node].iloc[0]
        capacity = node_capacities.get(node, 0)
        usage = node_usage.get(node, 0)
        unused = capacity - usage
        node_labels[node] = f"{node_info['node_name']}"  # Имя 

    nx.draw_networkx_labels(graph, pos=pos_labels, labels=node_labels, font_size=8, font_family='sans-serif')

    # определение минимальной и максимальной оставшейся пропускной способности
    min_unused_capacity = float('inf')
    max_unused_capacity = 0
    for u, v, data in graph.edges(data=True):
        capacity = data.get('capacity', 0)
        flow = data.get('flow', 0)
        unused_capacity = capacity - flow
        min_unused_capacity = min(min_unused_capacity, unused_capacity)
        max_unused_capacity = max(max_unused_capacity, unused_capacity)

    print(f"min_unused_capacity: {min_unused_capacity}, max_unused_capacity: {max_unused_capacity}")

    # Отрисовка ребер
    edge_colors = []
    edge_widths = []
    for u, v, data in graph.edges(data=True):
        capacity = data.get('capacity', 0)
        flow = data.get('flow', 0)
        unused_capacity = capacity - flow

        if unused_capacity > 0:
            # нормализация оставшейся пропускной способности
            normalized_unused_capacity = (unused_capacity - min_unused_capacity) / (max_unused_capacity - min_unused_capacity) \
                if max_unused_capacity > min_unused_capacity else 0
            normalized_unused_capacity = max(0, min(1, normalized_unused_capacity))

            color = plt.cm.BuGn(normalized_unused_capacity)
            edge_colors.append(color)
            edge_widths.append(unused_capacity / 100000)
        else:
            # серый цвет, если пропускная способность равна 0
            edge_colors.append('gray')
            edge_widths.append(0.5)

    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7)

    # подписи ребер
    edge_labels = {}
    for u, v, data in graph.edges(data=True):
        capacity = data.get('capacity', 0)
        flow = data.get('flow', 0)
        unused_capacity = capacity - flow
        edge_labels[(u, v)] = (
            #f"Пропускная способность: {capacity}\n"
            #f"Использовано: {flow:.2f}\n"
            f"Осталось: {unused_capacity:.2f}"
        )

    nx.draw_networkx_edge_labels(
        graph,
        pos=pos,
        edge_labels=edge_labels,
        font_size=7,
        font_family='sans-serif'
    )

    plt.axis('off')
    plt.tight_layout()
    plt.savefig("transport_network.png")
    plt.show()

# извлекаем объемы потоков из решения max flow
flow_values = extract_graph_data_max_flow(f1_mf, f2_mf, A1, A2, K)

# список всех узлов
nodes = set()
for (i, j) in A1 + A2:
    nodes.add(i)
    nodes.add(j)

# создаем граф
flow_network = build_flow_network(nodes, A1 + A2, route_capacities1, route_capacities2, df_nodes, flow_values)

# рисуем граф
draw_flow_network(flow_network, df_nodes, route_usage, node_capacities, node_usage)  # передача df_nodes


#ИНТЕРАКТИВНАЯ КАРТА

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

print('df_entry:', df_entry)

#Streamlit
st.title("Интерактивная карта ХКГМ")

#центр карты
center_latitude = 70.028470
center_longitude = 69.524715

# чекбоксы
show_entry_points = st.checkbox("Показать точки входа", value=True)
show_storage_points = st.checkbox("Показать точки хранения", value=True)
show_consumption_points = st.checkbox("Показать точки потребления", value=True)
show_routes = st.checkbox("Показать маршруты", value=True)

#карта
m = folium.Map(location=[center_latitude, center_longitude], zoom_start=7)

#маркеры ТВ (df_nodes)
for index, row in df_nodes.iterrows():
    latitude = row['latitude']
    longitude = row['longitude']
    node_type = row['node_type']
    node_name = row['node_name']

    if pd.notna(latitude) and pd.notna(longitude):
        if node_type == 'Точка входа' and show_entry_points:
            icon_color = "green"
        elif node_type == 'Точка хранения' and show_storage_points:
            icon_color = "orange"
        elif node_type == 'Точка потребления' and show_consumption_points:
            icon_color = "red"
        else:
            continue # Пропускаем маркер, если тип точки не соответствует условию

        folium.Marker(
            location=[latitude, longitude],
            popup=f"<b>{node_name} ({node_type})</b><br>Долгота:{latitude}<br>Широта:{longitude}",
            icon=folium.Icon(color=icon_color, icon="home"),
        ).add_to(m)

colors = {"Точка входа": "rgb(50,205,50)", "Точка хранения": "orange", "Точка потребления": "	rgb(220,20,60)"}

# HTML для легенды с иконками домиков
legend_html = """
     <div style="
         position: fixed;
         bottom: 50px;
         left: 50px;
         width: 180px;
         height: auto;
         z-index: 9999;
         background-color: white;
         border: 2px solid rgba(0,0,0,0.2);
         border-radius: 5px;
         padding: 10px;
         font-size: 14px;
         opacity: 0.8;
     ">
     <p><b>Легенда</b></p>
"""

for label, color in colors.items():
    legend_html += f"""
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <i class="fa fa-home fa-lg" style="color:{color}; margin-right: 5px;"></i>
            <div>{label}</div>
        </div>
    """

legend_html += """
     </div>
"""

# Добавляем CustomPane для легенды
m.add_child(folium.map.CustomPane("legend", "bottom left"))

# Привязываем HTML к CustomPane
m.get_root().html.add_child(
    folium.Element(legend_html)
)

# Предварительная загрузка Font Awesome
m.get_root().header.add_child(folium.Element('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" integrity="sha512-9usAa10IRO0HhonpyAIVpjrylPvoDwiPUiKdWk5t3PyolY1cOd4DSE0Ga+ri4AuTroPR5aQvXU9xC6qOPnzFeg==" crossorigin="anonymous" referrerpolicy="no-referrer" />'))

# дуги

def create_arc(start_lat, start_lon, end_lat, end_lon, height=0.2):

    #линия между точками
    line = LineString([(start_lon, start_lat), (end_lon, end_lat)])

    # середина между точками
    midpoint = line.interpolate(0.5, normalized=True)
    mid_lon, mid_lat = midpoint.x, midpoint.y

    #расстояние между точками
    distance = line.length

    #угол поворота дуги
    angle = np.arctan2(end_lat - start_lat, end_lon - start_lon)

    # создание Point в средней точке и смещение его по нормали к линии
    arc_point = Point(mid_lon, mid_lat)
    arc_point = Point(arc_point.x - np.sin(angle) * height * distance, arc_point.y + np.cos(angle) * height * distance)

    # создание дуги с использованием интерполяции
    num_points = 50  # Количество точек для отрисовки дуги
    arc = []
    for i in range(num_points + 1):
        alpha = i / num_points
        point = line.interpolate(alpha, normalized=True)
        x = point.x
        y = point.y
        # смещение точки по параболе
        z = 4 * height * distance * alpha * (1 - alpha)
        x = x + (arc_point.x - mid_lon) * z
        y = y + (arc_point.y - mid_lat) * z
        arc.append((y, x))  # Широта, долгота

    return arc

if show_routes:
    routes = []
    for index, row in df_routes.iterrows():
        #координаты и наименование начточки
        start_lat = None
        start_lon = None
        start_name = None
        if pd.notna(row['start_entry_id']):
            start_point = df_entry[df_entry['node_id'] == row['start_entry_id']]
            if not start_point.empty:
                start_lat = start_point['latitude'].iloc[0]
                start_lon = start_point['longitude'].iloc[0]
                start_name = start_point['node_name'].iloc[0]
        elif pd.notna(row['start_storage_id']):
            start_point = df_storage[df_storage['node_id'] == row['start_storage_id']]
            if not start_point.empty:
                start_lat = start_point['latitude'].iloc[0]
                start_lon = start_point['longitude'].iloc[0]
                start_name = start_point['node_name'].iloc[0]

        #координаты и наименование кон точки
        end_lat = None
        end_lon = None
        end_name = None

        if pd.notna(row['end_storage_id']):
            end_point = df_storage[df_storage['node_id'] == row['end_storage_id']]
            if not end_point.empty:
                end_lat = end_point['latitude'].iloc[0]
                end_lon = end_point['longitude'].iloc[0]
                end_name = end_point['node_name'].iloc[0]
        elif pd.notna(row['end_consumption_id']):
            end_point = df_consumption[df_consumption['node_id'] == row['end_consumption_id']]
            if not end_point.empty:
                end_lat = end_point['latitude'].iloc[0]
                end_lon = end_point['longitude'].iloc[0]
                end_name = end_point['node_name'].iloc[0]

        # рисуем пути (дуги)
        if start_lat is not None and start_lon is not None and end_lat is not None and end_lon is not None:
            # Определяем start_node и end_node здесь, внутри цикла
            start_node = int(row['start_node'])
            end_node = int(row['end_node'])

            # вспомогательный словарь для комментариев в тултипах
            route = {
                'route_id': row['route_id'],
                'start_lat': start_lat,
                'start_lon': start_lon,
                'start_name': start_name,
                'end_lat': end_lat,
                'end_lon': end_lon,
                'end_name': end_name,
                'transportation_cost': row['transportation_cost']  # Добавляем стоимость транспортировки
            }
            routes.append(route)

            # Определяем цвет и толщину линии на основе потока
            if flow_network.has_edge(start_node, end_node):
                total_flow = flow_network[start_node][end_node]['flow']
                capacity = flow_network[start_node][end_node]['capacity']
                color = 'gray' if total_flow > 0 else 'black'
                weight = max(1, total_flow / 20000)  # толщина линии пропорциональна потоку
            else:
                total_flow = 0
                capacity = 0
                color = 'gray'
                weight = 1

            # Создание дуги
            arc = create_arc(start_lat, start_lon, end_lat, end_lon, height=0.1)

            tooltip_text = f"<b>Номер маршрута:</b> {row['route_id']}<br>" \
                           f"<b>Начальная точка:</b> {start_name}<br>" \
                           f"<b>Конечная точка:</b> {end_name}<br>" \
                           f"<b>Стоимость транспортировки:</b> {row['transportation_cost']}<br>" \
                           f"<b>Поток:</b> {total_flow:.2f}<br>" \
                           f"<b>Пропускная способность:</b> {capacity}"

            folium.PolyLine(
                locations=arc,
                color=color,
                weight=weight,  # Устанавливаем толщину линии
                opacity=0.7,
                smooth_factor=0,
                tooltip=tooltip_text).add_to(m)

df_routes = df_routes.rename(columns={'route_type_x': 'route_type'})
st.write("df_routes", df_routes)
st.write("df_entry", df_entry)
st.write("df_storage", df_storage)
st.write("df_consumption", df_consumption)
st.write("df_nodes", df_nodes)

# Отображение карты в Streamlit
#folium_static(m, width=1500, height=800)
st.components.v1.html(m._repr_html_(), height=1000, width=1500)