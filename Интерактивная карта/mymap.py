import pandas as pd
import numpy as np
import psycopg2
import pulp
from pulp import *
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import folium
from streamlit_folium import folium_static
from shapely.geometry import Point, LineString

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
df_entry['capacity'] = [500000, 2050000, 200000, 30000, 210000, 1500000, 185000, 300000]
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
routes_query = "select route_id, start_entry_id, start_storage_id, end_storage_id, end_consumption_id, route_type, cargo_volume, capacity, transportation_cost, allowed_transport_vehicles from transport_routes"
routes_data = get_data(routes_query)
df_routes = pd.DataFrame(routes_data, columns=['route_id', 'start_entry_id', 'start_storage_id', 'end_storage_id', 'end_consumption_id', 'route_type', 'cargo_volume', 'capacity', 'transportation_cost', 'allowed_transport_vehicles'])

#  Переименуем столбцы, чтобы код соответствовал
df_entry = df_entry.rename(columns={'entry_id': 'node_id', 'entry_name': 'node_name'})
df_storage = df_storage.rename(columns={'storage_id': 'node_id',  'storage_name': 'node_name'})
df_consumption = df_consumption.rename(columns={'consumption_id': 'node_id', 'consumption_name': 'node_name'})

#словари для разных типов маршрутов
#прямые маршруты от ТВ до ТХ
A1 = []
#маршруты от ТХ до ТП
A2 = []

# Переименуем столбцы, чтобы код соответствовал
df_entry = df_entry.rename(columns={'entry_id': 'node_id', 'entry_name': 'node_name'})
df_storage = df_storage.rename(columns={'storage_id': 'node_id',  'storage_name': 'node_name'})
df_consumption = df_consumption.rename(columns={'consumption_id': 'node_id', 'consumption_name': 'node_name'})

# Добавляем столбцы start_node и end_node
df_routes['start_node'] = df_routes['start_entry_id'].fillna(df_routes['start_storage_id'])
df_routes['end_node'] = df_routes['end_storage_id'].fillna(df_routes['end_consumption_id'])

# Создание дуг
for index, row in df_routes.iterrows():
    start_node = row['start_node']
    end_node = row['end_node']

    if not pd.isna(start_node) and not pd.isna(end_node):
        start_node = int(start_node)
        end_node = int(end_node)

        # Определяем, к какому списку добавить дугу
        if not pd.isna(row['start_entry_id']): #Если start_entry_id определен, значит, это A1
          A1.append((start_node, end_node))
        elif not pd.isna(row['start_storage_id']): #Иначе если start_storage_id определен, значит, это A2
          A2.append((start_node, end_node))

# типы грузов
K = df_cargo['cargo_type'].tolist()

# общий датафрейм
#df_nodes = pd.concat([df_entry, df_storage, df_consumption], ignore_index=True)

# общий датафрейм
frames = []
if df_entry is not None:
    frames.append(df_entry)
if df_storage is not None:
    frames.append(df_storage)
if df_consumption is not None:
    frames.append(df_consumption)

if frames:
    df_nodes = pd.concat(frames, ignore_index=True)
else:
    df_nodes = None

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

#словарь со стоимостями доставки
costs = {}

for index, row in df_routes.iterrows():
    start_node = row['start_node']
    end_node = row['end_node']
    transportation_cost = row['transportation_cost']
    cargo_volume = row['cargo_volume']

    if not pd.isna(start_node) and not pd.isna(end_node):
        start_node = int(start_node)
        end_node = int(end_node)

        for k in K:
            costs[(start_node, end_node, k)] = transportation_cost / cargo_volume if cargo_volume > 0 else 0

print("K:", K)

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

print(f"Пропускная способность вершин:{node_capacities}")

#пропусные способности маршрутов от точки входа
route_capacities1 = {}
for index, row in df_routes.iterrows():
    start_entry_id = row['start_entry_id']
    end_storage_id = row['end_storage_id']
    end_consumption_id = row['end_consumption_id']
    capacity = row['capacity']

    #добавляем маршрут, если он существует (от точки входа до точки хранения)
    if not pd.isna(start_entry_id) and not pd.isna(end_storage_id):
        route_capacities1[(start_entry_id, end_storage_id)] = capacity

    #добавляем маршрут, если он существует (от точки входа до точки потребления)  
    elif not pd.isna(start_entry_id) and not pd.isna(end_consumption_id): 
        route_capacities1[(start_entry_id, end_consumption_id)] = capacity

#маршруты от точки хранения
route_capacities2 = {}
for index, row in df_routes.iterrows():
    start_storage_id = row['start_storage_id']
    end_consumption_id = row['end_consumption_id']
    capacity = row['capacity']

    if not pd.isna(start_storage_id) and not pd.isna(end_consumption_id):
        route_capacities2[(start_storage_id, end_consumption_id)] = capacity
        print(f"Добавлена дуга: {(start_storage_id, end_consumption_id)} с capacity: {capacity}")

print(f"Пропускная способность дуг A1:{route_capacities1}")
print(f"Пропускная способность дуг A2:{route_capacities2}")


#модель PuLP - максимизация общего потока
# максимизация общего потока
prob_max_flow = LpProblem("MaxFlow", LpMaximize)

#переменные объема потока

f1_mf = LpVariable.dicts("f1_mf", [(i, j, k) for (i, j) in A1 for k in K], lowBound=0, cat='Continuous')
f2_mf = LpVariable.dicts("f2_mf", [(i, j, k) for (i, j) in A2 for k in K], lowBound=0, cat='Continuous')

# ЦФ (максимизация общего обхема потока)
prob_max_flow += lpSum([f1_mf[(i, j, k)] for (i, j) in A1 for k in K]) + lpSum([f2_mf[(i, j, k)] for (i, j) in A2 for k in K]), "Total Flow"

#ограничения
constraint_counter = 0
#проверка, что в маршрутах от точких хранения конечная точка является ТХ или ТП
for (i, j) in A1:
    is_storage = j in df_storage['node_id'].values
    is_consumption = j in df_consumption['node_id'].values

    #если конечная точка - ТХ, то пишем ограничение о том, что суммарный поток по дуги не выше пропускной сопособности жтой дуги
    if is_storage:
        #<= route_capacities1.get((i, j) - возвращаем ПС дуги, если она есть, иначе - 0
        prob_max_flow += lpSum([f1_mf[(i, j, k)] for k in K]) <= route_capacities1.get((i, j), 0), f"C_A1_ES_{i}_{j}_{constraint_counter}"
        constraint_counter += 1 #увеличиваем счетчик
    #если конечная точка - ТП, то пишем ограничение о том, что суммарный поток по дуги не выше пропускной сопособности жтой дуги
    elif is_consumption:
        prob_max_flow += lpSum([f1_mf[(i, j, k)] for k in K]) <= route_capacities1.get((i, j), 0), f"C_A1_EC_{i}_{j}_{constraint_counter}"
        constraint_counter += 1 #увеличиваем счетчик

#также для маршрутов от тх
for (i, j) in A2:
    prob_max_flow += lpSum([f2_mf[(i, j, k)] for k in K]) <= route_capacities2.get((i, j), 0), f"C_A2_{i}_{j}_{constraint_counter}"
    constraint_counter += 1 #увеличиваем счетчик

#все уникальный узлы в сети из А1 и А2
nodes = set()
for (i, j) in A1 + A2:
    nodes.add(i)
    nodes.add(j)

#если узел не находится в  sources или sinks, то каждый входящий в него потока равен исходящему
for node in nodes:
    if node not in [item for sublist in sources.values() for item in sublist] and node not in [item for sublist in sinks.values() for item in sublist]:
        for k in K:
            prob_max_flow += lpSum([f1_mf[(i, node, k)] for (i, node) in A1 if (i, node, k) in f1_mf]) + lpSum([f2_mf[(i, node, k)] for (i, node) in A2 if (i, node, k) in f2_mf]) == \
                            lpSum([f1_mf[(node, j, k)] for (node, j) in A1 if (node, j, k) in f1_mf]) + lpSum([f2_mf[(node, j, k)] for (node, j) in A2 if (node, j, k) in f2_mf]), f"Flow_Balance_{node}_{k}"

# ограничения пропускной способности узлов
constraint_counter = 0 #счетчик ограничений для уникальных имен
#суммарный потока каждого типа груза, проходящий через узел равно выходящему из узла
for node in df_nodes['node_id']:
    for k in K:
        if node in node_capacities:
            prob_max_flow += lpSum([f1_mf[(i, node, k)] for (i, node) in A1 if (i, node, k) in f1_mf]) + lpSum([f2_mf[(i, node, k)] for (i, node) in A2 if (i, node, k) in f2_mf]) + \
                            lpSum([f1_mf[(node, j, k)] for (node, j) in A1 if (node, j, k) in f1_mf]) + lpSum([f2_mf[(node, j, k)] for (node, j) in A2 if (node, j, k) in f2_mf]) <= node_capacities[node], f"Node_Capacity_{node}_{k}_{constraint_counter}"
            constraint_counter += 1

# ограничения на количество груза, отправляемого из источников
for k in K:
    for s in sources[k]:
        # сумма всего поток, исходящий из источника s для груза k равно пропускной способности этого истокчника
        total_outgoing_flow = lpSum([f1_mf[(s, j, k)] for (s, j) in A1 if s == s and (s, j, k) in f1_mf]) + \
                              lpSum([f2_mf[(s, j, k)] for (s, j) in A2 if s == s and (s, j, k) in f2_mf])

        # Проверяем, есть ли у источника s ограничение пропускной способности
        if s in node_capacities:
            prob_max_flow += total_outgoing_flow <= node_capacities[s], f"Source_Capacity_{s}_{k}"

#максимизация потока
prob_max_flow.solve()
optimal_flow = value(prob_max_flow.objective)
print("Оптимальный поток:", optimal_flow)

#модель PuLP - мпинимизация затрат
prob_min_cost = LpProblem("MinCostGivenMaxFlow", LpMinimize)

f1_mc = LpVariable.dicts("f1_mc", [(i, j, k) for (i, j) in A1 for k in K], lowBound=0, cat='Continuous')
f2_mc = LpVariable.dicts("f2_mc", [(i, j, k) for (i, j) in A2 for k in K], lowBound=0, cat='Continuous')

prob_min_cost += lpSum([costs[(i, j, k)] * f1_mc[(i, j, k)] for (i, j) in A1 for k in K]) + lpSum([costs[(i, j, k)] * f2_mc[(i, j, k)] for (i, j) in A2 for k in K]), "Total Cost"

constraint_counter_min_cost = 0  # Initialize a separate counter for min cost
#ограничения
#проверка, что в маршрутах от точких хранения конечная точка является ТХ или ТП
for (i, j) in A1:
    is_storage = j in df_storage['node_id'].values
    is_consumption = j in df_consumption['node_id'].values

    #если конечная точка - ТХ, то пишем ограничение о том, что суммарный поток по дуги не выше пропускной сопособности жтой дуги
    if is_storage:
        #<= route_capacities1.get((i, j) - возвращаем ПС дуги, если она есть, иначе - 0
        prob_min_cost += lpSum([f1_mc[(i, j, k)] for k in K]) <= route_capacities1.get((i, j), 0), f"C_A1_ES_{i}_{j}_{constraint_counter_min_cost}"
        constraint_counter_min_cost += 1 #увеличиваем счетчик
    #если конечная точка - ТП, то пишем ограничение о том, что суммарный поток по дуги не выше пропускной сопособности жтой дуги
    elif is_consumption:
        prob_min_cost += lpSum([f1_mc[(i, j, k)] for k in K]) <= route_capacities1.get((i, j), 0), f"C_A1_EC_{i}_{j}_{constraint_counter_min_cost}"
        constraint_counter_min_cost += 1 #увеличиваем счетчик

#также для маршрутов от тх
for (i, j) in A2:
    prob_min_cost += lpSum([f2_mc[(i, j, k)] for k in K]) <= route_capacities2.get((i, j), 0), f"C_A2_{i}_{j}_{constraint_counter_min_cost}"
    constraint_counter_min_cost += 1 #увеличиваем счетчик

#все уникальные узлы
nodes = set()
for (i, j) in A1 + A2:
    nodes.add(i)
    nodes.add(j)

#ограничение баланса потока (входящий в узел поток равен потоку исходящему из него)
for node in nodes:
    if node not in [item for sublist in sources.values() for item in sublist] and node not in [item for sublist in sinks.values() for item in sublist]:
        for k in K:
            #вхрдящий поток == исходящему потоку
            prob_min_cost += lpSum([f1_mc[(i, node, k)] for (i, node) in A1 if (i, node, k) in f1_mc]) + lpSum([f2_mc[(i, node, k)] for (i, node) in A2 if (i, node, k) in f2_mc]) == \
                            lpSum([f1_mc[(node, j, k)] for (node, j) in A1 if (node, j, k) in f1_mc]) + lpSum([f2_mc[(node, j, k)] for (node, j) in A2 if (node, j, k) in f2_mc]), f"Flow_Balance_{node}_{k}_{constraint_counter_min_cost}"
            constraint_counter_min_cost += 1

#ограничение пропускной способности узла (суммарный поток , проходящий через узел равен пропускной способности ухла)
for node in df_nodes['node_id']:
    for k in K:
        if node in node_capacities:
            #суммарный вход поток+ суммарный исход поток <= пропускная способность узла
            prob_min_cost += lpSum([f1_mc[(i, node, k)] for (i, node) in A1 if (i, node, k) in f1_mc]) + lpSum([f2_mc[(i, node, k)] for (i, node) in A2 if (i, node, k) in f2_mc]) + \
                            lpSum([f1_mc[(node, j, k)] for (node, j) in A1 if (node, j, k) in f1_mc]) + lpSum([f2_mc[(node, j, k)] for (node, j) in A2 if (node, j, k) in f2_mc]) <= node_capacities[node], f"Node_Capacity_{node}_{k}_{constraint_counter_min_cost}"
            #счетчик ограничения
            constraint_counter_min_cost += 1

#пропускная способность источника (поток выходящий из источника не превышает пропускную способность этого источника)
source_constraint_counter_min_cost = 0
for k in K:
    for s in sources[k]:
        #исходящий поток только
        total_outgoing_flow = lpSum([f1_mc[(s, j, k)] for (s, j) in A1 if (s, j, k) in f1_mc]) + \
                              lpSum([f2_mc[(s, j, k)] for (s, j) in A2 if (s, j, k) in f2_mc])

        # суммарный поток, выходящий из источника s, не превышал его пропускную способность
        if s in node_capacities:
            prob_min_cost += total_outgoing_flow <= node_capacities[s], f"Source_Capacity_{s}_{k}_{source_constraint_counter_min_cost}"
            source_constraint_counter_min_cost += 1
        # Добавляем ограничение на минимальный поток из каждой точки входа
        prob_min_cost += total_outgoing_flow >= 1, f"Min_Source_Flow_{s}_{k}_{source_constraint_counter_min_cost}"
        source_constraint_counter_min_cost += 1

#оптимальный поток будет не меньше вычисленного оптимального потока
prob_min_cost += lpSum([f1_mc[(i, j, k)] for (i, j) in A1 for k in K]) + lpSum([f2_mc[(i, j, k)] for (i, j) in A2 for k in K]) >= optimal_flow, "Maintain_Max_Flow"

prob_min_cost.solve()
print("Общие затраты:", value(prob_min_cost.objective))


#ГРАФ 
def extract_graph_data_max_flow(f1_mf, f2_mf, A1, A2, K):
    """Извлекает значения потока и формирует словарь для визуализации."""
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
    """Строит граф потока с учетом пропускных способностей и потоков."""
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)

    # Добавляем все возможные ребра из df_routes
    for u, v in edges:
        # Определяем пропускную способность для ребра
        capacity = 0
        if (u, v) in route_capacities1:
            capacity = route_capacities1[(u, v)]
        elif (u, v) in route_capacities2:
            capacity = route_capacities2[(u, v)]
        
        # Вычисляем общий поток для ребра
        total_flow = 0
        for k in K:
            if (u, v, k) in flow_values:
                total_flow += flow_values[(u, v, k)]

        # Добавляем ребро и сохраняем информацию о потоке и пропускной способности
        graph.add_edge(u, v, capacity=capacity, flow=total_flow)

    # Определяем цвет узлов на основе df_nodes['node_type']
    if df_nodes is not None:
        node_type_dict = pd.Series(df_nodes['node_type'].values, index=df_nodes['node_id']).to_dict()
    else:
        node_type_dict = {}

    for node in nodes:
        if node in node_type_dict:
            node_type = node_type_dict[node]
            if node_type == 'Точка входа':
                graph.nodes[node]['node_color'] = 'green'
            elif node_type == 'Точка хранения':
                graph.nodes[node]['node_color'] = 'orange'  # Можете выбрать другой цвет
            elif node_type == 'Точка потребления':
                graph.nodes[node]['node_color'] = 'red'
            else:
                graph.nodes[node]['node_color'] = 'blue'  # Цвет по умолчанию
        else:
            graph.nodes[node]['node_color'] = 'blue'  # Цвет по умолчанию, если нет в df_nodes

    return graph

def draw_flow_network(graph):
    """Отрисовывает граф с пропускными способностями и потоками."""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42)  # для воспроизводимости

    # Цвета узлов
    node_colors = [graph.nodes[node]['node_color'] for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=500)

    # Разделяем ребра на две группы: с потоком и без потока
    edges_with_flow = [(u, v) for u, v in graph.edges() if graph[u][v]['flow'] > 0]
    edges_without_flow = [(u, v) for u, v in graph.edges() if graph[u][v]['flow'] == 0]

    # Рисуем ребра без потока серым цветом
    nx.draw_networkx_edges(graph, pos, edgelist=edges_without_flow, width=1, edge_color='gray', alpha=0.5)

    # Рисуем ребра с потоком синим цветом
    edges_width = [graph[u][v]['flow'] / 5000 for u, v in edges_with_flow]  # для удобочитаемости
    nx.draw_networkx_edges(graph, pos, edgelist=edges_with_flow, width=edges_width, edge_color='blue', alpha=0.7)

    # Подписи узлов
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family='sans-serif')

    # Подписи ребер (пропускная способность и поток)
    edge_labels = {(u, v): f"Cap:{graph[u][v]['capacity']}\nFlow:{graph[u][v]['flow']:.2f}"
                   for u, v in graph.edges()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, font_family='sans-serif')

    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Решаем задачу max flow
prob_max_flow.solve()
optimal_flow = value(prob_max_flow.objective) # Получаем оптимальный поток
print("Оптимальный поток:", optimal_flow)
print(df_nodes)
print('f1_mc:', f1_mc)
print('f2_mc:', f2_mc)
# Извлекаем объемы потоков из решения max flow
flow_values = extract_graph_data_max_flow(f1_mf, f2_mf, A1, A2, K)

# Список всех узлов
nodes = set()
for (i, j) in A1 + A2:
    nodes.add(i)
    nodes.add(j)

# Создаем граф
flow_network = build_flow_network(nodes, A1 + A2, route_capacities1, route_capacities2, df_nodes, flow_values)

# Рисуем граф
draw_flow_network(flow_network)


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

#маркеры ТВ
if show_entry_points:
    # Создаем булеву маску для фильтрации строк
    entry_points = df_nodes['node_type'] == 'Точка входа'
    for index, row in df_nodes[entry_points].iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"<b>{row['node_name']} (Точка входа)</b><br>Долгота:{row['latitude']}<br>Широта:{row['longitude']}",
            icon=folium.Icon(color="green", icon="home"),
        ).add_to(m)

#маркеры ТХ
if show_storage_points:
    # Создаем булеву маску для фильтрации строк
    storage_points = df_nodes['node_type'] == 'Точка хранения'
    for index, row in df_nodes[storage_points].iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"<b>{row['node_name']} (Точка хранения)</b><br>Долгота:{row['latitude']}<br>Широта:{row['longitude']}",
            icon=folium.Icon(color="orange", icon="home"),
        ).add_to(m)

#маркеры ТП
if show_consumption_points:
    # Создаем булеву маску для фильтрации строк
    consumption_points = df_nodes['node_type'] == 'Точка потребления'
    for index, row in df_nodes[consumption_points].iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"<b>{row['node_name']} (Точка потребления)</b><br>Долгота:{row['latitude']}<br>Широта:{row['longitude']}",
            icon=folium.Icon(color="red", icon="home"),
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

    # Создание Point в средней точке и смещение его по нормали к линии
    arc_point = Point(mid_lon, mid_lat)
    arc_point = Point(arc_point.x - np.sin(angle) * height * distance, arc_point.y + np.cos(angle) * height * distance)

    # Создание дуги с использованием интерполяции
    num_points = 50  # Количество точек для отрисовки дуги
    arc = []
    for i in range(num_points + 1):
        alpha = i / num_points
        point = line.interpolate(alpha, normalized=True)
        x = point.x
        y = point.y
        # Смещение точки по параболе
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
st.write(df_routes)
st.write(df_entry)
st.write(df_storage)
st.write(df_consumption)
st.write(df_nodes)

# Отображение карты в Streamlit
#folium_static(m, width=1500, height=800)
st.components.v1.html(m._repr_html_(), height=1000, width=1500)