import pandas as pd
import numpy as np
import psycopg2
import pulp
from pulp import *
import matplotlib.pyplot as plt
import networkx as nx

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
df_entry['capacity'] = [500000, 250000, 200000, 30000, 210000, 1500000, 185000, 300000]
#точки хранения
storage_query = "select storage_id, storage_name, latitude, longitude from points_of_storage"
storage_data = get_data(storage_query)
df_storage = pd.DataFrame(storage_data, columns=['storage_id', 'storage_name', 'latitude', 'longitude'])
df_storage['capacity'] = [250000, 1500000, 300000]

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

def calculate_route_usage(A1, A2, K, f1_mc, f2_mc):
    """Вычисляет использование маршрутов на основе решения модели минимизации стоимости."""
    route_usage = {}
    for (i, j) in A1:
        route_usage[(i, j)] = 0
        for k in K:
            # Если переменная существует и имеет значение, добавляем его
            if (i, j, k) in f1_mc and f1_mc[(i, j, k)].varValue is not None:
                route_usage[(i, j)] += f1_mc[(i, j, k)].varValue

    for (i, j) in A2:
        route_usage[(i, j)] = 0
        for k in K:
            # Если переменная существует и имеет значение, добавляем его
            if (i, j, k) in f2_mc and f2_mc[(i, j, k)].varValue is not None:
                route_usage[(i, j)] += f2_mc[(i, j, k)].varValue
    return route_usage

def calculate_node_capacities(df_nodes):
    """Создает словарь с пропускной способностью для каждого узла."""
    node_capacities = {}
    for index, row in df_nodes.iterrows():
        node_id = row['node_id']
        capacity = row['capacity']
        node_capacities[node_id] = capacity
    return node_capacities

def calculate_node_usage(df_nodes, A1, A2, K, f1_mc, f2_mc):
  """Вычисляет использование пропускной способности узлов."""
  node_usage = {node: 0 for node in df_nodes['node_id']}  # Инициализация для всех узлов
  for node in df_nodes['node_id']:
      for k in K:
          # Входящий поток в узел
          inflow = sum(
              [f1_mc[(i, node, k)].varValue for (i, node) in A1 if (i, node, k) in f1_mc and f1_mc[(i, node, k)].varValue is not None] +
              [f2_mc[(i, node, k)].varValue for (i, node) in A2 if (i, node, k) in f2_mc and f2_mc[(i, node, k)].varValue is not None]
          )

          # Исходящий поток из узла
          outflow = sum(
              [f1_mc[(node, j, k)].varValue for (node, j) in A1 if (node, j, k) in f1_mc and f1_mc[(node, j, k)].varValue is not None] +
              [f2_mc[(node, j, k)].varValue for (node, j) in A2 if (node, j, k) in f2_mc and f2_mc[(node, j, k)].varValue is not None]
          )

          node_usage[node] += inflow + outflow  # Суммируем входящий и исходящий поток
  return node_usage

route_usage = calculate_route_usage(A1, A2, K, f1_mc, f2_mc)

# Вычисляем node_capacities
node_capacities = calculate_node_capacities(df_nodes)

# Вычисляем node_usage
node_usage = calculate_node_usage(df_nodes, A1, A2, K, f1_mc, f2_mc)

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
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for u, v in edges:
        # Получаем capacity из route_capacities (убедитесь, что они правильно заполнены)
        capacity = route_capacities1.get((u, v), 0) or route_capacities2.get((u, v), 0)
        total_flow = 0
        for k in K:
             if (u, v, k) in flow_values:
                 total_flow += flow_values[(u, v, k)]
        graph.add_edge(u, v, capacity=capacity, flow=total_flow)
    # Добавляем атрибут node_color на основе типа узла
    for node in graph.nodes():
        node_type = df_nodes.loc[df_nodes['node_id'] == node, 'node_type'].iloc[0]
        if node_type == 'Точка потребления':
            node_color = 'red'
        elif node_type == 'Точка хранения':
            node_color = 'orange'
        else:  # 'Точка входа'
            node_color = 'green'
        graph.nodes[node]['node_color'] = node_color  # сохраняем цвет
    return graph

def exponential_normalization(flow, min_flow, max_flow, exponent=2):
    """Applies exponential normalization to flow values."""
    if max_flow == min_flow:
        return 1.0  # To avoid division by zero
    normalized_flow = (flow - min_flow) / (max_flow - min_flow)
    # Apply exponential scaling
    normalized_flow = normalized_flow ** exponent
    return max(0.0, min(normalized_flow, 1.0))  # Ensure value is within [0, 1] range

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def draw_flow_network(graph, df_nodes, route_usage, node_capacities, node_usage):
    """Отрисовывает граф с пропускными способностями и оставшейся пропускной способностью."""
    plt.figure(figsize=(12, 8))
    plt.title("Транспортная сеть с оставшейся пропускной способностью")
    pos = nx.spring_layout(graph)

    # Определяем расположение узлов
    pos = {}
    x_entry = 0.1  # Координата X для точек входа (слева)
    x_storage = 0.5  # Координата X для точек хранения (середина)
    x_consumption = 0.9  # Координата X для точек потребления (справа)
    y_spacing = 0.1  # Расстояние по Y между узлами одного типа

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

    # Цвета узлов
    node_colors = [graph.nodes[node]['node_color'] for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=500)

    # Смещаем подписи узлов
    pos_labels = {node: (x + 0.01, y + 0.01) for node, (x, y) in pos.items()}

    # Рисуем номера узлов внутри кружочков
    nx.draw_networkx_labels(graph, pos,
                           labels={node: node for node in graph.nodes()},  # Отображаем только номер узла
                           font_size=12, font_family='sans-serif', font_color='black')  # Черный цвет

    # Подписи узлов (с неиспользованной пропускной способностью)
    node_labels = {}
    for node in graph.nodes():
        node_info = df_nodes[df_nodes['node_id'] == node].iloc[0]
        capacity = node_capacities.get(node, 0)
        usage = node_usage.get(node, 0)
        unused = capacity - usage
        node_labels[node] = f"{node_info['node_name']}"  # Имя 

    nx.draw_networkx_labels(graph, pos=pos_labels, labels=node_labels, font_size=8, font_family='sans-serif')

    # Определение минимальной и максимальной оставшейся пропускной способности
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
            # Нормализация оставшейся пропускной способности
            normalized_unused_capacity = (unused_capacity - min_unused_capacity) / (max_unused_capacity - min_unused_capacity) \
                if max_unused_capacity > min_unused_capacity else 0
            normalized_unused_capacity = max(0, min(1, normalized_unused_capacity))  # Ensure value is within [0, 1]

            color = plt.cm.BuGn(normalized_unused_capacity)  # Используем Greens для оставшейся пропускной способности
            edge_colors.append(color)
            edge_widths.append(unused_capacity / 100000) #указать другой scale
        else:
            # Назначаем серый цвет, если пропускная способность равна 0
            edge_colors.append('gray') #черный
            edge_widths.append(0.5)

    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7)

    # Подписи ребер
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
draw_flow_network(flow_network, df_nodes, route_usage, node_capacities, node_usage)  # передача df_nodes

