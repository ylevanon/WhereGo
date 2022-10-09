import osmnx as ox
import pandas as pd
import networkx as nx
import warnings
import geopy
import folium
import time
import geopy.distance
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from scipy.spatial.distance import cdist
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
ox.config(use_cache=True, log_console=True)

def initialize_map(coord, distance) -> nx.MultiDiGraph:
    """returns MultiDiGraph from runners location."""
    G = ox.graph_from_point(coord, distance, network_type='walk',custom_filter='["highway"~"tertiary|unclassified|residential|tertiary_link|living_street|service|pedestrian|track|road|footway|bridleway|steps|path|sidewalk|crossing"]')
    return G

def node_database(G):
    raw_node_data = list(G.nodes(data=True))
    node_db = pd.DataFrame(raw_node_data)
    node_db = node_db.set_axis(['node from', 'codes'], axis=1, inplace=False)
    node_db = pd.concat([node_db, node_db["codes"].apply(pd.Series)], axis=1)
    node_db = node_db.drop(columns="codes")
    node_db = node_db.rename(columns={'x': 'y', 'y': 'x'})
    return node_db

def edge_database(G):
    raw_edge_data = list(G.edges(data=True))
    raw_edge_df = pd.DataFrame(raw_edge_data)
    transformed_edge_df = raw_edge_df.set_axis(['node from', 'node too', 'codes'], axis=1, inplace=False)
    edge_df = pd.concat([transformed_edge_df, transformed_edge_df["codes"].apply(pd.Series)], axis=1)
    database = edge_df.drop(columns="codes")
    return database

def get_adjacency_matrix(database):
    Gz = nx.from_pandas_edgelist(database, 'node from', 'node too', create_using=nx.DiGraph())
    adj = nx.adjacency_matrix(Gz)
    adj = nx.convert.to_dict_of_lists(Gz)
    return adj

def get_street_count_matrix(adj_matrix):
    street_dict = {x:len(adj_matrix[x]) for x in adj_matrix}
    return street_dict

def get_distance_matrix(database):
    database['from too'] = list(zip(database['node from'], database['node too']))
    dist = dict(zip(database['from too'], database['length']))
    return dist


def starting_path(G, start_node, end_node, node_db):
    if start_node == end_node:
        lat = node_db.loc[node_db['node from'] == start_node, 'x']
        lng = node_db.loc[node_db['node from'] == start_node, 'y']
        return [start_node], start_node, (lat, lng)
    intro = nx.shortest_path(G, start_node, end_node, weight='length')
    print(intro)
    new_start_node = intro[0]
    lat = node_db.loc[node_db['node from'] == new_start_node]['x']
    lng = node_db.loc[node_db['node from'] == new_start_node]['y']
    lat = lat.array[0]
    lng = lng.array[0]
    return intro, new_start_node, (lat, lng)


def gen_route_centers(lat, lng, d):
    start = geopy.Point(lat, lng)
    bearing_list = [0, 90, 180, 270]

    # converting distance from meters to kilometers

    d = d/1000

    # creating a new curved distance, making a diamond at each route center

    d = geopy.distance.geodesic(kilometers = (d / 4) * (2 ** (1 / 2)) / 2)
    route_centers = []
    for center in bearing_list:
        route_centers.append(d.destination(point=start, bearing=center))
    pointer = len(route_centers) / 2
    x = []
    y = []
    for center in route_centers:
        for i in range(len(bearing_list)):
            if i == pointer:
                x.append(start[0])
                y.append(start[1])
            else:
                loc = d.destination(point=center, bearing=bearing_list[i])
                x.append(loc[0])
                y.append(loc[1])
        pointer = (pointer + 1) % len(route_centers)
    route_corners_df =pd.DataFrame(
    {'x': x,
     'y': y,
    })
    return route_corners_df


def closest_point(point, points):
    """ Find closest point from a list of points. """
    return points[cdist([point], points).argmin()]

def match_value(df, col1, x, col2):
    """ Match value x from col1 row to value in col2. """
    return df[df[col1] == x][col2].values[0]

def find_closest_points(node_df, lat_lng_df):
    node_df['point'] = [(x, y) for x, y in zip(node_df['x'], node_df['y'])]
    lat_lng_df['point'] = [(x, y) for x,y in zip(lat_lng_df['x'], lat_lng_df['y'])]
    lat_lng_df['closest'] = [closest_point(x, list(node_df['point'])) for x in lat_lng_df['point']]
    lat_lng_df['node from'] = [match_value(node_df, 'point', x, 'node from') for x in lat_lng_df['closest']]
    route_corners_list = lat_lng_df["node from"].values.tolist()
    route_corners_list = list(split(route_corners_list, 4))
    return route_corners_list

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def BFS(street_adj, node_adj, start_node):
    print(street_adj[start_node])
    if street_adj[start_node] >= 4.0:
        return start_node
    my_queue = []
    for  i in node_adj[start_node]:
        my_queue.append(i)
    visited = [start_node]
    while len(my_queue) > 0:
        end_node = my_queue.pop(0)
        if street_adj[end_node] == 4.0:
            return end_node
        else:
            for j in node_adj[end_node]:
                if j not in visited:
                    my_queue.append(j)
            visited.append(end_node)

def path_to_start(start_node, end_node, G):
    if start_node == end_node:
        return []
    else:
        return nx.shortest_path(G, start_node, end_node)

def determine_start(start_node, new_start):
    if start_node == new_start:
        return True
    return False

def sort_routes(route_corners, lead_node):
    new_route_corners = []
    for route in route_corners:
        while route[0] != lead_node:
            to_back = route.pop(0)
            route.append(to_back)
        set_route = list(dict.fromkeys(route))
        set_route.append(lead_node)
        new_route_corners.append(set_route)
    return new_route_corners

def dijkstra_routes(G, route_corners, start_node, new_start):
    final_paths = []
    if determine_start(start_node, new_start):
        route_corners = sort_routes(route_corners, start_node)
    else:
        route_corners = sort_routes(route_corners,new_start)
    for route in route_corners:
        path = []
        for i in range(len(route) - 1):
            route1 = nx.shortest_path(G, route[i], route[i + 1], weight='length')
            for j in route1[:-1]:
                path.append(j)
        path.append(path[0])
        final_paths.append(path)
    return final_paths

def map_creation(G, lst, fol_map):
   return ox.plot_route_folium(G, lst,route_map=fol_map, color='blue', weight=5, opacity=0.7)


def loops_to_tuples(final_paths):
    final_paths_tuples = []
    for path in final_paths:
        path_tuple = []
        for i in range(len(path) - 1):
            path_tuple.append((path[i], path[i + 1]))
        final_paths_tuples.append(path_tuple)
    return final_paths_tuples

def check_cycle(trimmed_cycle_tuples):
    for path in trimmed_cycle_tuples:
        Gi = nx.Graph()
        Gi.add_edges_from(path)
        nx.draw(Gi, pos=graphviz_layout(Gi),  with_labels=True, font_size=8)
        pos = nx.spring_layout(Gi)
        nx.draw(Gi, pos, with_labels=True)
        plt.show()


def trim_ends(base_routes, start_node):
    final_routes = []
    for route in base_routes:
        path = []
        path_dict = {}
        curr = 0
        for node in route[:-1]:
            if node not in path_dict:
                path_dict[node] = curr
                path.append(node)
            else:
                path = path[:path_dict[node] + 1]
                curr = path_dict[node]
            curr += 1
        path.append(start_node)
        final_routes.append(path)
    return final_routes




def calculate_length(trimmed_cycles, dist_dict):
    trimmed_cycles_tuples = []
    cycle_list_tuple = loops_to_tuples(trimmed_cycles)
    trimmed_cycles_tuples.append(cycle_list_tuple)
    route_length_list = []
    for cycle in trimmed_cycles_tuples:
        for edge_tuple in cycle:
            route_length = 0
            for j in edge_tuple:
                route_length += dist_dict[j[0], j[1]]
            route_length_list.append(route_length)
    return route_length_list

def map_creation(G, lst, fol_map):
    return ox.plot_route_folium(G, lst,route_map=fol_map, color='blue', weight=5, opacity=0.7)

def model_builder(lat: float, lng: float, d, address):

    d = d * 1609.34
    time_start = time.time()
    G = initialize_map((lat, lng), d/3)
    print(time.time() - time_start)
    node_df = node_database(G)
    edge_df = edge_database(G)
    dist_mtrx = get_distance_matrix(edge_df)
    adj_mtrx = get_adjacency_matrix(edge_df)
    street_mtrx = get_street_count_matrix(adj_mtrx)
    

    test_map = folium.Map(location=(lat, lng), zoom_start=100, width='100%', height='55%')
    iframe = folium.IFrame(address, width=100, height=50)
    popup = folium.Popup(iframe, max_width=200)
    folium.Marker([lat, lng], popup=popup).add_to(test_map)
    
    user_location_df =pd.DataFrame({'x': [lat],'y': [lng],})

    start_node = find_closest_points(node_df,  user_location_df)[0][0]
    print(start_node)
    end_node = BFS(street_mtrx, adj_mtrx, start_node)
    print(end_node)
    intro = path_to_start(start_node, end_node, G)
    print(intro)

    if not intro:
        route_corners_coords_df = gen_route_centers(lat, lng, d)
    else:
        lat, lng = node_df.loc[node_df['node from'] == end_node, 'x'].values[0], node_df.loc[node_df['node from'] == end_node, 'y'].values[0]
        print(lat, lng)
        route_corners_coords_df = gen_route_centers(lat, lng, d)

    
    route_corner_nodes = find_closest_points(node_df, route_corners_coords_df)


    base_routes = dijkstra_routes(G, route_corner_nodes, start_node, end_node)

    final_routes = trim_ends(base_routes, end_node)
    print(final_routes)
    route_length_list = calculate_length(final_routes, dist_mtrx)
    final_length = 0
    index = 0
    for i in range(len(route_length_list)):
        if abs(route_length_list[i] - d) < abs(final_length - d):
            final_length = route_length_list[i]
            index = i
    final_tour = final_routes[index]

    print(final_tour)
    test_map = map_creation(G, final_tour, test_map)
    test_map.save('/app/templates/run.html')

    return G, final_tour, round(final_length/1609.34, 2), lat, lng, address