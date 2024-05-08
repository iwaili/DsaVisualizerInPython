import io
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from django.conf import settings 
import re 
from django.http import FileResponse
import os
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib import messages
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.models import User, auth
from django.contrib.auth.decorators import login_required
from .models import Profile
from .models import SavedGraph

@login_required(login_url='signin')
def rulesForMatrix(request):
   return render(request,'rulesForMatrix.html')

def index(request):
    return render(request,'index.html')
def signup(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        confirmpassword = request.POST['confirmPassword']

        if password == confirmpassword:
            if User.objects.filter(username=username).exists():
                messages.info(request,'Username already taken')
                return redirect('signup')
            else:
                check=0
                if len(password) < 8:
                    check=1
                # Check if password contains at least one digit
                elif not re.search(r'\d', password):
                    check=1
                # Check if password contains at least one special character
                elif not re.search(r'[!@#$%^&*()-_+=]', password):
                    check=1
                if check==1:
                    messages.info(request,'Password is not strong enough ')
                    return redirect('signup')
                else:
                    user = User.objects.create_user(username=username,password=password)
                    user.save( )
                    user_model = User.objects.get(username = username)#profile belongs to which user
                    new_profile = Profile.objects.create(user=user_model,id_user=user_model.id)
                    new_profile.save()
                    messages.info(request,'Signup Complete')
                    return redirect('signin')
                

        else:
            messages.info(request,'Passwords not matching')
            return redirect('signup')
    else: 
        return render(request,'signup.html')

def signin(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(username = username,password = password)

        if user is not None:
            auth.login(request,user)
            return render(request,'index.html')
        else:
            messages.info(request,'Credentials are Incorrect')
            return redirect('signin')
    else:
        return render(request,'signin.html')

@login_required(login_url='signin')
def logout(request):
    auth.logout(request)
    messages.success(request,'You were logged out')
    return redirect('signin')

def about(request):
    return render(request,'about.html')

@login_required(login_url='signin')
def actual(request):
    return render(request,'actual.html')

#----------------------------------------------------------------------------------------------

xv=0
def addXV():
  global xv
  xv=xv+1


def r(adjacency_matrix):
  checkArray=[0] * (len(adjacency_matrix))
  print(checkArray)
  u=0
  m=0
  for i in adjacency_matrix:
    u=0
    for j in i:
      if j!=0:
        checkArray[u]=checkArray[u]+1
        checkArray[m]=checkArray[m]+1
      u=u+1
    m=m+1
  print(checkArray)

def drawInitialGraph(adjacency_matrix):
  print(adjacency_matrix)
  G = nx.from_numpy_array(adjacency_matrix)
   # Position nodes using the spring layout algorithm
  #nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=10)
  # Set edge weights
  weights = {(i, j): adjacency_matrix[i][j] for i in range(len(adjacency_matrix)) for j in range(len(adjacency_matrix)) if adjacency_matrix[i][j] != 0}
  nx.set_edge_attributes(G, weights, 'weight')
  pos = nx.spring_layout(G)
  # Draw edge labels
  #nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

  coordinates_list = []
  for node, coord in pos.items():
      coordinates_list.append((node, coord))

  #TO REMOVE NEGATIVE VALUES FROM coodinates_list
  cord=[]
  min1=0
  min2=0
  for a in coordinates_list:
    print(a[1][0])
    cord.append((a[1][0],a[1][1]))
    if min1 > a[1][0]:
      min1 = a[1][0]
    if min2 > a[1][1]:
      min2 = a[1][1]
  print(cord)

  #TO MAKE COORDINATES OF cord INTEGER NOT FLOAT
  coord=[]
  for a in cord:
    coord.append((int(100*(a[0]+(-min1))),int(100*(a[1]+(-min2)))))
  print("hellllo",coord)

  coordinates={}
  for i, c in enumerate(coord, start=0):
      coordinates[i] = c
  print(coordinates)
  edges = []
  # Iterate over the adjacency matrix to find edges
  for i in range(len(adjacency_matrix)):
      for j in range(i+1, len(adjacency_matrix)):
          if adjacency_matrix[i][j] != 0:
              edge = (coordinates[i], coordinates[j])
              edges.append(edge)

  # Print the list of edges
  print("Edges:")
  for edge in edges:
      print(edge)

  lines = []
  print('-------------------------')
  print(len(edge))
  for tr in range(len(edges)):
    lines.append(edges[tr])
  print(lines)

  rt=["1","2","3","4","5","6","7"]
  ui=0
  # Create a plot
  plt.figure()
  up=0
  for y in coord:
    plt.scatter(y[0], y[1], s=100, color='black')
    plt.text(y[0], y[1],str(up), fontsize=14, color='red', ha='center', va='bottom')
    up=up+1
  # Plot each line and calculate distance
  for line in lines:
      x_values = [point[0] for point in line]
      y_values = [point[1] for point in line]
      plt.plot(x_values, y_values, color='black', marker=None)

      # Calculate distance between points
      distance = np.sqrt((x_values[1] - x_values[0])*2 + (y_values[1] - y_values[0])*2)

      # Display distance on the graph
      x_center = (x_values[0] + x_values[1]) / 2
      y_center = (y_values[0] + y_values[1]) / 2

      plt.text(x_center, y_center,str(int(distance)), ha='center', va='bottom')
      ui=ui+1
  # Hide the axes
  plt.axis('off')

  # Show the plot
  plt.show()

  isItOk = input("is the graph ok? ")
  if isItOk=="1":
    print("We will now save this info ")
    tempEdgeInfo=[]
    for i in range(len(adjacency_matrix)):
          for j in range(len(adjacency_matrix)):
            if adjacency_matrix[i][j]!=0:
              tempEdgeInfo.append({i,j})
    print(tempEdgeInfo)
    color_list = ["black"] * len(adjacency_matrix)
    graphInfo = {
        "nodeCoordinates" : coordinates,
        "EdgeInfo" : tempEdgeInfo,
        "EdgeColor" : color_list
    }
    print(graphInfo)
    return graphInfo
  else:
    drawInitialGraph(adjacency_matrix)

def makeInitialInfo(adjacency_matrix):
  name=[]
  for i in range(len(adjacency_matrix)):
    name.append(chr(ord('A')+i))
  G = nx.from_numpy_array(adjacency_matrix)
  pos = nx.spring_layout(G)

  coordinates_list = []
  for node, coord in pos.items():
    coordinates_list.append((node, coord))
  print(pos)
  cord =[]
  min1=0
  min2=0
  for a in coordinates_list:
    print(a[1][0])
    cord.append((a[1][0],a[1][1]))
    if min1 > a[1][0]:
      min1 = a[1][0]
    if min2 > a[1][1]:
      min2 = a[1][1]
  print(cord)
  coord=[]
  for a in cord:
    coord.append((int(100*(a[0]+(-min1))),int(100*(a[1]+(-min2)))))
  print(coord)

  ij='0'
  hj=0
  coordinates={}
  for i, c in enumerate(coord, start=0):
      coordinates[i] = c
  print(coordinates)

  edges = []

  # Iterate over the adjacency matrix to find edges
  for i in range(len(adjacency_matrix)):
      for j in range(i+1, len(adjacency_matrix)):
          if adjacency_matrix[i][j] != 0:
              edge = (coordinates[i], coordinates[j])
              edges.append(edge)
  print("Edges:")
  for edge in edges:
      print(edge)

  lines = []
  print('-------------------------')
  print(len(edge))
  for tr in range(len(edges)):
    lines.append(edges[tr])
  print(lines)
  plt.figure()
  tempInt=-1
  for y in coord:
    tempInt=tempInt+1
    plt.scatter(y[0], y[1], s=100, color='black')
    plt.text(y[0], y[1],name[tempInt], fontsize=14, color='red', ha='center', va='bottom')
  for line in lines:
      x_values = [point[0] for point in line]
      y_values = [point[1] for point in line]
      plt.plot(x_values, y_values,color="black", marker=None)
      distance = np.sqrt((x_values[1] - x_values[0])*2 + (y_values[1] - y_values[0])*2)
      x_center = (x_values[0] + x_values[1]) / 2
      y_center = (y_values[0] + y_values[1]) / 2
      plt.text(x_center, y_center," ", ha='center', va='bottom')
  plt.axis('off')
  tempEdgeInfo=[]
  noOfEdges=0
  for i in range(len(adjacency_matrix)):
    for j in range(len(adjacency_matrix)):
      if adjacency_matrix[i][j]!=0:
        tempEdgeInfo.append({i,j})
        noOfEdges=noOfEdges+1
  tempEdgeColor = {}
  tempTextOnEdge = {}
  TempToShowEdgeOrNot = {}
  for i in tempEdgeInfo:
    j=list(i)
    tempEdgeInfo1 = f'({j[0]},{j[1]})'
    tempEdgeColor[tempEdgeInfo1]="darkslategrey"
    tempTextOnEdge[tempEdgeInfo1] = 'None'
    TempToShowEdgeOrNot[tempEdgeInfo1] = '1'
  tempSphereColor = {}
  for i in coord:
    j=list(i)
    tempSphereInfo1 = f'({j[0]},{j[1]})'
    tempSphereColor[tempSphereInfo1] = 'black'
  tempRedList = ["red"] * noOfEdges
  tempTextOnEdge = ["temp"] * noOfEdges
  TempToShowEdgeOrNot = ['1'] * noOfEdges
  info = {
      "CoOrdinatesOfVertices" : coord ,
      "verticeNames" : name,
      "edges" : tempEdgeInfo,
      "edgeColor" : tempEdgeColor,
      "sphereColor" : tempEdgeColor,
      "textOnEdge" : tempTextOnEdge ,
      "toShowEdgeOrNot" : TempToShowEdgeOrNot
  }
  checkArray=[0] * (len(adjacency_matrix))
  print(checkArray)
  u=0
  m=0
  for i in adjacency_matrix:
    u=0
    for j in i:
      if j!=0:
        checkArray[u]=checkArray[u]+1
        checkArray[m]=checkArray[m]+1
      u=u+1
    m=m+1
  j=0
  for i in checkArray:
    print(i)
    if i==0:
      del info['CoOrdinatesOfVertices'][j]
    j=j+1
  print(info)
  return info


'''
info = {
  vertices :
  name :
  edges :
  color :
  sphere color :
}
'''
import os
import matplotlib.pyplot as plt
from django.conf import settings
from .models import SavedGraph

def drawGraph(info, username, requestno, ab):
    tempInt = 0
    plt.figure(facecolor='black')  # Set background color to black
    for i, vertex in enumerate(info['CoOrdinatesOfVertices']):
        plt.scatter(vertex[0], vertex[1], s=100, color='yellow')
        plt.text(vertex[0], vertex[1]+2, info['verticeNames'][tempInt], fontsize=14, color='red', ha='center', va='bottom')
        tempInt += 1

    tempInt = -1
    for edge1 in info['edges']:
        tempInt += 1
        edge = list(edge1)
        print(tempInt)
        x_values = [info['CoOrdinatesOfVertices'][edge[0]][0], info['CoOrdinatesOfVertices'][edge[1]][0]]
        y_values = [info['CoOrdinatesOfVertices'][edge[0]][1], info['CoOrdinatesOfVertices'][edge[1]][1]]
        tempEdgeInfo1 = f'({edge[0]},{edge[1]})'
        plt.plot(x_values, y_values, color=info['edgeColor'][tempEdgeInfo1], marker=None)
        x_center = (x_values[0] + x_values[1]) / 2
        y_center = (y_values[0] + y_values[1]) / 2
        if info['edgeColor'][tempEdgeInfo1]=='blue':
           plt.text(x_center, y_center, info['textOnEdge'][tempInt], color='red', ha='center', va='bottom')
        else:
           plt.text(x_center, y_center, info['textOnEdge'][tempInt], color='olive', ha='center', va='bottom')

    filename = f"{username}_{requestno}_{ab}.png"

    # Define the file path where the image will be saved
    filepath = os.path.join(settings.MEDIA_ROOT, 'graphs', filename)

    # Turn off axis
    plt.axis('off')

    # Save the figure
    plt.savefig(filepath)

    # Clear the current figure
    plt.clf()  # Clear the current figure to release memory

    # Create an instance of the SavedGraph model and save it to the database
    saved_graph = SavedGraph.objects.create(
        username=username,
        requestno=requestno,
        ab=ab,
        image=f'graphs/{filename}'
    )
    return filename
def BFS(bfs_adjacency_matrix, start_node, username, requestno):
    ab=0
    graphs=[]
    n = len(bfs_adjacency_matrix)
    visited = [False] * n
    queue = [start_node]
    visited[start_node] = True
    parent = [-1] * n  # To store the parent of each node in the BFS tree

    bfs_adjacency_matrix_np = np.array(bfs_adjacency_matrix)  # Convert the list to a NumPy array
    info = makeInitialInfo(bfs_adjacency_matrix_np)

    while queue:
        curr_node = queue.pop(0)
        graphs.append(draw_bfs_step(info, visited, parent, curr_node, username, requestno, ab))
        ab+=1
        for neighbor in range(n):
            if bfs_adjacency_matrix[curr_node][neighbor] != 0 and not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = curr_node
                queue.append(neighbor)
    return graphs

def draw_bfs_step(info, visited, parent, curr_node, username, requestno,ab):
    tempInt=-1
    for edge1 in info['edges']:
        tempInt+=1
        info['textOnEdge'][tempInt]=" "
        edge = list(edge1)
        u, v = edge
        if visited[u] and visited[v]:
            info['edgeColor'][f'({u},{v})'] = 'green'
        elif ((visited[u] and parent[v] == u) or (visited[v] and parent[u] == v)):
            info['edgeColor'][f'({u},{v})'] = 'red'
        else:
            info['edgeColor'][f'({u},{v})'] = 'blue'

    for i in range(len(visited)):
        if visited[i]:
            info['sphereColor'][f'({info["CoOrdinatesOfVertices"][i][0]},{info["CoOrdinatesOfVertices"][i][1]})'] = 'green'
        else:
            info['sphereColor'][f'({info["CoOrdinatesOfVertices"][i][0]},{info["CoOrdinatesOfVertices"][i][1]})'] = 'red'
    return drawGraph(info, username, requestno, ab)

'''{'CoOrdinatesOfVertices': [(93, 53), (193, 82), (7, 60), (182, 16), (0, 0), (83, 28)],
 'verticeNames': ['A', 'B', 'C', 'D', 'E', 'F'],
 'edges': [{0, 1}, {0, 2}, {0, 3}, {0, 5}, {1, 3}, {2, 4}, {3, 5}, {4, 5}],
 'edgeColor': {'(0,1)': 'blue', '(0,2)': 'blue', '(0,3)': 'blue', '(0,5)': 'blue',
 '(1,3)': 'blue', '(2,4)': 'blue', '(3,5)': 'blue', '(4,5)': 'blue'},
 'sphereColor': {'(0,1)': 'blue', '(0,2)': 'blue', '(0,3)': 'blue', '(0,5)': 'blue', '(1,3)': 'blue', '(2,4)': 'blue', '(3,5)': 'blue', '(4,5)': 'blue'},
 'textOnEdge': ['temp', 'temp', 'temp', 'temp', 'temp', 'temp', 'temp', 'temp'],
 'toShowEdgeOrNot': ['1', '1', '1', '1', '1', '1', '1', '1']}'''

def kruskal(matrix,username,requestno):
    ab=0
    graphs=[]
    list_of_all_nodes_we_can_visit_from_this_node = []
    for i in range(len(matrix)):
        list_of_all_nodes_we_can_visit_from_this_node.append([])
    print(list_of_all_nodes_we_can_visit_from_this_node)
    info = makeInitialInfo(matrix)
    print(info)
    disOfEdges = {}
    tempInt=-1
    for edge1 in info['edges']:
        tempInt+=1
        edge = list(edge1)
        x_values = [info['CoOrdinatesOfVertices'][edge[0]][0], info['CoOrdinatesOfVertices'][edge[1]][0]]
        y_values = [info['CoOrdinatesOfVertices'][edge[0]][1], info['CoOrdinatesOfVertices'][edge[1]][1]]
        distance_squared = (x_values[1] - x_values[0])**2 + (y_values[1] - y_values[0])**2
        if distance_squared >= 0:
            distance = int(np.sqrt(distance_squared))
            tempEdgeInfo = f'({edge[0]},{edge[1]})'
            disOfEdges[tempEdgeInfo] = distance
            info['textOnEdge'][tempInt] = str(distance)
        else:
            print(f"Skipping edge {edge}: distance calculation resulted in a negative value.")
    print("--------",info)
    tempDisOfEdges = disOfEdges
    disOfEdges = dict(sorted(tempDisOfEdges.items(), key=lambda item: item[1]))
    allVisited = 0
    for key, value in disOfEdges.items():
        print(allVisited)
        if allVisited == len(info['verticeNames']) - 1:
            graphs.append(drawGraph(info,username,requestno,ab))
            print('tree is complete')
            break
        print(list_of_all_nodes_we_can_visit_from_this_node)
        isCycle = 0
        list_of_all_nodes_we_can_visit_from_this_node[int(key[1])].append(int(key[3]))
        list_of_all_nodes_we_can_visit_from_this_node[int(key[3])].append(int(key[1]))
        tempInt = 0
        for tempList in list_of_all_nodes_we_can_visit_from_this_node:
            if tempInt in tempList:
                print('Cycle is present')
                break
        print(disOfEdges)
        if isCycle == 0:
            info['edgeColor'][key] = 'blue'
            allVisited = allVisited + 1
        graphs.append(drawGraph(info,username,requestno,ab))
        ab+=1
    print(disOfEdges)
    print(graphs)
    return graphs
import os
import matplotlib.pyplot as plt
import numpy as np
from queue import PriorityQueue

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from heapq import heappop, heappush

def dijkstra(adj_matrix, start,username,requestno):
    graphs=[]
    ab=0
    num_vertices = len(adj_matrix)
    distances = {v: float('inf') for v in range(num_vertices)}
    distances[start] = 0
    visited = set()
    pq = [(0, start)]

    step = 0  # Step counter

    while pq:
        distance, vertex = heappop(pq)
        if vertex in visited:
            continue
        visited.add(vertex)

        plt.figure(figsize=(10, 6))
        G = nx.from_numpy_array(adj_matrix)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=10)

        nx.draw_networkx_nodes(G, pos, nodelist=[vertex], node_color='red', node_size=1500)

        # Annotate current step and distances
        plt.text(0.5, 1.05, f"Step {step}: Exploring vertex {vertex}", horizontalalignment='center', fontsize=14, transform=plt.gca().transAxes)
        plt.text(0.5, 1.00, "Shortest distances:", horizontalalignment='center', fontsize=12, transform=plt.gca().transAxes)
        for v, d in distances.items():
            plt.text(pos[v][0], pos[v][1] + 0.05, f"({d})", horizontalalignment='center', fontsize=10)
        filename = f"{username}_{requestno}_{ab}.png"
        ab+=1
        # Define the file path where the image will be saved
        filepath = os.path.join(settings.MEDIA_ROOT, 'graphs', filename)

        # Turn off axis
        plt.axis('off')

        # Save the figure
        plt.savefig(filepath)

        # Clear the current figure
        plt.clf()  # Clear the current figure to release memory

        # Create an instance of the SavedGraph model and save it to the database
        saved_graph = SavedGraph.objects.create(
            username=username,
            requestno=requestno,
            ab=ab,
            image=f'graphs/{filename}'
        )
        graphs.append(filename)
        plt.close()

        for neighbor, weight in enumerate(adj_matrix[vertex]):
            if weight > 0 and distances[vertex] + weight < distances[neighbor]:
                distances[neighbor] = distances[vertex] + weight
                heappush(pq, (distances[neighbor], neighbor))

        step += 1

    return graphs

# Example adjacency matrix (replace with your own)




def generate_adjacency_matrix(matrix):
    # Split the text input by lines and then split each line by whitespace to get the individual elements
    rows = [list(map(int, filter(None, line.split()))) for line in matrix.split('\n')]
    # Create the adjacency matrix using the generate_adjacency_matrix function
    adj_matrix = np.array(rows, dtype=int)
    return adj_matrix

def process_data(request):
    if request.method == 'POST':
        text_input = request.POST.get('text_input')
        file_input = request.FILES.get('file_input')
        if not text_input:
          messages.info(request, '')
          return redirect('actual')
        
        selected_button = request.POST.get('button')
        # Process the text and file data as needed
        if text_input:
          adj_matrix = generate_adjacency_matrix(text_input)
          print(adj_matrix)
          print("Text Input:", text_input)
          username = request.user.username
          profile = Profile.objects.get(user=request.user)
          profile.incnum()
          requestno = profile.num()
          print("Selected Button:", selected_button)
          if selected_button=="Button 1":
             graphs = BFS(adj_matrix,0,username,requestno)
          elif selected_button=="Button 2":
             graphs = kruskal(adj_matrix,username,requestno)
          elif selected_button=="Button 3":
             graphs = dijkstra(adj_matrix,0,username,requestno)
          print(graphs)
          profile.incnum()
          profile.save()
          return redirect('show', username=username, requestno=requestno)
        
        if file_input:
            print("File Input:", file_input.name)
        
        return render(request, 'actual.html')
    else:
        return HttpResponse('Invalid request method')
    
def show(request, username, requestno):
    media_path = os.path.join(settings.MEDIA_ROOT, 'graphs')
    filenames = [filename for filename in os.listdir(media_path) if filename.startswith(f"{username}_{requestno}_")]
    return render(request, 'show.html', {'username': username, 'requestno': requestno, 'filenames': filenames})
