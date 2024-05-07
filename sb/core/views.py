from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib import messages
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.models import User, auth
from django.contrib.auth.decorators import login_required
from .models import Profile
import io
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re 
from django.http import FileResponse
import os
import datetime

def index(request):
    if request.user.is_authenticated:
        user_profile = Profile.objects.get(user=request.user)
        return render(request, 'index.html', {'user_profile': user_profile})
    else:
        return render(request, 'index.html')
    
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User
from .models import Profile
import re

def signup(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        confirmpassword = request.POST['confirmPassword']
        profileimg = request.FILES.get('image')  # Access image from request.FILES
        if password == confirmpassword:
            if User.objects.filter(username=username).exists():
                messages.info(request, 'Username already taken')
                return redirect('signup')
            else:
                check = 0
                if len(password) < 8:
                    check = 1
                # Check if password contains at least one digit
                elif not re.search(r'\d', password):
                    check = 1
                # Check if password contains at least one special character
                elif not re.search(r'[!@#$%^&*()-_+=]', password):
                    check = 1
                if check == 1:
                    messages.info(request, 'Password is not strong enough ')
                    return redirect('signup')
                else:
                    user = User.objects.create_user(username=username, password=password)
                    user.save()
                    new_profile = Profile.objects.create(user=user, id_user=user.id, profileimg=profileimg)
                    messages.info(request, 'Signup Complete')
                    return redirect('signin')

        else:
            messages.info(request, 'Passwords not matching')
            return redirect('signup')
    else:
        return render(request, 'signup.html')


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
  plt.show()
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
    tempEdgeColor[tempEdgeInfo1]="blue"
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
def drawGraph(info,username):
  xvi=0
  tempInt=0
  for i, vertex in enumerate(info['CoOrdinatesOfVertices']):
    plt.scatter(vertex[0], vertex[1], s=100, color='red')
    plt.text(vertex[0], vertex[1], info['verticeNames'][tempInt], fontsize=14, color='black', ha='center', va='bottom')
    tempInt=tempInt+1
  tempInt=0
  for edge1 in info['edges']:
    edge = list(edge1)
    print(tempInt)
    x_values = [info['CoOrdinatesOfVertices'][edge[0]][0], info['CoOrdinatesOfVertices'][edge[1]][0]]
    y_values = [info['CoOrdinatesOfVertices'][edge[0]][1], info['CoOrdinatesOfVertices'][edge[1]][1]]
    tempEdgeInfo1 = f'({edge[0]},{edge[1]})'
    plt.plot(x_values, y_values, color = info['edgeColor'][tempEdgeInfo1], marker=None)
    x_center = (x_values[0] + x_values[1]) / 2
    y_center = (y_values[0] + y_values[1]) / 2
    plt.text(x_center, y_center," ", color='blue', ha='center', va='bottom')
    tempInt=tempInt+1
    '''
  for edge1, color in zip(info['edges'], info['edgeColor']):
    edge = list(edge1)
    print(tempInt)
    x_values = [info['CoOrdinatesOfVertices'][edge[0]][0], info['CoOrdinatesOfVertices'][edge[1]][0]]
    y_values = [info['CoOrdinatesOfVertices'][edge[0]][1], info['CoOrdinatesOfVertices'][edge[1]][1]]
    tempEdgeInfo1 = f'({edge[0]},{edge[1]})'
    plt.plot(x_values, y_values, color = info['edgeColor'][tempEdgeInfo1], marker=None)
    x_center = (x_values[0] + x_values[1]) / 2
    y_center = (y_values[0] + y_values[1]) / 2
    plt.text(x_center, y_center," ", color='blue', ha='center', va='bottom')
    tempInt=tempInt+1
    '''
  plt.axis('off')
  # Retrieve the username (assuming you have a logged-in user)
        
        # Generate the current date
  current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Construct a unique filename
  filename = f"{username}_{current_date}"
  addXV()
  plt.savefig(filename)
  print(f"Saved graph as {filename}")
  plt.show()
  return filename

def kruskal(matrix,username):
    graphs=[]
    list_of_all_nodes_we_can_visit_from_this_node = []
    for i in range(len(matrix)):
        list_of_all_nodes_we_can_visit_from_this_node.append([])
    print(list_of_all_nodes_we_can_visit_from_this_node)
    info = makeInitialInfo(matrix)
    disOfEdges = {}
    for edge1 in info['edges']:
        edge = list(edge1)
        x_values = [info['CoOrdinatesOfVertices'][edge[0]][0], info['CoOrdinatesOfVertices'][edge[1]][0]]
        y_values = [info['CoOrdinatesOfVertices'][edge[0]][1], info['CoOrdinatesOfVertices'][edge[1]][1]]
        distance_squared = (x_values[1] - x_values[0])**2 + (y_values[1] - y_values[0])**2
        if distance_squared >= 0:
            distance = int(np.sqrt(distance_squared))
            tempEdgeInfo = f'({edge[0]},{edge[1]})'
            disOfEdges[tempEdgeInfo] = distance
        else:
            print(f"Skipping edge {edge}: distance calculation resulted in a negative value.")
    tempDisOfEdges = disOfEdges
    disOfEdges = dict(sorted(tempDisOfEdges.items(), key=lambda item: item[1]))
    allVisited = 0
    for key, value in disOfEdges.items():
        print(allVisited)
        if allVisited == len(info['verticeNames']) - 1:
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
            info['edgeColor'][key] = 'green'
            allVisited = allVisited + 1
        graphs.append(drawGraph(info,username))
    print(disOfEdges)
    return graphs
  #drawGraph(info)

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
        selected_button = request.POST.get('button')
        
        # Process the text and file data as needed
        if text_input:
            adj_matrix = generate_adjacency_matrix(text_input)
            print(adj_matrix)
            print("Text Input:", text_input)
            username = request.user.username
            profile = Profile.objects.get(user=request.user)
            requestno = profile.num()
            return HttpResponse(f'Text input: {username}, Selected button: {requestno}')
            graphs = kruskal(adj_matrix,username)
            Profile.incnum()
            
        
        if file_input:
            print("File Input:", file_input.name)
        
        return HttpResponse(f'Text input: {adj_matrix}, Selected button: {selected_button}')
    else:
        return HttpResponse('Invalid request method')

