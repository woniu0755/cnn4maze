#!/usr/bin/python
import random  
      
#warning: x and y confusing  
def get_maze(w=10,h=10):      
    sx = w 
    sy = h  
    dfs = [[0 for col in range(sx)] for row in range(sy)]  
    maze = [[' ' for col in range(2*sx+1)] for row in range(2*sy+1)]  
    #1:up 2:down 3:left 4:right  
    operation = {1:(0,-1),2:(0,1),3:(-1,0),4:(1,0)}  
    direction = [1, 2, 3, 4]  
    stack = []  
      
    for i in range(2*sx+1):  
        if i%2 == 0:  
            for j in range(2*sx+1):  
                maze[i][j] = '#'  
    for i in range(2*sy+1):  
        if i%2 == 0:  
            for j in range(2*sy+1):  
                maze[j][i] = '#'  
      
     
    def generateMaze(start):  
        x, y = start  
        dfs[y][x] = 1  
        random.shuffle(direction)  
        for d in direction:  
            px, py = (x + y for x, y in zip(start, operation[d]))  
            if px < 0 or px >= sx or py < 0 or py >= sy:  
                pass  
            else:  
                if dfs[py][px] is not 1:  
                    mx = 2*x + 1  
                    my = 2*y + 1  
                    if d == 1:  
                        maze[my-1][mx] = ' '  
                    elif d == 2:  
                        maze[my+1][mx] = ' '  
                    elif d == 3:  
                        maze[my][mx-1] = ' '  
                    elif d == 4:  
                        maze[my][mx+1] = ' '  
                    generateMaze((px,py))  
    generateMaze((0,0))
    maze[0][1]=' '
    maze[-1][-2]=' '
    return maze
    
if __name__=='__main__':
	print(get_maze())
