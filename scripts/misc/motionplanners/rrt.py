import numpy as np
import os
import cv2
class TreeNode:
    def __init__(self,pos):
        self.pos = pos
        self.children = []
        self.parent = None

class RRT: 
    def __init__(self,start,goal,numIterations,grid,stepsize): 
        self.randomTree  = TreeNode(start)
        self.goal = TreeNode(goal)
        self.nearestnode = None
        self.iterations = numIterations
        self.grid = grid
        self.rho = stepsize
        self.path_distance = 0
        self.nearestDist = 10000
        self.numwaypoints = 0
        self.waypoints = []

    def addChild(self,pos):
        if (pos[0] == self.goal.pos[0]): 
            self.nearestnode.children.append(self.goal)
            self.goal.parent = self.nearestnode
        else: 
            tempNode = TreeNode(pos)
            self.nearestnode.children.append(tempNode)
            self.tempNode.parent = self.nearestnode

    def sample(self):
        pos = [np.random.randint(low=-self.grid.shape[0],high=self.grid.shape[0]), np.random.randint(low=-self.grid.shape[1],high=self.grid.shape[1])]
        return np.array(pos)
    
    def move(self,currPos,nextPos):
        offset = self.rho*self.unitvector(currPos,nextPos)
        point  = np.array([currPos.pos[0] + offset[0], currPos.pos[1] + offset[1]])
        if point[0] >= self.grid.shape[1]:point[0] = self.grid.shape[1]-1
        if point[1] >=  self.grid.shape[0]: point[1] = self.grid.shape[0]-1
        return point
    
    def isObstacle(self,currPos,nextPos):
        unitvector = self.unitvector(currPos,nextPos)
        testPoint = np.zeros(2)
        for i in range(self.rho):
            testPoint[0] = i*unitvector[0] + currPos.pos[0]
            testPoint[1] = i*unitvector[1] + currPos.pos[1]
            if self.grid[int(testPoint[0]),int(testPoint[1])] == 1: 
                return True
        return False
    
    def unitvector(self,currPos,nextPos):
        oa,ob = currPos.pos, nextPos
        ab = ob-oa
        ab_hat = ab/np.linalg.norm(ab)
        return ab_hat
    
    def distance(self,node1,node2): 
        pt1,pt2 = node1.pos, node2
        return np.linalg.norm(pt1-pt2)
    
    def reachedgoal(self,pos):
        if self.distance(self.goal,pos) <= self.rho: return True
        else: return False

    def reset(self):
        self.nearestDist = 10000
        self.nearestnode = None

    def backtrack(self,goal):
        if goal.pos[0] == self.randomTree.pos[0]:
            return 
        self.numwaypoints+=1
        self.waypoints.append(np.array([goal.pos[0],goal.pos[1]]))
        self.path_distance += self.rho
        self.backtrack(goal.parent)
    
    def findNearest(self,root,pos):
        ## DFS to find the nearest node. All the nodes are connected to the root node in RRT
        if not root: 
            return 
        dist = self.distance(root,pos)
        if dist <= self.nearestDist: 
            self.nearestnode = root
            self.nearestDist = dist

        for child in root.children: 
            self.findNearest(child,pos)




## Define these parameters as you want.
grid = cv2.cvtColor(cv2.imread('./obstacle.png'),cv2.COLOR_BGR2RGB)
stepSize = 10
numIterations = 200
start = (256,256)
goal =  (96,64)

if __name__ == "__main__":
    rrt = RRT(start,goal,numIterations,grid,stepSize)
    ## Implement the algorithm
    