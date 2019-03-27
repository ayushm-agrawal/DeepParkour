import pybullet as p
import time
import math
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

p.resetSimulation()
p.resetDebugVisualizerCamera(15,-346,-16,[-1,0,1]);

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

boxHalfLength = 0.5
boxHalfWidth = 2.5
boxHalfHeight = 0.5
segmentLength = 10

colBoxId = p.createCollisionShape(p.GEOM_BOX,halfExtents=[boxHalfLength,boxHalfWidth,boxHalfHeight])

mass = 1
visualShapeId = -1

segmentStart = -75
obstacle_gap = 20
	
for i in range (segmentLength):
	height = 1
	p.createMultiBody(baseMass=0,baseCollisionShapeIndex = colBoxId,basePosition = [segmentStart+obstacle_gap,0,0.9])
	segmentStart=segmentStart+15


link_Masses=[1]
linkCollisionShapeIndices=[colBoxId]
linkVisualShapeIndices=[-1]
linkPositions=[[0,0,0]]
linkOrientations=[[0,0,0,1]]
linkInertialFramePositions=[[0,0,0]]
linkInertialFrameOrientations=[[0,0,0,1]]
indices=[0]
jointTypes=[p.JOINT_REVOLUTE]
axis=[[1,0,0]]

baseOrientation = [0,0,0,1]

p.loadURDF("plane100.urdf")
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
while (1):
	camData = p.getDebugVisualizerCamera()
	viewMat = camData[2]
	projMat = camData[3]
	p.getCameraImage(256,256,viewMatrix=viewMat, projectionMatrix=projMat, renderer=p.ER_BULLET_HARDWARE_OPENGL)
	keys = p.getKeyboardEvents()
	p.stepSimulation()	
	time.sleep(0.01)
	
