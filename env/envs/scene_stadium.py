""" This file has been taken from the PyBullet repo """
import inspect
import os
import random

import numpy as np
import pybullet_data
from pybullet_envs.scene_abstract import Scene

import pybullet

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)


class StadiumScene(Scene):
    # if False, center of coordinates (0,0,0) will be at the middle of the stadium
    zero_at_running_strip_start_line = True
    stadium_halflen = 105*0.25  # FOOBALL_FIELD_HALFLEN
    stadium_halfwidth = 50*0.25	 # FOOBALL_FIELD_HALFWID
    stadiumLoaded = 0

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        # contains cpp_world.clean_everything()
        Scene.episode_restart(self, bullet_client)
        if (self.stadiumLoaded == 0):
            self.stadiumLoaded = 1

            filename = os.path.join(
                pybullet_data.getDataPath(), "plane_stadium.sdf")
            self.ground_plane_mjcf = self._p.loadSDF(filename)

            # inserts blocks in the environment shape of the block
            obstacleLength = 0.1
            obstacleWidth = 1.5
            obstacleHeight = 0.3
            num_of_obstacles = 10

            # create a block
            obstacleId = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[
                obstacleLength, obstacleWidth, obstacleHeight])

            # initial position of the block
            position = -5
            # gaps between blocks
            obstacle_gap = 7.5

            # creates all obstacles in the environment
            for i in range(num_of_obstacles):
                pybullet.createMultiBody(baseMass=0, baseCollisionShapeIndex=obstacleId, basePosition=[
                                         position+obstacle_gap, 0, 0])
                # update the position of the next obstacle
                position = position+10

            for i in self.ground_plane_mjcf:
                self._p.changeDynamics(
                    i, -1, lateralFriction=0.8, restitution=0.5)
                self._p.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.8])
                self._p.configureDebugVisualizer(
                    pybullet.COV_ENABLE_PLANAR_REFLECTION, 1)


class SinglePlayerStadiumScene(StadiumScene):
    "This scene created by environment, to work in a way as if there was no concept of scene visible to user."
    multiplayer = False


class MultiplayerStadiumScene(StadiumScene):
    multiplayer = True
    players_count = 3

    def actor_introduce(self, robot):
        StadiumScene.actor_introduce(self, robot)
        i = robot.player_n - 1  # 0 1 2 => -1 0 +1
        robot.move_robot(0, i, 0)
