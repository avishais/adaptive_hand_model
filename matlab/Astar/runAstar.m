clear all

addpath('../gp');

% These are in coordinates relative to the base frame (marker 0)
start = [116.131130162080,-409.232343685138,44,-44];
goal = [92.6758362418096,-410.172923332106,58,-8];

path = Astar(start, goal);