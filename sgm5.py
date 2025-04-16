"""
python implementation of the semi-global matching algorithm from Stereo Processing by Semi-Global Matching
and Mutual Information (https://core.ac.uk/download/pdf/11134866.pdf) by Heiko Hirschmuller.

author: David-Alexandre Beaupre
date: 2019/07/12

MIT License

Copyright (c) 2019 David-Alexandre Beaupre

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

modified by: Steven Beninati
date: 2025/04/16
from: https://github.com/beaupreda/semi-global-matching
modification to allow input of specified values for P1 and P2 in the cost function
2024/04/25: modified to vectorize operations and improve speed - SB
2024/06/21: modified to support negative disparity values - SB
2025/02/11: 
    - moved command line arguments to wrapper
    - sgm now accepts python arguments
    - multithreaded cost aggregation function
    - disparity output is now UINT16 png instead of UINT8 png. This would allow more disparity levels for higher elevations - SB
"""

import argparse
import sys
import time as t

import cv2
import numpy as np
import multiprocessing as mp
import os
class Direction:
    def __init__(self, direction=(0, 0), name='invalid'):
        """
        represent a cardinal direction in image coordinates (top left = (0, 0) and bottom right = (1, 1)).
        :param direction: (x, y) for cardinal direction.
        :param name: common name of said direction.
        """
        self.direction = direction
        self.name = name


# 8 defined directions for sgm
N = Direction(direction=(0, -1), name='north')
NE = Direction(direction=(1, -1), name='north-east')
E = Direction(direction=(1, 0), name='east')
SE = Direction(direction=(1, 1), name='south-east')
S = Direction(direction=(0, 1), name='south')
SW = Direction(direction=(-1, 1), name='south-west')
W = Direction(direction=(-1, 0), name='west')
NW = Direction(direction=(-1, -1), name='north-west')


class Paths:
    def __init__(self):
        """
        represent the relation between the directions.
        """
        self.paths = [N, NE, E, SE, S, SW, W, NW]
        self.size = len(self.paths)
        self.effective_paths = [(E,  W), (SE, NW), (S, N), (SW, NE)]


class Parameters:
    def __init__(self, max_disparity=64, min_disparity = 0, P1=5, P2=70, csize=(7, 7), bsize=(3, 3)):
        """
        represent all parameters used in the sgm algorithm.
        :param max_disparity: maximum distance between the same pixel in both images.
        :param P1: penalty for disparity difference = 1
        :param P2: penalty for disparity difference > 1
        :param csize: size of the kernel for the census transform.
        :param bsize: size of the kernel for blurring the images and median filtering.
        """
        
        self.max_disparity = max_disparity
        self.min_disparity = min_disparity
        self.P1 = P1
        self.P2 = P2
        self.csize = csize
        self.bsize = bsize
        print("P1 is ", self.P1, "and P2 is ", self.P2)


def load_images(left_name, right_name, parameters):
    """
    read and blur stereo image pair.
    :param left_name: name of the left image.
    :param right_name: name of the right image.
    :param parameters: structure containing parameters of the algorithm.
    :return: blurred left and right images.
    """
    left = cv2.imread(left_name, 0)
    right = cv2.imread(right_name, 0)
    return left, right


def get_indices(offset, dim, direction, height):
    """
    for the diagonal directions (SE, SW, NW, NE), return the array of indices for the current slice.
    :param offset: difference with the main diagonal of the cost volume.
    :param dim: number of elements along the path.
    :param direction: current aggregation direction.
    :param height: H of the cost volume.
    :return: arrays for the y (H dimension) and x (W dimension) indices.
    """
    if direction == SE.direction:
        if offset < 0:
            y_indices=np.arange(-offset,-offset+dim)
            x_indices=np.arange(0,dim)
        else:
            y_indices=np.arange(0,dim)
            x_indices=np.arange(offset,offset + dim)

    if direction == SW.direction:
        if offset < 0:
            y_indices=np.arange(height + offset, height + offset - dim, -1)
            x_indices=np.arange(0, dim)
        else:
            y_indices=np.arange(height, height - dim, -1)
            x_indices=np.arange(offset, offset + dim)

    return y_indices, x_indices


def get_path_cost(slice, offset, parameters):
    """
    part of the aggregation step, finds the minimum costs in a D x M slice (where M = the number of pixels in the
    given direction)
    :param slice: M x D array from the cost volume.
    :param offset: ignore the pixels on the border.
    :param parameters: structure containing parameters of the algorithm.
    :return: M x D array of the minimum costs for a given slice in a given direction.
    """
    other_dim = slice.shape[0]
    disparity_dim = slice.shape[1]

    disparities = [d for d in range(disparity_dim)] * disparity_dim #multiplying lists in python makes copies of the values in the list ...
    disparities = np.array(disparities).reshape(disparity_dim, disparity_dim)

    penalties = np.zeros(shape=(disparity_dim, disparity_dim), dtype=slice.dtype)
    penalties[np.abs(disparities - disparities.T) == 1] = parameters.P1
    penalties[np.abs(disparities - disparities.T) > 1] = parameters.P2

    minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=slice.dtype)
    minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

    for i in range(offset, other_dim):
        #previous_cost = minimum_cost_path[i - 1, :] #removing unnecessary array creations
        #current_cost = slice[i, :] #remove unnecessary array creations
        costs = np.repeat(minimum_cost_path[i - 1, :], repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
        costs = np.amin(costs + penalties, axis=0)
        minimum_cost_path[i, :] = slice[i,:] + costs - np.amin(minimum_cost_path[i - 1, :])
    return minimum_cost_path

def aggregate_direction(height, width, disparities, path, parameters, path_id, cost_volume, q):
    #print('\tProcessing paths {} and {}...'.format(path[0].name, path[1].name), end='')

    main_aggregation = np.zeros(shape=(height, width, disparities), dtype=cost_volume.dtype)
    opposite_aggregation = np.copy(main_aggregation)
    start = -(height - 1)
    end = width - 1

    main = path[0]
    if main.direction == S.direction:
        for x in range(0, width):
            south = cost_volume[0:height, x, :]
            north = np.flip(south, axis=0)
            main_aggregation[:, x, :] = get_path_cost(south, 1, parameters)
            opposite_aggregation[:, x, :] = np.flip(get_path_cost(north, 1, parameters), axis=0)

    if main.direction == E.direction:
        for y in range(0, height):
            east = cost_volume[y, 0:width, :]
            west = np.flip(east, axis=0)
            main_aggregation[y, :, :] = get_path_cost(east, 1, parameters)
            opposite_aggregation[y, :, :] = np.flip(get_path_cost(west, 1, parameters), axis=0)

    if main.direction == SE.direction:
        for offset in range(start, end):
            south_east = cost_volume.diagonal(offset=offset).T
            north_west = np.flip(south_east, axis=0)
            dim = south_east.shape[0]
            y_se_idx, x_se_idx = get_indices(offset, dim, SE.direction, None)
            y_nw_idx = np.flip(y_se_idx, axis=0)
            x_nw_idx = np.flip(x_se_idx, axis=0)
            main_aggregation[y_se_idx, x_se_idx, :] = get_path_cost(south_east, 1, parameters)
            opposite_aggregation[y_nw_idx, x_nw_idx, :] = get_path_cost(north_west, 1, parameters)

    if main.direction == SW.direction:
        for offset in range(start, end):
            south_west = np.flipud(cost_volume).diagonal(offset=offset).T
            north_east = np.flip(south_west, axis=0)
            dim = south_west.shape[0]
            y_sw_idx, x_sw_idx = get_indices(offset, dim, SW.direction, height - 1)
            y_ne_idx = np.flip(y_sw_idx, axis=0)
            x_ne_idx = np.flip(x_sw_idx, axis=0)
            main_aggregation[y_sw_idx, x_sw_idx, :] = get_path_cost(south_west, 1, parameters)
            opposite_aggregation[y_ne_idx, x_ne_idx, :] = get_path_cost(north_east, 1, parameters)
    q.put((main_aggregation, opposite_aggregation, path_id))


def aggregate_costs(cost_volume, parameters, paths):
    """
    second step of the sgm algorithm, aggregates matching costs for N possible directions (8 in this case).
    :param cost_volume: array containing the matching costs.
    :param parameters: structure containing parameters of the algorithm.
    :param paths: structure containing all directions in which to aggregate costs.
    :return: H x W x D x N array of matching cost for all defined directions.
    """
    height = cost_volume.shape[0]
    width = cost_volume.shape[1]
    disparities = cost_volume.shape[2]

    aggregation_volume = np.zeros(shape=(height, width, disparities, paths.size), dtype=cost_volume.dtype)

    path_id = 0
    processes = []
    sys.stdout.flush()
    dawn = t.time()
    #start queue to receive aggregation data
    q = mp.Queue()
    #start threads for all paths
    for path in paths.effective_paths:
        processes.append(mp.Process(target = aggregate_direction, args=(height, width, disparities, path, parameters, path_id, cost_volume, q)))
        processes[-1].start()
        path_id = path_id + 2
    #get the data from the queue and put it into the aggregation volume
    for p in range(len(processes)):
        (main_aggregation, opposite_aggregation, path_id) = q.get()
        aggregation_volume[:, :, :, path_id] = main_aggregation
        aggregation_volume[:, :, :, path_id + 1] = opposite_aggregation
    
   #wait for each path to finish and join the thread 
    [p.join() for p in processes]
    dusk = t.time()
    print('\t(Cost Aggregation done in {:.2f}s)'.format(dusk - dawn))
    

    return aggregation_volume

def compute_costs(left, right, parameters, save_images, sgm_dir = '.'):
    """
    first step of the sgm algorithm, matching cost based on census transform and hamming distance.
    :param left: left image.
    :param right: right image.
    :param parameters: structure containing parameters of the algorithm.
    :param save_images: whether to save census images or not.
    :return: H x W x D array with the matching costs.
    """
    assert left.shape[0] == right.shape[0] and left.shape[1] == right.shape[1], 'left & right must have the same shape.'
    #assert parameters.max_disparity > 0, 'maximum disparity must be greater than 0.'

    height = left.shape[0]
    width = left.shape[1]
    cheight = parameters.csize[0]
    cwidth = parameters.csize[1]
    y_offset = int(cheight / 2)
    x_offset = int(cwidth / 2)
    max_disparity = parameters.max_disparity
    min_disparity = parameters.min_disparity

    left_img_census = np.zeros(shape=(height, width), dtype=np.uint8)
    right_img_census = np.zeros(shape=(height, width), dtype=np.uint8)
    left_census_values = np.zeros(shape=(height, width), dtype=np.uint64)
    right_census_values = np.zeros(shape=(height, width), dtype=np.uint64)

    print('\tComputing left and right census...', end='')
    sys.stdout.flush()
    dawn = t.time()
    for j in range(-x_offset, x_offset+1):
        for i in range(-y_offset, y_offset+1):
            if (i, j) != (0, 0):
                left_census_values = left_census_values*2
                left_census_values = left_census_values | np.uint64( (np.roll(np.roll(left.astype(np.int16), i, axis = 0), j, axis = 1) - left.astype(np.int16)) < 0)
    left_census_values[0:y_offset,:] = 0 #zero out borders
    left_census_values[height - y_offset:height-1,:] = 0 #zero out borders
    left_census_values[:,0:x_offset] = 0 #zero out borders
    left_census_values[:,width-x_offset:width-1] = 0 #zero out borders
    left_img_census = np.uint8(left_census_values)

    for j in range(-x_offset, x_offset+1):
        for i in range(-y_offset, y_offset+1):
            if (i, j) != (0, 0):
                right_census_values = right_census_values*2
                right_census_values = right_census_values | np.uint64( (np.roll(np.roll(right.astype(np.int16), i, axis = 0), j, axis = 1) - right.astype(np.int16)) < 0)
    right_census_values[0:y_offset,:] = 0 #zero out borders
    right_census_values[height - y_offset:height-1,:] = 0 #zero out borders
    right_census_values[:,0:x_offset] = 0 #zero out borders
    right_census_values[:,width-x_offset:width-1] = 0 #zero out borders
    right_img_census = np.uint8(right_census_values)

    dusk = t.time()
    print('\t(Left and Right census done in {:.2f}s)'.format(dusk - dawn))

    if save_images:
        cv2.imwrite(os.path.join(sgm_dir, 'left_census.png'), left_img_census)
        cv2.imwrite(os.path.join(sgm_dir, 'right_census.png'), right_img_census)

    print('\tComputing cost volumes...', end='')
    sys.stdout.flush()
    dawn = t.time()
    left_cost_volume = np.zeros(shape=(height, width, (max_disparity-min_disparity)), dtype=np.uint32)
    right_cost_volume = np.zeros(shape=(height, width, (max_disparity-min_disparity)), dtype=np.uint32)
    lcensus = np.zeros(shape=(height, width), dtype=np.int64)
    rcensus = np.zeros(shape=(height, width), dtype=np.int64)
    for d in range(0, max_disparity-min_disparity):
        rcensus = np.roll(right_census_values, d+min_disparity, axis =1)
        left_xor = np.int64(np.bitwise_xor(np.uint64(left_census_values), rcensus))
        #left_cost_volume[:, :, d] = np.array([[x.bit_count() for x in y] for y in left_xor.astype(int).tolist()])
        left_cost_volume[:, :, d] = np.bitwise_count(left_xor)
        lcensus = np.roll(left_census_values, -(d+min_disparity), axis =1)
        right_xor = np.int64(np.bitwise_xor(np.uint64(right_census_values), lcensus))
        #right_cost_volume[:, :, d] = np.array([[x.bit_count() for x in y] for y in right_xor.astype(int).tolist()])
        right_cost_volume[:, :, d] = np.bitwise_count(right_xor)

    dusk = t.time()
    print('\t(Cost volume computation done in {:.2f}s)'.format(dusk - dawn))

    return left_cost_volume, right_cost_volume


def select_disparity(aggregation_volume):
    """
    last step of the sgm algorithm, corresponding to equation 14 followed by winner-takes-all approach.
    :param aggregation_volume: H x W x D x N array of matching cost for all defined directions.
    :return: disparity image.
    """
    volume = np.sum(aggregation_volume, axis=3)
    disparity_map = np.argmin(volume, axis=2)
    return disparity_map


def normalize(volume, parameters):
    """
    transforms values from the range (mindisp, maxdisp) to (0, 255).
    :param volume: n dimension array to normalize.
    :param parameters: structure containing parameters of the algorithm.
    :return: normalized array.
    """
    return 65535.0 * (volume) / (parameters.max_disparity-parameters.min_disparity)


def get_recall(disparity, gt, args):
    """
    computes the recall of the disparity map.
    :param disparity: disparity image.
    :param gt: path to ground-truth image.
    :param args: program arguments.
    :return: rate of correct predictions.
    """
    gt = np.float32(cv2.imread(gt, cv2.IMREAD_GRAYSCALE))
    gt = np.int16(gt / 255.0 * float(args.maxdisp-args.mindisp) + mindisp)
    disparity = np.int16(np.float32(disparity) / 255.0 * float(args.maxdisp-agrs.mindisp) + mindisp)
    correct = np.count_nonzero(np.abs(disparity - gt) <= 3)
    return float(correct) / gt.size


#sgm_cmd.py
#command line wrapper for the sgm function
def sgm_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', default=None, help='name (path) to the left image')
    parser.add_argument('--right', default=None, help='name (path) to the right image')
    parser.add_argument('--left_gt', default=None, help='name (path) to the left ground-truth image')
    parser.add_argument('--right_gt', default=None, help='name (path) to the right ground-truth image')
    parser.add_argument('--output', default='disparity_map.png', help='name of the output image')
    parser.add_argument('--max_disp', default=64, type=int, help='maximum disparity for the stereo pair')
    parser.add_argument('--min_disp', default=0, type=int, help='minimum disparity for the stereo pair')
    parser.add_argument('--images', default=False, type=bool, help='save intermediate representations')
    parser.add_argument('--eval', default=False, type=bool, help='evaluate disparity map with 3 pixel error')
    parser.add_argument('--P1_in', default=50, type=float, help='cost for pixel disparities of 1')
    parser.add_argument('--P2_in', default=350, type=float, help='cost for pixel disparities >1')
    parser.add_argument('--c_dim', default=7, type=int, help='size of the census transform kernel')
    parser.add_argument('--b_dim', default=3, type=int, help='size of the blurring kernel')
    parser.add_argument('--sgm_dir', default='.', type=int, help='output directory for SGM data')
    args = parser.parse_args()

    left_name = args.left
    right_name = args.right
    left_gt_name = args.left_gt
    right_gt_name = args.right_gt
    output_name = args.output
    max_disparity = args.max_disp
    min_disparity = args.min_disp
    csz = args.c_dim
    bsz = args.b_dim
    save_images = args.images
    evaluation = args.eval
    P1 = args.P1_in
    P2 = args.P2_in
    sgm(left_name, right_name, left_gt_name = left_gt_name, right_gt_name = right_gt_name, output_name = output_name, max_disparity = max_disparity, min_disparity = min_disparity, save_images = save_images, evaluation = evaluation, P1 = P1, P2 = P2, csz = csz, bsz = bsz)    

def sgm(left_name, right_name, left_gt_name = None, right_gt_name = None, output_name = 'disparity_map.tif', max_disparity = 64, min_disparity = 0, save_images = False, evaluation = False, P1 = 50, P2 = 350, csz = 7, bsz = 5, sgm_dir = '.'):
    """
    main function applying the semi-global matching algorithm.
    :return: void.
    """
    dawn = t.time()
    print("bsize is ", bsz, "and csize is ", csz)
    parameters = Parameters(max_disparity=max_disparity, min_disparity=min_disparity, P1=P1, P2=P2, csize=(csz,csz), bsize=(bsz, bsz))
    paths = Paths()

    print('\nLoading images...')
    left, right = load_images(left_name, right_name, parameters)

    print('\nStarting cost computation...')
    left_cost_volume, right_cost_volume = compute_costs(left, right, parameters, save_images)
    if save_images:
        left_disparity_map = np.uint16(normalize(np.argmin(left_cost_volume, axis=2), parameters))
        cv2.imwrite('disp_map_left_cost_volume.png', left_disparity_map)
        right_disparity_map = np.uint16(normalize(np.argmin(right_cost_volume, axis=2), parameters))
        cv2.imwrite('disp_map_right_cost_volume.png', right_disparity_map)

    print('\nStarting left aggregation computation...')
    left_aggregation_volume = aggregate_costs(left_cost_volume, parameters, paths)
    print('\nStarting right aggregation computation...')
    right_aggregation_volume = aggregate_costs(right_cost_volume, parameters, paths)

    print('\nSelecting best disparities...')
    left_disparity_map = np.uint16(normalize(select_disparity(left_aggregation_volume), parameters))
    right_disparity_map = np.uint16(normalize(select_disparity(right_aggregation_volume), parameters))
    if save_images:
        cv2.imwrite('left_disp_map_no_post_processing.png', left_disparity_map)
        cv2.imwrite('right_disp_map_no_post_processing.png', right_disparity_map)

    lout = os.path.join(sgm_dir, f'left_{output_name}')
    rout = os.path.join(sgm_dir, f'right_{output_name}')
    cv2.imwrite(lout, left_disparity_map)
    cv2.imwrite(rout, right_disparity_map)

    if evaluation:
        print('\nEvaluating left disparity map...')
        recall = get_recall(left_disparity_map, left_gt_name, args)
        print('\tRecall = {:.2f}%'.format(recall * 100.0))
        print('\nEvaluating right disparity map...')
        recall = get_recall(right_disparity_map, right_gt_name, args)
        print('\tRecall = {:.2f}%'.format(recall * 100.0))

    dusk = t.time()
    print('\nFin.')
    print('\nTotal execution time = {:.2f}s'.format(dusk - dawn))
    return lout, rout

if __name__ == '__main__':
    sgm()
