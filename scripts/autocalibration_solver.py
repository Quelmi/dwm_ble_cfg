""" 
@file: autocalibration_solver.py
@description:   python module autocalibrate anchor position based on inter-anchor ranging data
                This process takes an initial anchors coords guess as starting point of the iterative
                optimization.
@author: Esau Ortiz
@date: October 2021
@usage: python autocalibration_solver.py <nodes_cfg_label> <n_samples>
                <nodes_configuration_label> is a yaml file which includes networks, 
                tag ids, anchor ids and anchor coords
                <n_samples> samples to save when retrieving ranges
"""

from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from AutocalibrationSolver import AutocalibrationSolver
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, yaml

def readYaml(file):
    with open(file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def getData(PATH_TO_DATA, anchor_id_list, n_samples):
    """ Returns autocalibration samples
    Parameters
    ----------
    PATH_TO_DATA: string
        global path to data
    anchor_id_list: list
    n_samples: int
        number of samples (i.e. number of inter-anchor ranges)
    """
    n_anchors = len(anchor_id_list)
    n_samples -= 1 # since we discard first sample
    autocalibration_samples = np.empty((n_anchors, n_samples, n_anchors))
    for i in range(n_anchors):
        try:
            anchor_data = np.loadtxt(PATH_TO_DATA + '/' + anchor_id_list[i] + '_ranging_data.txt')
            anchor_data = anchor_data[1:] # discard first sample, usually filled with bad lectures (i.e. -1 values)
        except:
            anchor_data = -np.ones((n_samples, n_anchors))
        autocalibration_samples[i] = anchor_data
    return autocalibration_samples, n_samples

def main():

    # load nodes configuration label
    nodes_configuration_label = sys.argv[1]
    n_samples = int(sys.argv[2])
    PATH_TO_DATA = '/home/esau/catkin_ws/src/uwb_pkgs/dwm1001_drivers/autocalibration_datasets/uart/' + nodes_configuration_label

    # load anchors cfg
    current_path = Path(__file__).parent.resolve()
    dwm1001_drivers_path = str(current_path.parent.parent)
    nodes_cfg = readYaml(dwm1001_drivers_path + "/params/nodes_cfg/" + nodes_configuration_label + ".yaml")

    # set some node configuration variables    
    n_networks = nodes_cfg['n_networks']
    anchor_id_list = []
    initial_guess = [] # anchor coordinates initial guess
    anchor_coords_gt = []
    for i in range(n_networks):
        network_cfg = nodes_cfg['network' + str(i)]
        n_anchors = network_cfg['n_anchors']
        anchors_in_network_list = [network_cfg['anchor' + str(i) + '_id'] for i in range(n_anchors)]
        anchors_coords_in_network_list = [network_cfg['anchor' + str(i) + '_coordinates'] for i in range(n_anchors)]
        anchors_coords_gt_in_network_list = [network_cfg['gt']['anchor' + str(i) + '_coordinates'] for i in range(n_anchors)]
        anchor_id_list += anchors_in_network_list
        initial_guess = initial_guess + anchors_coords_in_network_list
        anchor_coords_gt = anchor_coords_gt + anchors_coords_gt_in_network_list
    initial_guess = np.array(initial_guess)
    anchor_coords_gt = np.array(anchor_coords_gt)
    n_total_anchors = len(anchor_coords_gt)

    # build fixed anchors mask
    fixed_anchor_id_list = nodes_cfg['fixed_anchors']
    fixed_anchors = np.zeros((len(anchor_id_list)), dtype = bool)
    for fixed_anchor_id in fixed_anchor_id_list:
        try:
            fixed_anchors[anchor_id_list.index(fixed_anchor_id)] = True
        except ValueError:
            continue

    # get autocalibration_samples
    autocalibration_samples, n_samples = getData(PATH_TO_DATA, anchor_id_list, n_samples)

    # solve multi-stage procedure for all k samples
    autocalibration_solver = AutocalibrationSolver(autocalibration_samples, initial_guess, fixed_anchors, lower_percentile = 0.1, upper_percentile = 0.9)
    """
    # solve stages 1 and 2 for samples' median
    autocalibration_solver.stageOne()
    autocalibration_solver.stageTwo()
    autocalibrated_coords = np.copy(autocalibration_solver.autocalibrated_coords)    
    """
    # solve stages 1 and 2 for all samples j
    autocalibrated_coords_j = np.empty((n_total_anchors, 3, n_samples), dtype = float)
    for j in range(n_samples):
        autocalibration_solver.stageOne(sample_idx = j)
        autocalibration_solver.stageTwo(sample_idx = j)
        autocalibrated_coords_j[:,:,j] = np.copy(autocalibration_solver.autocalibrated_coords)
        # print progress
        percentage_completed = "{:.2f}".format(j/n_samples * 100)
        sys.stdout.write(f'\r {percentage_completed} % Complete')
        #sys.stdout.flush()
    print('\n')

    # results df
    main_df = pd.DataFrame({
        'anchor_id' : [],
        'error [m]' : [],
        'x_error [m]' : [],
        'y_error [m]' : [],
        'z_error [m]' :  []
    })

    # plot estimation results
    fig = plt.figure()
    ax = Axes3D(fig)
    cmap = plt.get_cmap('gist_rainbow')
    my_colors = cmap(np.linspace(0,1,n_total_anchors))
    # legend plot
    ax.scatter([],[],[], label = 'estimation (fmin)', marker = 'x', color = 'black')
    ax.scatter([],[],[], label = 'ground truth', color = 'black')

    for i in range(n_total_anchors):
        # solving stages 1 and 2 for all samples j therefore there will be j coord estimations for each anchor
        estimated_anchor_coords = autocalibrated_coords_j[i,:,:].T
        centroid = np.mean(estimated_anchor_coords, axis = 0)
        # solving stages 1 and 2 for samples' median
        #centroid = autocalibrated_coords[i]
        
        ax.scatter(centroid[0], centroid[1], centroid[2], color = my_colors[i], marker = 'x')
        ax.scatter(anchor_coords_gt[i,0], anchor_coords_gt[i,1], anchor_coords_gt[i,2], color = my_colors[i], label = anchor_id_list[i])

        axis_error = anchor_coords_gt[i] - centroid
        euclidean_error = np.linalg.norm((anchor_coords_gt[i] - centroid))
        row = pd.DataFrame({'anchor_id' : [anchor_id_list[i]],
                            'error [m]' :   "{:.2f}".format(euclidean_error),
                            'x_error [m]' : "{:.2f}".format(axis_error[0]),
                            'y_error [m]' : "{:.2f}".format(axis_error[1]),
                            'z_error [m]' : "{:.2f}".format(axis_error[2])
                            })

        main_df = main_df.append(row)

    print(main_df.to_string(index=False))
    plt.legend(loc='best')
    plt.axis('auto')
    plt.show()   

if __name__ == '__main__':
    main()