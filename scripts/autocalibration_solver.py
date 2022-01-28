""" 
@file: autocalibration_solver.py
@description:   python module autocalibrate anchor position based on inter-anchor ranging data
                This process takes an initial anchors coords guess as starting point of the iterative
                optimization.
@author: Esau Ortiz
@date: October 2021
@usage: python autocalibration_solver.py <nodes_cfg_label> <n_samples>
                <nodes_cfg_label> is a yaml file which includes networks, 
                tag ids, anchor ids and anchor coords
                <n_samples> samples to save when retrieving ranges
"""

from yaml.error import Mark
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from AutocalibrationSolver import AutocalibrationSolver
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, yaml, random

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
        list of anchor identifiers
    n_samples: int
        number of samples (i.e. number of inter-anchor ranges)
    """
    n_anchors = len(anchor_id_list)
    n_discarded_samples = 1
    n_samples -= n_discarded_samples # since we discard first sample
    autocalibration_samples = np.empty((n_anchors, n_samples, n_anchors))
    for i in range(n_anchors):
        try:
            anchor_data = np.loadtxt(PATH_TO_DATA + '/' + anchor_id_list[i] + '_ranging_data.txt')
            anchor_data = anchor_data[n_discarded_samples:] # discard first sample, usually filled with bad lectures (i.e. -1 values)
        except:
            anchor_data = -np.ones((n_samples, n_anchors))
        autocalibration_samples[i] = anchor_data
    return autocalibration_samples, n_samples

def main():

    # load nodes configuration label
    nodes_cfg_label = sys.argv[1]
    n_samples = int(sys.argv[2])

    # load anchors cfg
    current_path = Path(__file__).parent.resolve()
    dwm1001_drivers_path = str(current_path.parent.parent)
    nodes_cfg = readYaml(dwm1001_drivers_path + "/params/nodes_cfg/" + nodes_cfg_label + ".yaml")
    PATH_TO_DATA = dwm1001_drivers_path + '/autocalibration_datasets/uart/' + nodes_cfg_label

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

    # plot estimation results
    fig = plt.figure()
    ax = Axes3D(fig)
    cmap = plt.get_cmap('gist_rainbow')
    my_colors = cmap(np.linspace(0,1,n_total_anchors))
    my_colors_light = np.copy(my_colors)
    my_colors_light[:,3] = 0.25 # adjusting transparency

    # legend plot
    ax.scatter([],[],[], label = 'estimation', marker = 'x', color = 'black')
    ax.scatter([],[],[], label = 'initial guess', marker = '*', color = 'black')
    ax.scatter([],[],[], label = 'ground truth', color = 'black')
    intensities = np.linspace(0,1,n_samples) # color intensities to plot each estimation

    # add noise to initial_guess (initial_guess is set to anchor_coords_gt and then noise is added to emulate not accurate initial guessing)
    for i, is_fixed in zip(range(n_total_anchors), fixed_anchors):
        if is_fixed: continue
        for j in range(2): # add initial_error to x and y axis
            ERROR_MAGNITUDE = 2.0
            initial_error = float(random.randrange(-1,2)) * ERROR_MAGNITUDE
            while initial_error == 0.0: initial_error = float(random.randrange(-1,2)) * ERROR_MAGNITUDE
            initial_guess[i, j] = initial_guess[i, j] + initial_error

    # here we compute anchor coordinates based on inter-anchor ranges
    # solve multi-stage procedure for all k samples
    autocalibration_solver = AutocalibrationSolver(autocalibration_samples, np.copy(initial_guess), fixed_anchors)

    # solve stages 1 and 2 for samples' median (i.e. for a multiple samples of anchorA-anchorB we use the median value)
    autocalibration_solver.stageOne()
    autocalibrated_coords_stage_one = np.copy(autocalibration_solver.autocalibrated_coords)    
    autocalibration_solver.stageTwo()
    autocalibrated_coords = np.copy(autocalibration_solver.autocalibrated_coords)

    """
    # solve stages 1 and 2 for all samples j 
    # (not using the median value, thus the resulting estimation will be the centroid of all coordinates estimation with each j)
    # [!] this procedure is discarded since estimations with median value are more accurate
    autocalibrated_coords_j = np.empty((n_total_anchors, 3, n_samples), dtype = float)
    for j in range(n_samples):
        autocalibration_solver.autocalibrated_coords = np.copy(initial_guess)
        autocalibration_solver.stageOne(sample_idx = None)
        autocalibration_solver.stageTwo(sample_idx = None)
        autocalibrated_coords_j[:,:,j] = np.copy(autocalibration_solver.autocalibrated_coords)
        # print progress
        percentage_completed = "{:.2f}".format((j + 1)/n_samples * 100)
        sys.stdout.write(f'\r {percentage_completed} % Complete')

        # print estimation for all samples
        my_colors[:,3] = intensities[j]
        ax.scatter(autocalibration_solver.autocalibrated_coords[:,0], autocalibration_solver.autocalibrated_coords[:,1], autocalibration_solver.autocalibrated_coords[:,2], color = my_colors, marker = 'x')
    print('\n')
    """

    # results df
    main_df = pd.DataFrame({
        'anchor_id' : [],
        'error [m]' : [],
        'x_error [m]' : [],
        'y_error [m]' : [],
        'z_error [m]' :  []
    })

    # Used to save the values calculated at <nodes_cfg_label>.yaml
    file1 = open(dwm1001_drivers_path + "/params/nodes_cfg/" + nodes_cfg_label + ".yaml", 'r')
    Lines = file1.readlines()
    f_gt = open(dwm1001_drivers_path + "/params/nodes_cfg/" + nodes_cfg_label + "_temp.yaml", "w")
    for i in range(12):
        f_gt.write(Lines[i])
    f_gt.write('  anchor0_coordinates: ' + str(autocalibrated_coords[0]) + ' #DW009A\n')
    f_gt.write('  anchor1_coordinates: ' + str(autocalibrated_coords[1]) + ' #DW2D9C\n')
    f_gt.write('  anchor2_coordinates: ' + str(autocalibrated_coords[2]) + ' #DW4848\n')
    f_gt.write('  anchor3_coordinates: ' + str(autocalibrated_coords[3]) + ' #DW0038\n')
    f_gt.write('  anchor4_coordinates: ' + str(autocalibrated_coords[4]) + ' #DW47FC\n')
    for i in range(17,32):
        f_gt.write(Lines[i])
    f_gt.write('  anchor0_coordinates: ' + str(autocalibrated_coords[5]) + ' #DW4984\n')
    f_gt.write('  anchor1_coordinates: ' + str(autocalibrated_coords[6]) + ' #DW4806\n')
    f_gt.write('  anchor2_coordinates: ' + str(autocalibrated_coords[7]) + ' #DW4814\n')
    f_gt.write('  anchor3_coordinates: ' + str(autocalibrated_coords[8]) + ' #DW43EB\n')
    f_gt.write('  anchor4_coordinates: ' + str(autocalibrated_coords[9]) + ' #DW1632\n')
    for i in range(37,43):
        f_gt.write(Lines[i])

    for i in range(n_total_anchors):
        # solving stages 1 and 2 for all samples j therefore there will be j coord estimations for each anchor
        #estimated_anchor_coords = autocalibrated_coords_j[i,:,:].T
        #centroid = np.mean(estimated_anchor_coords, axis = 0)
        
        centroid = autocalibrated_coords[i]        
        ax.scatter(centroid[0], centroid[1], centroid[2], color = my_colors[i], marker = 'x')

        ax.scatter(initial_guess[i,0], initial_guess[i,1], initial_guess[i,2], color = my_colors_light[i], marker = '*')
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
    #main_df.to_csv('autocalibration_campus_sport_new.csv')
    #np.savetxt('landmarks_est.txt', autocalibrated_coords)
    plt.legend(loc='best')
    plt.axis('auto')
    plt.show()   

if __name__ == '__main__':
    main()