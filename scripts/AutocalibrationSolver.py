""" 
@file: AutocalibrationSolver.py
@description:   python class to autocalibrate anchor position based on inter-anchor ranging data
                This process takes an initial anchors coords guess as starting point of the iterative
                optimization.
@author: Esau Ortiz
@date: october 2021
"""

import numpy as np
import sys, yaml
from pathlib import Path
from scipy.optimize import fmin

def readYaml(file):
    with open(file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

class AutocalibrationSolver(object):
    def __init__(self, autocalibration_samples, initial_guess, fixed_anchors, max_iters = 1500, convergence_thresh = 0.01, LSq_min_anchors = 4, lower_percentile = 0.25, upper_percentile = 0.75, verbose = False):
        """ AutocalibrationSolver is a multi-stage procedure to autocalibrate
            anchor coordinates based on inter-anchor ranges
        Parameters
        ----------
        autocalibration_samples: (N, M, N) array
            inter-anchor ranges (e.g. autocalibration_samples(0,:,1) contains M ranges between anchor_0 and anchor_1)
        initial_guess: (N, 3) array
            initial guess of anchor coordinates
        autocalibrated_coords: (N, 3) array
            current autocalibrated anchor coordinates
        fixed_anchors: (N, ) bool array
            bool mask of anchors whose position is assumed to be fixed
        max_iters: int
            Stage 1 maximum number of iterations
        convergence_thresh: float
            maximum difference between inter-anchor distances between stage 1 iterations 't' and 't-1'
            to consider that stage 1 has converged
        LSq_min_anchors: int
            minimum number of anchors to perform a LSq anchor coordinates optimization (coordinatesOpt method)
        lower_percentile: float
            inter-anchor range percentile to filter M samples. If a given sample 'm' `range_m` between anchors `anchor_0` and `anchor_1` satisfies range_m >= np.percentile(autocalibration_samples[0,1,:], lower_percentile) is considered in Stage 2 (costOpt method)
        upper_percentile: float
            same as lower percentile but this time is upper limit
        """
        self.samples_ijk = autocalibration_samples
        self.initial_guess = initial_guess
        self.autocalibrated_coords = initial_guess
        self.fixed_anchors = fixed_anchors
        self.max_iters = max_iters
        self.convergence_thresh = convergence_thresh
        self.LSq_min_anchors = LSq_min_anchors
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.verbose = verbose

    def stageOne(self, sample_idx = None):
        n_anchors, _, _ = self.samples_ijk.shape
        # starting point is initial guess
        self.autocalibrated_coords = np.copy(self.initial_guess)
        # if sample_idx is provided stageOne is performed for that index and for the median otherwise
        if sample_idx is None: 
            # compute median (n_anchors, n_anchors) array discarding bad ranges (i.e. range = -1.0)
            samples_ik = np.empty((n_anchors, n_anchors), dtype = float)
            for i in range(n_anchors):
                for k in range(n_anchors):
                    samples = self.samples_ijk[i,:,k][self.samples_ijk[i,:,k] > 0]
                    if samples.shape[0] > 0: samples_ik[i,k] = np.median(samples)
                    else: samples_ik[i,k] = -1.0
        else: 
            samples_ik = np.copy(self.samples_ijk[:,sample_idx,:])

        for _ in range(self.max_iters):
            # save previous anchors coords for termination condition
            anchors_coords_old = np.copy(self.autocalibrated_coords)

            # update anchors coords
            for i in range(n_anchors):
                # don't update fixed anchor
                if self.fixed_anchors[i] == True: continue
                _anchors_coords = []
                _ranges = []
                for k in range(n_anchors):
                    # skip if range has not been received (i.e. range  == -1)
                    if samples_ik[i,k] < 0.0: continue
                    _anchors_coords.append(self.autocalibrated_coords[k])
                    _ranges.append(samples_ik[i,k])

                if len(_anchors_coords) >= self.LSq_min_anchors:
                    _anchors_coords = np.array(_anchors_coords)
                    _ranges = np.array(_ranges)
                    self.autocalibrated_coords[i] = AutocalibrationSolver.coordinatesOpt(_anchors_coords, _ranges)

            # termination criterion -> distances between anchors have not been modified significantly
            """
            inter_anchor_distances = np.sqrt(np.einsum("ijk->ij", (self.autocalibrated_coords[:, None, :] - self.autocalibrated_coords) ** 2))
            inter_anchor_distances_old = np.sqrt(np.einsum("ijk->ij", (anchors_coords_old[:, None, :] - anchors_coords_old) ** 2))
            if np.abs(np.linalg.norm(inter_anchor_distances - inter_anchor_distances_old)) < self.convergence_thresh:
            """
            # termination criterion -> autocalibrated coords have not been modified significantly
            if np.abs(np.linalg.norm(self.autocalibrated_coords - anchors_coords_old)) < self.convergence_thresh:
                break

    def stageTwo(self, sample_idx = None):
        n_anchors, _, _ = self.samples_ijk.shape
        # if sample_idx is provided stageOne is performed for that index and for the median otherwise
        if sample_idx is None: 
            # compute median (n_anchors, n_anchors) array discarding bad ranges (i.e. range = -1.0)
            _samples_ik = np.empty((n_anchors, n_anchors), dtype = float)
            for i in range(n_anchors):
                for k in range(n_anchors):
                    samples = self.samples_ijk[i,:,k][self.samples_ijk[i,:,k] > 0]
                    if samples.shape[0] > 0: _samples_ik[i,k] = np.median(samples)
                    else: _samples_ik[i,k] = -1.0
        else:  
            samples_ik = np.copy(self.samples_ijk[:,sample_idx,:])        
            # filter samples that are considered bad (outside percentiles limits)
            lower_bounds = np.percentile(self.samples_ijk, self.lower_percentile, axis = 1)
            upper_bounds = np.percentile(self.samples_ijk, self.upper_percentile, axis = 1)
            mask1 = samples_ik <= upper_bounds
            mask2 = samples_ik >= lower_bounds
            mask = mask1 & mask2
            _samples_ik = -np.ones(samples_ik.shape)
            _samples_ik[mask] = samples_ik[mask]

        # optimization based on scipy.optimize.fmin
        return AutocalibrationSolver.costOpt(self.autocalibrated_coords, _samples_ik, self.fixed_anchors, self.verbose)

    def estimationError(self, gt, axis = None):
        """ Return estimation error computed as euclidean
        distance
        Parameters
        ----------
        gt: (N, 3) array
            anchor coordinates ground truth
        axis (optional): int
            if axis is provided error in given axis
            is returned
        Returns
        -------
        error: (N, ) array
            euclidean distance between ground truth and 
            estimation
        """
        if axis is not None:
            est_error = self.autocalibrated_coords - gt
            return est_error[:, axis]
        else:
            return np.sqrt(np.einsum("ijk->ij", (self.autocalibrated_coords[:, None, :] - gt) ** 2))

    @staticmethod
    def coordinatesOpt(anchors_coords, ranges):
            """ Least squares optimization 
            Parameters
            ----------
            anchors_coords: (N, 3) array
                anchor coordinates
            ranges: (N, ) array
                anchor-tag range
            """
            # build A matrix
            A = 2 * np.copy(anchors_coords)
            for i in range(A.shape[0] - 1): A[i] = A[-1] - A[i]
            A = A[:-1] # remove last row

            # build B matrix
            B = np.copy(ranges)**2
            B = B[:-1] - B[-1] - np.sum(anchors_coords**2, axis = 1)[:-1] + np.sum(anchors_coords[-1]**2, axis = 0)
            return np.dot(np.linalg.pinv(A), B)    

    @staticmethod
    def costOpt(anchors_coords, ranges_ik, fixed_anchors, verbose = False):
        """ Cost optimization based on scipy.optimize.fmin 
        Parameters
        ----------
        anchors_coords: (N, 3) array
            anchor coordinates
        ranges_ik: array (N, N) 
            inter anchor ranges
        fixed_anchors: (N, ) array
            bool mask of fixed anchors
        Returns
        -------
        Theta_opt: (N, 3) array
            optimized anchor coordinates
        """
        def _my_opt_func(Theta, *args):
            """ Optimize target function
            Parameters
            ----------
            Theta: (N, 3)
                current array of anchors coordinates (x,y,z)
            args:
                ranges_ik: array (N, N) 
                    inter anchor ranges
                n_anchors: int
                    total number of anchors
                fixed_anchors: (N, ) array
                    bool mask of fixed anchors
                Theta_init: (N, 3) array
                    initial anchor coordinates
            Returns
            -------
            cost: float
            """
            ranges_ik, n_anchors, fixed_anchors, Theta_init = args
            Theta = Theta.reshape(n_anchors, 3)

            # Modify current Theta to keep anchors fixed
            Theta[fixed_anchors] = Theta_init[fixed_anchors]
            # compute cost
            distances_ij = np.einsum("ijk->ij", (Theta[:, None, :] - Theta) ** 2)
            cost_ij = (distances_ij - ranges_ik ** 2) ** 2
            # remove j = i costs
            cost_ij[distances_ij == 0] = 0
            # remove costs computed with invalid ranges (i.e. ranges < 0)
            cost_ij[ranges_ik < 0] = 0
            return np.sum(np.einsum("ij->i", cost_ij))

        anchors_coords = anchors_coords.T.reshape(-1,3)
        Theta_init = np.copy(anchors_coords)
        n_anchors = anchors_coords.shape[0]
        args = ranges_ik, n_anchors, fixed_anchors, Theta_init
        
        if verbose: print(f'Before optimization: Cost = {_my_opt_func(anchors_coords, *args)}')
        Theta_opt = fmin(_my_opt_func, anchors_coords, args = args, disp=False)    
        if verbose: print(f'After optimization: Cost = {_my_opt_func(Theta_opt, *args)}')
        Theta_opt = Theta_opt.reshape(anchors_coords.shape)
        Theta_opt[fixed_anchors] = Theta_init[fixed_anchors]

        return Theta_opt