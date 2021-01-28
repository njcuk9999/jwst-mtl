#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Web Jan 27 10:31 2021

@author: MCR

File containing all diagnostic plotting functions for the simple solver.
"""

import matplotlib.pyplot as plt
import numpy as np
import corner


def _plot_corner(sampler):
    '''Utility function to produce the corner plot for results of _do_emcee.
    '''
    labels = [r"ang", "xshift", "yshift"]
    flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)
    fig = corner.corner(flat_samples, labels=labels)
    plt.show()


def _plot_transformation_steps(data_shift, data_rot, data_shiftback,
                               data_offset):
    '''Plot the outcomes of the various steps involved in the simple_solver
    transformation:
        1. Initial shift of rotation center to the center of the frame.
        2. Rotation around the rotation center.
        3. Shift of the data back to its initial position.
        4. Application of the offset.
    '''

    plt.figure(figsize=(15, 3))
    plt.imshow(data_shift, origin='lower')
    plt.title('Shift')
    plt.show()

    plt.figure(figsize=(15, 3))
    plt.imshow(data_rot, origin='lower')
    plt.title('Rotate')
    plt.show()

    plt.figure(figsize=(15, 3))
    plt.imshow(data_shiftback, origin='lower')
    plt.title('Reshift')
    plt.show()

    plt.figure(figsize=(15, 3))
    plt.imshow(data_offset, origin='lower')
    plt.title('Offset')
    plt.show()
