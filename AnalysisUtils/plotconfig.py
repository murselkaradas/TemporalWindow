from roifile import ImagejRoi
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import logging

import sys
import os

import logging

# Suppress logging from matplotlib and roifile
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('roifile').setLevel(logging.WARNING)


plt.rc('font', size=10)
plt.rc('axes', labelsize=10, titlesize=10)
plt.rc('legend', fontsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family']=  'sans-serif'
matplotlib.rcParams['font.sans-serif']=  'Arial'
import matplotlib.cbook as cbook
from matplotlib import cm

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"]})

title_font = {'fontname':'Arial', 'size':'10', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space

def save_fig(fig_id, fig=None, tight_layout=True, fig_extension="pdf", resolution=300, Imagespath='ResultImgs'):
    # Redirect stdout and stderr to null to suppress log messages
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    
    try:
        path = Imagespath / f"{fig_id}.{fig_extension}"
        if fig is None:
            if tight_layout:
                plt.tight_layout()
            plt.savefig(path, format=fig_extension, dpi=resolution)
        else:
            if tight_layout:
                fig.tight_layout()
            fig.savefig(path, format=fig_extension, dpi=resolution)
    finally:
        # Reset stdout and stderr after saving the figure
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__



