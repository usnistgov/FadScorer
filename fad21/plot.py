import logging
import h5py
import matplotlib.pyplot as plt
import numpy as np
from .io import *

log = logging.getLogger(__name__)

def plot_prs(h5f, root_path, title, output_fn):
    plt.style.use('_mpl-gallery')
    precDS = h5f["{}/prs/precision".format(root_path)]
    recallDS = h5f["{}/prs/recall".format(root_path)]
    stderrDS = h5f["{}/prs/stderror".format(root_path)]
    x = recallDS[()]
    y = precDS[()]
    y_err = stderrDS[()]    
    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    ax.plot(x, y, linewidth=1.0)
    ax.fill_between(x, y - y_err, y + y_err, alpha=0.2)
    ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.1),
        ylim=(0, max(1.0, max(y+y_err))), yticks=np.arange(0, 1, 0.1))
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)        
    plt.savefig(output_fn)

def gen_sub_prs_plot(h5f, root_path, iou, fig, ax, prefix=None):
    log.debug(root_path)
    precDS = h5f["{}/prs/precision".format(root_path)]
    recallDS = h5f["{}/prs/recall".format(root_path)]
    stderrDS = h5f["{}/prs/stderror".format(root_path)]
    x = recallDS[()]
    y = precDS[()]
    y_err = stderrDS[()]        
    label = ('%s (%s)' % (prefix, iou)) if prefix is not None else str(iou)
    ax.plot(x, y, linewidth=1.0, label=label)
    ax.fill_between(x, y - y_err, y + y_err, alpha=0.05)    
    ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.1),
           ylim=(0, 1), yticks=np.arange(0, 1, 0.1))

def plot_all_activity_pr(h5f, output_dir):
    for activity in h5f['activity']:
        ofn = os.path.join(output_dir, "{}.png".format(activity))        
        plot_pr(h5f, "activity/{}".format(activity), ofn)

def plot_pr(h5f, root_path, output_fn):
    """ Plot Precision / Recall Curves """
    plt.style.use('_mpl-gallery')
    precDS = h5f["{}/prt/precision".format(root_path)]
    recallDS = h5f["{}/prt/recall".format(root_path)]    
    x = recallDS[()]
    y = precDS[()]
    fig, ax = plt.subplots(figsize=(12,10), constrained_layout=True)    
    ax.plot(x, y, linewidth=1.0)
    
    ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.1),
           ylim=(0, 1), yticks=np.arange(0, 1, 0.1))    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(root_path)
    plt.savefig(output_fn)    

def plot_ac(h5f, output_dir):    
    title = "Mean Precision Recall"
    plot_prs(h5f, '/system', title, os.path.join(output_dir, "ac_prs.png"))

# Plot each IoU PR Plot ndividually
def plot_tad_single(h5f, output_dir):
    iouG = h5f['system']
    for iou in iouG.keys():
        path = "/system/iou/{}".format(iou)        
        title = "Mean Precision Recall @ {}".format(iou)
        plot_prs(h5f, path, title, os.path.join(output_dir, "tad_prs_{}.png".format(iou)))

# Plot all IoU graphs into one w/ legend
def plot_tad(h5f, output_dir, legend_loc='upper right', prefix=None):
    iouG = h5f['/system/iou']
    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    plt.style.use('_mpl-gallery')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title("Mean Precision Recall @ IoU")
    for iou in iouG.keys():
        path = "/system/iou/{}".format(iou)    
        gen_sub_prs_plot(h5f, path, iou, fig, ax, prefix=prefix)
    #ax.set(xlim=(0, 1.0), xticks=np.arange(0, 1, 0.1), 
    #    ylim=(0, 1.0), yticks=np.arange(0, 1, 0.1))
    ax.grid(True)
    plt.legend(loc=legend_loc)        
    plt.savefig(os.path.join(output_dir, "tad_prs.png"))
    plt.savefig(os.path.join(output_dir, "tad_prs.pdf"))

def compute_timeseries_steps(v):
    cl = np.concatenate(list(zip(v['frame_start'], v['frame_end'])))
    ocl = []
    for idx in range(0,len(cl)):
        ocl.append(0 if (idx % 2) == 0 else 1)
    cl1 = np.concatenate([[0], cl, [cl[-1]]])
    ocl1 = np.concatenate([[0], ocl, [0]])
    return [cl1, ocl1]

#def plot_activities(h5f, output_dir):
    # scoresG = h5f['system/scores']
    # for metric in scoresG.keys():
    #     print("{}: {}".format(metric, scoresG[metric][()]))
    # #header("Activity Scores")
    # #activitiesG = h5f['activities']
    # #for activity in activitiesG.keys():
    # #    scoresG = activitiesG["{}/scores".format(activity)]
    # #    for metric in scoresG.keys():
    # #        print("{}: {}".format(metric, scoresG[metric][()]))
