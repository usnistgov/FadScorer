import logging
from pickle import FALSE
import h5py
import matplotlib.pyplot as plt
import numpy as np
from .io import *
from .aggregators import ap_interp_pr, fix_pr_tail
log = logging.getLogger(__name__)

def plot_prs(h5f, root_path, title, output_fn, interp=False):
    """ Plot aggregated P/R curve

    Parameters
    ----------
    h5f: HDF5 File handle
        Opened H5 file's handle
    root_path: str
        h5 Path to plot.
        /system/
        /system/iou/0.2/
    """
    plt.style.use('_mpl-gallery')
    if interp:
        precDS = h5f["{}/prs_interp/precision".format(root_path)]
        recallDS = h5f["{}/prs_interp/recall".format(root_path)]
        stderrDS = h5f["{}/prs_interp/stderror".format(root_path)]
    else:
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
    log.info("Saving plot {}".format(output_fn))        
    plt.savefig(output_fn)
    plt.close()

def plot_all_ac_activity_pr(h5f, output_dir):
    """ Generate PR plots, one per activity and store them in the
    [output_dir]/activities folder.

    Parameters
    ----------
    h5f: HDF5 File Handle
        Opened H5 file
    output_dir: str
        Output dir to create plots.
    """
    for activity in h5f['activity']:
        ofn = os.path.join(output_dir, "{}.png".format(activity))        
        plot_pr(h5f, "activity/{}".format(activity), ofn)

def plot_pr(h5f, root_path, output_fn):
    """ Plot Precision/Recall Curve for a specific path
    
    Parameters
    ----------
    h5f: H5FS File handle
        Opened H5 file's handle
    root_path: str
        h5 Path to plot.
        /system/iou/0.2/prs/precision Dataset {100}
    """ 
    plt.style.use('_mpl-gallery')    
    precDS = h5f["{}/prt/precision".format(root_path)]
    recallDS = h5f["{}/prt/recall".format(root_path)] 
    x = recallDS[()]
    y = precDS[()]    
    prec, recl, _ = ap_interp_pr(y[::-1], x[::-1])
    prec, recl = fix_pr_tail(prec, recl)
    fig, ax = plt.subplots(figsize=(12,10), constrained_layout=False)    
    plt.tight_layout(pad=1.2, h_pad=1.1, w_pad=1.1)
    
    ax.plot(x, y, linewidth=1.0, label="p")    
    ax.plot(recl, prec, linewidth=0.5, label="p_interp")    
    ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.1),
           ylim=(0, 1), yticks=np.arange(0, 1, 0.1))    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    #plt.title("precision vs. recall curve")
    ax.legend(loc='right', bbox_to_anchor=(1.0, 0.9))
    ax.set_title(root_path)
    log.info("Saving plot {}".format(output_fn))    
    
    plt.savefig(output_fn)
    plt.close()

def plot_sub_pr(h5f, root_path, iou, fig, ax, prefix=None):
    """ All P/R Plot to graph handle for data a specific path in H5F file.    
    
    Parameters
    ----------
    h5f: H5FS File handle
        Opened H5 file's handle
    root_path: str
        h5 Path to plot. /activity/Test (prs/precision) /activity/Test/iou/0.2
        (/prs/precision)
    iou: str
        tIoU Value the plot refers to.
    fig: matplotlib.Figure
        NOT IN USE
    ax: matplotlib.Axis
        Plot handle
    prefix: str
        NOT IN USE
    """    
    precDS = h5f["{}/prt/precision".format(root_path)]
    recallDS = h5f["{}/prt/recall".format(root_path)]    
    x = recallDS[()]
    y = precDS[()]    
    prec, recl, _ = ap_interp_pr(y[::-1], x[::-1])
    prec, recl = fix_pr_tail(prec, recl)    
    ax.plot(x, y, linewidth=1.0, label="@Temporal IoU={}".format(iou))
    ax.plot(recl, prec, linewidth=0.5, label="@Temporal IoU={} (interp)".format(iou))    
    ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.1),
           ylim=(0, 1), yticks=np.arange(0, 1, 0.1))    

def gen_sub_prs_plot(h5f, root_path, iou, fig, ax, label):
    """ All aggregated P/R Plot to graph handle for data a specific path in H5F file.            
    
    Parameters
    ----------
    h5f: H5FS File handle
        Opened H5 file's handle
    root_path: str
        h5 Path to plot.
        /system/ (prs/precision)
        /system/iou/0.2 (/prs/precision)
    iou: str
        NOT IN USE
    fig: matplotlib.Figure
        NOT IN USE
    ax: matplotlib.Axis
        Plot handle
    label: str
        Label for plot (f.e. includes iou level if applicable)
    """      
    log.debug(root_path)
    largs = {} # {"drawstyle": "steps-post"}
    precDS = h5f["{}/prs/precision".format(root_path)]
    recallDS = h5f["{}/prs/recall".format(root_path)]
    stderrDS = h5f["{}/prs/stderror".format(root_path)]
    x = recallDS[()]
    y = precDS[()]
    y_err = stderrDS[()]
    ax.plot(x, y, linewidth=1.0, label=label, **largs)
    ax.fill_between(x, y - y_err, y + y_err, alpha=0.05)    
    ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.1),
           ylim=(0, 1), yticks=np.arange(0, 1, 0.1))

def plot_all_tad_activity_pr(h5f, output_dir):
    """ Generate all (TAD) PR plots, one per activity for all IoU and store them
    in the [output_dir]/activities folder.

    Parameters
    ----------
    h5f: HDF5 File Handle
        Opened H5 file
    output_dir: str
        Output dir to create plots.
    """    
    for activity in h5f['activity']:
        ofn = os.path.join(output_dir, "{}.png".format(activity))
        iouG = h5f['/activity/{}/iou'.format(activity)]
        fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
        plt.style.use('_mpl-gallery')
        for iou in iouG.keys():    
            plot_sub_pr(h5f, "activity/{}/iou/{}".format(activity,iou), iou, fig, ax)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        plt.title("Precision/Recall {}".format(activity))
        ax.legend(loc='right', bbox_to_anchor=(1.0, 0.9))        
        ofn = os.path.join(output_dir, "{}.png".format(activity))        
        log.info("Saving plot {}".format(ofn))    
        plt.savefig(ofn)
        plt.close()        

# Plot each IoU PR Plot ndividually
def plot_tad_single_system(h5f, output_dir):
    """ Plot separate aggregated system PR plots per IoU.
    
    Parameters
    ----------
    h5f: HDF5 File Handle
        Opened H5 file
    output_dir: str
        Output dir to create plots.    
    """
    iouG = h5f['system']
    for iou in iouG.keys():
        path = "/system/iou/{}".format(iou)        
        title = "Mean Precision/Recall @ {}".format(iou)
        plot_prs(h5f, path, title, os.path.join(output_dir, "tad_prs_{}.png".format(iou)))

# 
def plot_tad_system(h5f, output_dir, legend_loc='upper right', prefix=None):
    """ Plot all aggregated PR IoU graphs into one w/ legend
    
    Parameters
    ----------
    h5f: HDF5 File Handle
        Opened H5 file
    output_dir: str
        Output dir to create plots.
    legend_loc: str
        Plot parameter
    prefix: str
        Plot Label.
    """        
    iouG = h5f['/system/iou']
    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    prefix = 'Temporal IoU = {}' if prefix is None else prefix
    plt.style.use('_mpl-gallery')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title("Mean Precision/Recall @ Temporal IoU")
    for iou in iouG.keys():
        path = "/system/iou/{}".format(iou)
        gen_sub_prs_plot(h5f, path, iou, fig, ax, prefix.format(iou))
    #ax.set(xlim=(0, 1.0), xticks=np.arange(0, 1, 0.1), 
    #    ylim=(0, 1.0), yticks=np.arange(0, 1, 0.1))
    ax.grid(True)
    plt.legend(loc=legend_loc)        
    plt.savefig(os.path.join(output_dir, "tad_prs.png"))
    plt.savefig(os.path.join(output_dir, "tad_prs.pdf"))  # for non-bitmap fonts in publications