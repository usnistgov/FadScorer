# -*- coding: utf-8 -*-
import argparse, sys, logging
import json
import traceback
 
from .scoring import score_ac, score_tad
from .validation import validate_ac, validate_tad, detect_missing_video_id, process_subset_args
from .datatypes import Dataset
from .io import *
from .plot import plot_tad_system, plot_prs, plot_all_tad_activity_pr, plot_all_ac_activity_pr
from .aggregators import compute_aggregate_pr, compute_aggregate_iou_pr

def ac_scorer_cmd(args):
    """ AC Scoring CLI Wrapper: Loads, validates and preprocesses dataset.
    Scores against dataset storing AP scores, and PR curves per activity in H5
    archive. Writes system and activity level scores to CSV and stdout.

    Parameters
    ----------
    args: argparse ns
        Command parameters set by argparse
    """    
    
    ds = Dataset(load_ref(args.reference_file), load_hyp(args.hypothesis_file))    
    ensure_output_dir(args.output_dir)    
    process_subset_args(args, ds)
    # Prevent auto-filtering of faulty data    
    if not args.skip_validation:
        validate_ac(ds)
    
    argstr = json.dumps(args, default=lambda o: o.__dict__, sort_keys=True)
    score_ac(ds, args.metrics, int(args.filter_top_n), args.output_dir, argstr)

    h5f = h5_open_archive(os.path.join(args.output_dir, 'scoring_results.h5'))
    data = h5_extract_system_scores(h5f)          
    aData = h5_extract_activity_scores(h5f)
    write_system_level_scores(os.path.join(args.output_dir, 'system_scores.csv'), data)
    write_activity_level_scores(os.path.join(args.output_dir, 'activity_scores.csv'), aData)
    print("Activity Scores")
    print("---------------")
    print(open(os.path.join(args.output_dir, 'activity_scores.csv')).read())
    print("System Score")
    print("-------------")
    print(open(os.path.join(args.output_dir, 'system_scores.csv')).read())

def tad_scorer_cmd(args):
    """ TAD Scoring CLI Wrapper: Loads, validates and preprocesses dataset.
    Scores against dataset storing AP scores, and PR curves per activity and
    tIoU threshold in H5 archive. Writes system and activity level scores to
    CSV and stdout.

    Parameters
    ----------
    args: argparse ns
        Command parameters set by argparse
    """
    ensure_output_dir(args.output_dir)
    ds = Dataset(load_tad_ref(args.reference_file), load_tad_hyp(args.hypothesis_file))            
    process_subset_args(args, ds)

    # Prevent auto-filtering of faulty data    
    if not args.skip_validation:
        validate_tad(ds) 

    argstr = json.dumps(args, default=lambda o: o.__dict__, sort_keys=True)
    thresholds = [float(i) for i in args.iou_thresholds.split(',')]
    
    #score_tad(ds, args.metrics, thresholds , args.output_dir, argstr)
    score_tad(ds, thresholds, args.metrics, args.output_dir, int(args.nb_jobs), argstr)
    
    h5f = h5_open_archive(os.path.join(args.output_dir, 'scoring_results.h5'))
    data = h5_extract_system_iou_scores(h5f)
    aData = h5_extract_activity_iou_scores(h5f)

    write_system_level_scores(os.path.join(args.output_dir, 'system_scores.csv'), data)
    write_activity_level_scores(os.path.join(args.output_dir, 'activity_scores.csv'), aData)
    print("Activity Scores")
    print("---------------")
    print(open(os.path.join(args.output_dir, 'activity_scores.csv')).read())
    print("System Score")
    print("-------------")
    print(open(os.path.join(args.output_dir, 'system_scores.csv')).read())
       
def ac_hyp_validator_cmd(args):
    """ AC system-output validation CLI Wrapper

    Parameters
    ----------
    args: argparse ns
        Command parameters set by argparse
    """
    ds = Dataset(load_index(args.video_index_file), load_hyp(args.hypothesis_file))    
    detect_missing_video_id(ds) 

def plot_cmd(args):
    """ AC and TAD Plot CLI Wrapper. Determine type of plot automatically via H5
    content (AC or TAD) and plot aggegated or individual P/R curves.

    Parameters
    ----------
    args: argparse ns
        Command parameters set by argparse
    """     
    h5f = h5_open_archive(args.score_file, 'r+')    
    ensure_output_dir(args.output_dir)
    act = load_list_file(args.activities_file) if len(args.activities_file) else []
    if h5f.attrs['scorer-mode'] == 'AC':
        compute_aggregate_pr(h5f, act)
        plot_prs(h5f, '/system', "Mean Precision/Recall", os.path.join(args.output_dir, "ac_prs.png"))        
        if args.generate_activity_plots:
            odir = os.path.join(args.output_dir, "activities")
            ensure_output_dir(odir)
            plot_all_ac_activity_pr(h5f, odir)
    else:                
        compute_aggregate_iou_pr(h5f, act)
        plot_tad_system(h5f, args.output_dir, prefix=args.prefix)
        if args.generate_activity_plots:
            odir = os.path.join(args.output_dir, "activities")
            ensure_output_dir(odir)
            plot_all_tad_activity_pr(h5f, odir)

def main(args=None):
    """ FADScorer Command line interface: Defines all commands and their
    options and executes respective command wrapper.
    """
    parser = argparse.ArgumentParser(description='FADScorer: Fine-grained Activity Detection Scorer', prog="fad-scorer")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose level output (default: off)")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug level output (default: off)")
    subparsers = parser.add_subparsers(help='command help')

    parser_score_ac = subparsers.add_parser('score-ac', help='Score system activity-classification output against ground-truth.')
    parser_score_ac.add_argument("-r", '--reference_file', type=str, required=True)
    parser_score_ac.add_argument("-y", '--hypothesis_file', type=str, required=True)
    parser_score_ac.add_argument("-a", '--activity_list_file', type=str, required=False, help="Use to filter activities from scoring (REF + HYP)")
    parser_score_ac.add_argument("-f", '--video_list_file', type=str, required=False, help="Used to filter files from scoring (REF + HYP)")
    parser_score_ac.add_argument("-o", "--output_dir", nargs='?', type=str, default="tmp")
    # TODO: move/add top-1 and top-5 metrics to metrics array
    parser_score_ac.add_argument("-m", "--metrics", nargs='?', default="map", help="Available metrics: map, map_interp")
    parser_score_ac.add_argument("-t", "--filter_top_n", nargs='?', default="0", help="Use only top-n confidence system results (0=all)")
    parser_score_ac.add_argument("-p", "--skip_validation", action="store_true", help="Skip validation step (default: off)")    
    parser_score_ac.set_defaults(func = ac_scorer_cmd)

    parser_score_tad = subparsers.add_parser('score-tad', help='Score system activity-detection output against ground-truth.')
    parser_score_tad.add_argument("-r", '--reference_file', type=str, required=True)
    parser_score_tad.add_argument("-y", '--hypothesis_file', type=str, required=True)
    parser_score_tad.add_argument("-a", '--activity_list_file', type=str, required=False, help="Use to filter activities from scoring (REF + HYP)")
    parser_score_tad.add_argument("-f", '--video_list_file', type=str, required=False, help="Used to filter files from scoring (REF + HYP)")
    parser_score_tad.add_argument("-o", "--output_dir", nargs='?', type=str, default="tmp")
    parser_score_tad.add_argument("-m", "--metrics", nargs='?', default="map", help="Available metrics: map, map_11, map_101, map_avg, map_auc")
    parser_score_tad.add_argument("-i", "--iou_thresholds", nargs='?', default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95", help="A comma separated list of IoU thresholds.")
    parser_score_tad.add_argument("-p", "--skip_validation", action="store_true", help="Skip validation step (default: off)")
    parser_score_tad.add_argument("-j", "--nb_jobs", nargs='?', default="-1", help="Number of threads to use for MP (-1=one, 1=all)")
    #parser_score_tad.add_argument("-n", "--ignore_noscore_region", action="store_true", default=False, help="Ignore missing activity_id segements from reference. (default: off)")
    parser_score_tad.set_defaults(func = tad_scorer_cmd)

    parser_validate_ac_hyp = subparsers.add_parser('validate-ac-hyp', help='Validate system output against video index.')
    parser_validate_ac_hyp.add_argument("-r", '--video_index_file', type=str, required=True)
    parser_validate_ac_hyp.add_argument("-y", '--hypothesis_file', type=str, required=True)
    parser_validate_ac_hyp.set_defaults(func = ac_hyp_validator_cmd)

    parser_plot = subparsers.add_parser('plot-results', help='Extract system and activity results and generate plots.')
    parser_plot.add_argument("-a", "--generate_activity_plots", action="store_true", default=False, help="Extract and output P/R plots for each activity individually (default: off)")
    parser_plot.add_argument("-f", '--score_file', type=str, required=True)    
    parser_plot.add_argument("-e", '--activities_file', type=str, required=False, default="")    
    parser_plot.add_argument("-o", "--output_dir", nargs='?', type=str, default="tmp")
    parser_plot.add_argument("-p", "--prefix", help='Prefix to prepend to legend on TAD plot', type=str, default=None)
    parser_plot.set_defaults(func = plot_cmd)

    args = parser.parse_args()
    FORMAT = '%(message)s'
     # Note that Logging (singleton) is inherited in submodules !
    if args.debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
    elif args.verbose:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=FORMAT)
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format=FORMAT)

    #log = logging.getLogger(__name__) 
    # Need to extra handle this as the error-message is cryptic to the user
    try:
        func = args.func
    except AttributeError:
        parser.error("Too few arguments.")
    try:
        # Execute sub-commands    
        args.func(args)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        log.error("[{}] {}".format(exc_type.__name__, exc_value))
        if args.debug:            
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)
        else:
            exit(1)

if __name__ == "__main__":
    sys.exit(main())
