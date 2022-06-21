# -*- coding: utf-8 -*-
import argparse, sys, logging
import json
import traceback

from .generation import ACGenerator, TADGenerator
from .scoring import score_ac, score_tad
from .validation import validate_ac, validate_tad, validate_gt, validate_ac_via_index
from .filters import append_missing_video_id
from .datatypes import Dataset
from .io import *
from .plot import plot_tad, plot_ac, plot_all_activity_pr

#
# Public Scorer Version
# 

def process_subset_args(args, ds):
    """ 
    Method to check filter args and apply them to DS so they can be used by
    multiple commands.    
    """
    if args.activity_list_file:        
        raw_al = load_list_file(args.activity_list_file)
        activity_list = list(filter(None, raw_al))
        log.info("Using {} activity-id from '{}' activities-file.".format(len(activity_list), args.activity_list_file))
        #log.debug(activity_list)
        ds.ref = ds.ref.loc[ds.ref.activity_id.isin(activity_list)]
        ds.hyp = ds.hyp.loc[ds.hyp.activity_id.isin(activity_list)]        
    if args.video_list_file:
        raw_vl = load_list_file(args.video_list_file)
        video_list = list(filter(None, raw_vl))
        log.info("Using {} video-id from '{}' video-id-file.".format(len(video_list), args.video_list_file))
        log.debug(video_list)
        ds.ref = ds.ref.loc[ds.ref.video_file_id.isin(video_list)]
        ds.hyp = ds.hyp.loc[ds.hyp.video_file_id.isin(video_list)]
    log.debug(ds)

def ac_scorer_cmd(args):
    """
    AC Scoring:
    - validate
    - score
    - extract + print
    """    
    
    ds = Dataset(load_ref(args.reference_file), load_hyp(args.hypothesis_file))
    log.debug("Loaded REF/HYP:")
    log.debug(ds)
    ensure_output_dir(args.output_dir)    
    process_subset_args(args, ds)
    # Prevent auto-filtering of faulty data    
    if not args.skip_validation:
        validate_ac(ds)
    log.debug("Validated:")     
    log.debug(ds)    
    
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
    """
    TAD Scoring (output still needs work to differentiate iou's):
    - validate
    - score
    - extract + print
    """
    ensure_output_dir(args.output_dir)
    ds = Dataset(load_tad_ref(args.reference_file), load_tad_hyp(args.hypothesis_file))        
    log.debug(ds)
    process_subset_args(args, ds)
    # Prevent auto-filtering of faulty data    
    if not args.skip_validation:
        validate_tad(ds) 
    argstr = json.dumps(args, default=lambda o: o.__dict__, sort_keys=True)
    thresholds = [float(i) for i in args.iou_thresholds.split(',')]
    score_tad(ds, args.metrics, thresholds , args.output_dir, argstr)
    h5f = h5_open_archive(os.path.join(args.output_dir, 'scoring_results.h5'))
    data = h5_extract_system_iou_scores(h5f)
    aData = h5_extract_activity_iou_scores(h5f)
    if args.print_alignment:
        alignData = h5_extract_alignment(os.path.join(args.output_dir, 'scoring_results.h5'))
        write_alignment_file(os.path.join(args.output_dir, 'alignments.csv'), alignData)
        print("Alignments")
        print("----------")
        print(open(os.path.join(args.output_dir, 'alignments.csv')).read())

    write_system_level_scores(os.path.join(args.output_dir, 'system_scores.csv'), data)
    write_activity_level_scores(os.path.join(args.output_dir, 'activity_scores.csv'), aData)
    print("Activity Scores")
    print("---------------")
    print(open(os.path.join(args.output_dir, 'activity_scores.csv')).read())
    print("System Score")
    print("-------------")
    print(open(os.path.join(args.output_dir, 'system_scores.csv')).read())
       
def ac_hyp_validator_cmd(args):    
    ds = Dataset(load_index(args.video_index_file), load_hyp(args.hypothesis_file))    
    validate_ac_via_index(ds) 

def plot_cmd(args):   
    h5f = h5_open_archive(args.score_file)    
    ensure_output_dir(args.output_dir)
    if h5f.attrs['scorer-mode'] == 'AC':   
        plot_ac(h5f, args.output_dir)
        if args.generate_activity_plots:
            odir = os.path.join(args.output_dir, "activities")
            ensure_output_dir(odir)
            plot_all_activity_pr(h5f, odir)
    else:
        plot_tad(h5f, args.output_dir, prefix=args.prefix)

# -----------------------------------------------------------------------------
def main(args=None):
    """
    All tool cli is defined here.

    Note: Using argparse to enforce input parameters. No checks are performed in
    lower level methods.
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
    parser_score_ac.add_argument("-m", "--metrics", nargs='?', default="map", help="Available metrics: map, map_11, map_101, map_avg, map_auc")
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
    parser_score_tad.add_argument("-i", "--iou_thresholds", nargs='?', default="0.2", help="A comma separated list of IoU thresholds.")
    parser_score_tad.add_argument("-p", "--skip_validation", action="store_true", help="Skip validation step (default: off)")
    parser_score_tad.add_argument("-n", "--print_alignment", action="store_true", default=False, help="Additionally extract and output alignment (default: off)")
    parser_score_tad.set_defaults(func = tad_scorer_cmd)

    parser_validate_ac_hyp = subparsers.add_parser('validate-ac-hyp', help='Validate system output against video index.')
    parser_validate_ac_hyp.add_argument("-r", '--video_index_file', type=str, required=True)
    parser_validate_ac_hyp.add_argument("-y", '--hypothesis_file', type=str, required=True)
    parser_validate_ac_hyp.set_defaults(func = ac_hyp_validator_cmd)

    parser_plot = subparsers.add_parser('plot-results', help='Extract system and activity results and generate plots.')
    parser_plot.add_argument("-a", "--generate_activity_plots", action="store_true", default=False, help="Extract and output P/R plots for each activity individually (default: off)")
    parser_plot.add_argument("-f", '--score_file', type=str, required=True)    
    parser_plot.add_argument("-o", "--output_dir", nargs='?', type=str, default="tmp")
    parser_plot.add_argument("-p", "--prefix", help='Prefix to append to legend on TAD plot', type=str, default=None)
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
