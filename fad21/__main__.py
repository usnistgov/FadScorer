# -*- coding: utf-8 -*-
import argparse, sys, logging
import json
import traceback

from .generation import ACGenerator, TADGenerator
from .scoring import score_ac, score_tad
from .validation import validate_ac, validate_tad, validate_gt
from .datatypes import Dataset
from .io import *
from .plot import plot_tad, plot_ac

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

def ac_generator_cmd(args):
    generator = ACGenerator(args.reference_file)
    content = generator.generate(args.method, float(args.match_factor))
    write_output(args.output_file, content.to_csv(index=False, header=False))

def tad_generator_cmd(args):    
    ds = Dataset(load_tad_ref(args.reference_file))
    validate_gt(ds)
    generator = TADGenerator(ds)
    content = generator.generate(args.method, float(args.match_factor))
    write_output(args.output_file, pd.DataFrame(content, columns = [
        "video_file_id", 
        "activity_id", 
        "confidence_score", 
        "frame_start", 
        "frame_end" ]).to_csv(index=False, header=False))    

def ac_scorer_cmd(args):    
    ensure_output_dir(args.output_dir)    
    ds = Dataset(load_ref(args.reference_file), load_hyp(args.hypothesis_file))
    log.debug(ds)
    process_subset_args(args, ds)
    # Prevent auto-filtering of faulty data    
    if not args.skip_validation:
        validate_ac(ds)     
    log.debug(ds)
    argstr = json.dumps(args, default=lambda o: o.__dict__, sort_keys=True)
    score_ac(ds, args.metrics, int(args.topk), args.output_dir, argstr)    

def tad_scorer_cmd(args):        
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

def remap_cmd(args):    
    """
    Map v-id|a-id labelset of a file using a maping-file.
    - Works with hyp and ref files.
    """
    mdict = load_mapping_file(args.mapping_file)
    log.info("  {} mapping relations found".format(len(mdict.keys())))
    hyp_df = load_hyp(args.score_file)    
    nhyp_df = hyp_df.replace({'activity_id': mdict})
    fh = open(args.output_file, 'w+')
    if args.reference:        
        columns = ["video_file_id", "activity_id"]        
    else:
        columns = ["video_file_id", "activity_id", "confidence_score"]        
    write_header_output(fh, nhyp_df, columns)

def extractor_cmd(args):   
    h5f = h5_open_archive(args.score_file)
    if h5f.attrs['scorer-mode'] == "AC":
        log.debug("[extract] AC Task detected")
        data = h5_extract_system_scores(h5f)          
        aData = h5_extract_activity_scores(h5f)
    else:
        log.debug("[extract] TAD Tas/tmp/pytest-of-count0/pytest-28/test_scoring_smoothcurve0/system_scores.csvk detected")
        data = h5_extract_system_iou_scores(h5f)
        aData = h5_extract_activity_iou_scores(h5f)
        if args.alignments:
            alignData = h5_extract_alignment(args.score_file)
            write_alignment_file(os.path.join(args.output_dir, 'alignments.csv'), alignData)

    ensure_output_dir(args.output_dir)    
    write_system_level_scores(os.path.join(args.output_dir, 'system_scores.csv'), data)
    write_activity_level_scores(os.path.join(args.output_dir, 'activity_scores.csv'), aData)

def ac_validator_cmd(args):
    ds = Dataset(load_ref(args.reference_file), load_hyp(args.hypothesis_file))    
    validate_ac(ds)    

def ac_gt_validator_cmd(args):
    ds = Dataset(load_ref(args.reference_file))    
    validate_gt(ds)    

def tad_validator_cmd(args):    
    ds = Dataset(load_tad_ref(args.reference_file), load_tad_hyp(args.hypothesis_file))        
    validate_tad(ds)

def tad_gt_validator_cmd(args):    
    ds = Dataset(load_tad_ref(args.reference_file))
    validate_gt(ds)

def ref_validator_cmd(args):    
    ds = Dataset(load_tad_ref(args.reference_file))
    validate_gt(ds)

def plot_cmd(args):   
    h5f = h5_open_archive(args.score_file)    
    ensure_output_dir(args.output_dir)
    if h5f.attrs['scorer-mode'] == 'AC':    
        plot_ac(h5f, args.output_dir)
    else:
        plot_tad(h5f, args.output_dir)

# -----------------------------------------------------------------------------
def main(args=None):
    """
    All tool cli is defined here.

    Note: Using argparse to enforce input parameters. No checks are performed in
    lower level methods.
    """
    parser = argparse.ArgumentParser(description='FAD21: Fine-grained Activity Detection Scorer, V.2022-01-21', prog="fad21")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose level output (default: off)")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug level output (default: off)")
    subparsers = parser.add_subparsers(help='command help')

    parser_gen_ac = subparsers.add_parser('generate-ac', help='Generate system output from ground-truth.')  
    parser_gen_ac.add_argument("-r", '--reference_file', type=str, required=True)
    parser_gen_ac.add_argument("-m", '--method', type=str, required=False, default="match", help="Available methods: random, match")        
    parser_gen_ac.add_argument("-f", '--match_factor', type=str, required=False, default=0.5)      
    parser_gen_ac.add_argument("-o", "--output_file", nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    parser_gen_ac.set_defaults(func = ac_generator_cmd)    

    parser_gen_tad = subparsers.add_parser('generate-tad', help='Generate system output from ground-truth.')  
    parser_gen_tad.add_argument("-r", '--reference_file', type=str, required=True)
    parser_gen_tad.add_argument("-m", '--method', type=str, required=False, default="match", help="Available methods: random, match")        
    parser_gen_tad.add_argument("-f", '--match_factor', type=str, required=False, default=0.5)      
    parser_gen_tad.add_argument("-o", "--output_file", nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    parser_gen_tad.set_defaults(func = tad_generator_cmd)    

    parser_score_ac = subparsers.add_parser('score-ac', help='Score system activity-classification output against ground-truth.')
    parser_score_ac.add_argument("-r", '--reference_file', type=str, required=True)
    parser_score_ac.add_argument("-y", '--hypothesis_file', type=str, required=True)
    parser_score_ac.add_argument("-a", '--activity_list_file', type=str, required=False, help="Use to filter activities from scoring (REF + HYP)")
    parser_score_ac.add_argument("-f", '--video_list_file', type=str, required=False, help="Used to filter files from scoring (REF + HYP)")
    parser_score_ac.add_argument("-o", "--output_dir", nargs='?', type=str, default="tmp")
    parser_score_ac.add_argument("-m", "--metrics", nargs='?', default="map", help="Available metrics: map, map_11, map_101, map_avg, map_auc")
    parser_score_ac.add_argument("-t", "--topk", nargs='?', default="1")
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
    parser_score_tad.set_defaults(func = tad_scorer_cmd)

    parser_validate_tad = subparsers.add_parser('validate-ref', help='Validate reference data (system dev)')
    parser_validate_tad.add_argument("-r", '--reference_file', type=str, required=True)
    parser_validate_tad.set_defaults(func = ref_validator_cmd)

    parser_validate_ac = subparsers.add_parser('validate-ac', help='Validate system output.')
    parser_validate_ac.add_argument("-r", '--reference_file', type=str, required=True)
    parser_validate_ac.add_argument("-y", '--hypothesis_file', type=str, required=False)
    parser_validate_ac.set_defaults(func = ac_validator_cmd)

    parser_validate_ac_gt = subparsers.add_parser('validate-ac-gt', help='Validate reference data.')
    parser_validate_ac_gt.add_argument("-r", '--reference_file', type=str, required=True)    
    parser_validate_ac_gt.set_defaults(func = ac_gt_validator_cmd)

    parser_validate_tad = subparsers.add_parser('validate-tad', help='Validate system output.')
    parser_validate_tad.add_argument("-r", '--reference_file', type=str, required=True)
    parser_validate_tad.add_argument("-y", '--hypothesis_file', type=str, required=True)
    parser_validate_tad.set_defaults(func = tad_validator_cmd)

    parser_validate_tad_gt = subparsers.add_parser('validate-tad-gt', help='Validate reference data.')
    parser_validate_tad_gt.add_argument("-r", '--reference_file', type=str, required=True)
    parser_validate_tad_gt.set_defaults(func = tad_gt_validator_cmd)

    parser_extract = subparsers.add_parser('extract', help='Extract system and activity level scores.')
    parser_extract.add_argument("-s", '--score_file', type=str, required=True)    
    parser_extract.add_argument("-o", "--output_dir", nargs='?', type=str, default="tmp")
    parser_extract.add_argument("-a", "--alignments", action="store_true", help="Extract alignment if TAD task (default: false)")
    parser_extract.set_defaults(func = extractor_cmd)

    parser_plot = subparsers.add_parser('plot-results', help='Extract system and activity results and generate plots.')
    parser_plot.add_argument("-f", '--score_file', type=str, required=True)    
    parser_plot.add_argument("-o", "--output_dir", nargs='?', type=str, default="tmp")
    parser_plot.set_defaults(func = plot_cmd)

    parser_remap = subparsers.add_parser('remap', help='Create new system output given label pair config.')
    parser_remap.add_argument("-m", '--mapping_file', type=str, required=True)    
    parser_remap.add_argument("-s", '--score_file', type=str, required=True)    
    parser_remap.add_argument("-o", '--output_file', type=str, required=True)        
    parser_remap.add_argument("-r", "--reference_file", action="store_true", help="Transform reference")
    parser_remap.set_defaults(func = remap_cmd)

    args = parser.parse_args()

     # Note that Logging (singleton) is inherited in submodules !
    if args.debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

    #log = logging.getLogger(__name__) 
    # Need to extra handle this as the error-message is cryptic to the user
    try:
        func = args.func
    except AttributeError:
        parser.error("too few arguments")
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