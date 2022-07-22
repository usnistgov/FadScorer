# -*- coding: utf-8 -*-
import argparse, sys, logging
import traceback

from .validation import validate_ac, validate_tad, validate_gt
from .datatypes import Dataset
from .io import *
from .plot import plot_tad_system, plot_prs, plot_all_tad_activity_pr, plot_all_ac_activity_pr

#
# Internal Scorer Version
# 

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
    if args.reference_file:        
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
        log.debug("[extract] TAD Task detected")
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
    parser = argparse.ArgumentParser(description='FADScorer: Fine-grained Activity Detection Scorer', prog="fad-scorer")
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
