from datetime import datetime
from pathlib import Path
import yaml
import sys
import shutil 
import random
import pickle
import os
import json
import yaml
import copy
import numpy as np

from tqdm import tqdm
import re
import shutil
import subprocess
from collections import OrderedDict

GARMENTCODERC_ROOT = str(Path(os.environ.get('GARMENTCODERC_ROOT', Path(__file__).resolve().parents[2] / 'GarmentCodeRC')).resolve())
if GARMENTCODERC_ROOT not in sys.path:
    sys.path.insert(1, GARMENTCODERC_ROOT)

# Custom
from assets.garment_programs.meta_garment import MetaGarment
from assets.bodies.body_params import BodyParameters
from pathlib import Path


wb_config_name = 'waistband'
skirt_configs =  {
    'SkirtCircle': 'flare-skirt',
    'AsymmSkirtCircle': 'flare-skirt',
    'GodetSkirt': 'godet-skirt',
    'Pants': 'pants',
    'Skirt2': 'skirt',
    'SkirtManyPanels': 'flare-skirt',
    'PencilSkirt': 'pencil-skirt',
    'SkirtLevels': 'levels-skirt',
}
all_skirt_configs = ['skirt', 'flare-skirt', 'godet-skirt', 'pencil-skirt', 'levels-skirt', 'pants']

with open('docs/all_float_paths.json', 'r') as f:
    all_float_paths = json.load(f)


with open('assets/design_params/design_used.yaml', 'r') as f:
    designs_config = yaml.safe_load(f)
    

def recursive_change_params(cfg, pred_cfg, invnorm_float=False, parent_path='design'):
    if ('type' not in cfg) or not isinstance(cfg['type'], str):
        for subpattern_n, subpattern_cfg in cfg.items():
            if (not isinstance(pred_cfg, dict)) or (subpattern_n not in pred_cfg):
                # print('isinstance(pred_cfg, dict)', cfg, pred_cfg, type(pred_cfg))
                continue

            subconfig = recursive_change_params(
                subpattern_cfg, pred_cfg[subpattern_n], invnorm_float=invnorm_float,
                parent_path=parent_path + '.' + subpattern_n)
            cfg[subpattern_n] = subconfig
    else:
        v = pred_cfg
        vtype = cfg['type']
        if vtype == 'float':
            if isinstance(v, str):  
                v = v.strip().replace(' ', '')
            v = float(v)

            if invnorm_float:
                v = np.clip(v, 0, 1)
                # Inverse normalization for float values
                lower_bd = float(cfg['range'][0])
                upper_bd = float(cfg['range'][1])
                v = v * (upper_bd - lower_bd) + lower_bd
                v = float(v)

            # if parent_path == 'design.waistband.waist':
            #     v = 0.85

        elif vtype == 'int':
            if isinstance(v, str):  
                v = v.strip().replace(' ', '')
            v = int(float(v) + 0.5)

            lower_bd = float(cfg['range'][0])
            upper_bd = float(cfg['range'][1])
            v = int(np.clip(v, lower_bd, upper_bd))
        
        elif vtype == 'bool':
            if isinstance(v, str):  
                v = v.strip().replace(' ', '')
            v = bool(v)

        elif vtype == 'select':
            all_ranges = cfg['range']
            if isinstance(v, str):  
                v = v.strip().replace(' ', '')
            
            if v not in all_ranges:
                v = all_ranges[0]
        
        elif vtype == 'select_null':
            all_ranges = cfg['range']
            if isinstance(v, str):  
                v = v.strip().replace(' ', '')
            
            if (v is None) or (v == 'null') or (v == 'None'):
                v = None
            elif v not in all_ranges:
                v = None

        cfg['v'] = v

    return cfg



def recursive_change_params_1float(cfg, pred_cfg, float_dict, invnorm_float=False, parent_path='design'):
    if ('type' not in cfg) or not isinstance(cfg['type'], str):
        for subpattern_n, subpattern_cfg in cfg.items():
            if (not isinstance(pred_cfg, dict)) or (subpattern_n not in pred_cfg):
                # print('isinstance(pred_cfg, dict)', cfg, pred_cfg, type(pred_cfg))
                continue
            
            if parent_path is None:
                parent_path_new = subpattern_n
            else:
                parent_path_new = parent_path + '.' + subpattern_n
            subconfig = recursive_change_params_1float(subpattern_cfg, pred_cfg[subpattern_n], float_dict=float_dict,
                                parent_path=parent_path_new, invnorm_float=invnorm_float)

            cfg[subpattern_n] = subconfig
    else:
        v = pred_cfg
        vtype = cfg['type']
        if vtype == 'float':
            assert parent_path in float_dict, (parent_path, list(float_dict.keys())[:3])
            v = float_dict[parent_path]
            if isinstance(v, str):  
                v = v.strip().replace(' ', '')
            v = float(v)

            if invnorm_float:
                v = np.clip(v, 0, 1)
                # Inverse normalization for float values
                lower_bd = float(cfg['range'][0])
                upper_bd = float(cfg['range'][1])
                v = v * (upper_bd - lower_bd) + lower_bd
                v = float(v)

        elif vtype == 'int':
            if isinstance(v, str):  
                v = v.strip().replace(' ', '')
            v = int(float(v) + 0.5)

            lower_bd = float(cfg['range'][0])
            upper_bd = float(cfg['range'][1])
            v = int(np.clip(v, lower_bd, upper_bd))
        
        elif vtype == 'bool':
            if isinstance(v, str):  
                v = v.strip().replace(' ', '')
            v = bool(v)

        elif vtype == 'select':
            all_ranges = cfg['range']
            if isinstance(v, str):  
                v = v.strip().replace(' ', '')
            
            if v not in all_ranges:
                v = all_ranges[0]
        
        elif vtype == 'select_null':
            all_ranges = cfg['range']
            if isinstance(v, str):  
                v = v.strip().replace(' ', '')
            
            if (v is None) or (v == 'null') or (v == 'None'):
                v = None
            elif v not in all_ranges:
                v = None

        cfg['v'] = v

    return cfg


def try_generate_garments(body_measurement_path, garment_output, garment_name, output_path, 
                          body_measurement='neutral', invnorm_float=False, float_dict=None):
    global designs_config
    bodies_measurements = {
        # Our model
        'neutral': './assets/bodies/mean_all.yaml',
        'mean_female': './assets/bodies/mean_female.yaml',
        'mean_male': './assets/bodies/mean_male.yaml',
        # SMPL
        'f_smpl': './assets/bodies/f_smpl_average_A40.yaml',
        'm_smpl': './assets/bodies/m_smpl_average_A40.yaml'
    }

    design_pred_raw = garment_output
    default_config = copy.deepcopy(designs_config)

    if 'design' not in design_pred_raw:
        design_pred_raw = {'design': design_pred_raw}

    if float_dict is not None:
        design = recursive_change_params_1float(default_config, design_pred_raw, float_dict,
                                                invnorm_float=invnorm_float, parent_path=None)
    else:
        design = recursive_change_params(default_config, design_pred_raw, invnorm_float=invnorm_float)

    design = design['design']

    config = {'design': design}
    if not os.path.exists(os.path.join(output_path, f'valid_garment_{garment_name}')):
        os.makedirs(os.path.join(output_path, f'valid_garment_{garment_name}'))

    with open(os.path.join(output_path, f'valid_garment_{garment_name}', 'design.yaml'), 'w') as f:
        yaml.dump(config, f)

    if body_measurement_path is None:
        body_measurement_path = os.environ.get('CHATGARMENT_BODY_MEASUREMENT_PATH')

    if body_measurement_path is None:
        print('bodies_measurements[body_measurement]', bodies_measurements[body_measurement])
        body = BodyParameters(bodies_measurements[body_measurement])
    else:
        if not os.path.isfile(body_measurement_path):
            raise FileNotFoundError(
                f'CHATGARMENT_BODY_MEASUREMENT_PATH does not exist: {body_measurement_path}'
            )
        print('body_measurement_path', body_measurement_path)
        body = BodyParameters(body_measurement_path)
        
    test_garment = MetaGarment('valid_garment', body, design)

    pattern = test_garment.assembly()
    if test_garment.is_self_intersecting():
        print(f'{test_garment.name} is Self-intersecting')

    # Save as json file
    folder = pattern.serialize(
        output_path, 
        tag=garment_name, 
        to_subfolder=True, 
        with_3d=False, with_text=False, view_ids=False)
    
    body.save(folder)
    print(f'Success! Garment {garment_name} saved to {folder}')


def text_float_composite(text, float_list, precision=None):
    text_new = ''
    if float_list is None:
        return text
    float_list = float_list.reshape(-1)
    
    seg_positions = [i for i in range(len(text)) if text.startswith('[SEG]', i)]
    if ('-1' in text) and len(seg_positions) < 3:
        seg_positions = [i for i in range(len(text)) if text.startswith('-1', i)]
        

    assert len(seg_positions) == len(float_list), (len(seg_positions), len(float_list), text)
    # assert len(seg_positions) == 50

    last_end = 0
    for i, seg_pos in enumerate(seg_positions):
        float_num = float_list[i].item()

        if precision is not None:
            float_num = round(float_num, precision)
            float_num = str(float_num)
        else:
            float_num = str(float_num)
        text_new = text_new + text[last_end:seg_pos] + float_num
        
        if '[SEG]' in text:
            last_end = seg_pos + 5
        else:
            last_end = seg_pos + 2
    
    text_new = text_new + text[last_end:]
    return text_new





class LazyDecoder(json.JSONDecoder):
    def decode(self, s, **kwargs):
        regex_replacements = [
            (re.compile(r'([^\\])\\([^\\])'), r'\1\\\\\2'),
            (re.compile(r',(\s*])'), r'\1'),
        ]
        for regex, replacement in regex_replacements:
            s = regex.sub(replacement, s)
        return super().decode(s, **kwargs)
    

def run_simultion_warp(pattern_spec, sim_config, output_path, easy_texture_path=None):
    from pygarment.meshgen.boxmeshgen import BoxMesh
    from pygarment.meshgen.simulation import run_sim
    import pygarment.data_config as data_config
    from pygarment.meshgen.sim_config import PathCofig
    
    props = data_config.Properties(sim_config) 
    props.set_section_stats('sim', fails={}, sim_time={}, spf={}, fin_frame={}, body_collisions={}, self_collisions={})
    props.set_section_stats('render', render_time={})

    spec_path = Path(pattern_spec)
    garment_name, _, _ = spec_path.stem.rpartition('_')  # assuming ending in '_specification'

    paths = PathCofig(
        in_element_path=spec_path.parent,  
        out_path=output_path, 
        in_name=garment_name,
        body_name='mean_all',    # 'f_smpl_average_A40'
        smpl_body=False,   # NOTE: depends on chosen body model
        add_timestamp=False,
        system_path=os.path.join(GARMENTCODERC_ROOT, 'system.json'),
        easy_texture_path=easy_texture_path
    )

    # Generate and save garment box mesh (if not existent)
    print(f"Generate box mesh of {garment_name} with resolution {props['sim']['config']['resolution_scale']}...")
    print('\nGarment load: ', paths.in_g_spec)

    garment_box_mesh = BoxMesh(paths.in_g_spec, props['sim']['config']['resolution_scale'])
    garment_box_mesh.load()
    garment_box_mesh.serialize(
        paths, store_panels=False, uv_config=props['render']['config']['uv_texture'])

    props.serialize(paths.element_sim_props)

    run_sim(
        garment_box_mesh.name, 
        props, 
        paths,
        save_v_norms=False,
        store_usd=False,  # NOTE: False for fast simulation!
        optimize_storage=False,   # props['sim']['config']['optimize_storage'],
        verbose=False
    )
    
    props.serialize(paths.element_sim_props)




def run_garmentcode_sim(json_paths_json):
    process = subprocess.Popen(
        ["python", "run_garmentcode_sim.py", 
        "--garment_json_path", str(json_paths_json)], stdout=subprocess.PIPE
    )

    process.wait()
    print('finished', json_paths_json, process.returncode)
    return


def run_garmentcode_parser_float50(all_json_spec_files, json_output, float_preds, output_dir):
    if 'upperbody_garment' in json_output:
        upper_config = json_output['upperbody_garment']
        lower_config = json_output['lowerbody_garment']

        float_preds = float_preds.reshape(2, -1)
        assert len(float_preds[0]) == len(all_float_paths)
        float_dict_upper = {
            k: v for k, v in zip(all_float_paths, float_preds[0])
        }
        float_dict_lower = {
            k: v for k, v in zip(all_float_paths, float_preds[1])
        }

        try_generate_garments(None, upper_config, 'upper', output_dir, invnorm_float=True, float_dict=float_dict_upper)
        try_generate_garments(None, lower_config, 'lower', output_dir, invnorm_float=True, float_dict=float_dict_lower)

        all_json_spec_files.append(
            os.path.join(output_dir, 'valid_garment_upper', f'valid_garment_upper_specification.json')
        )
        all_json_spec_files.append(
            os.path.join(output_dir, 'valid_garment_lower', f'valid_garment_lower_specification.json')
        )
    else:
        wholebody_config = json_output['wholebody_garment']

        float_preds = float_preds.reshape(-1)
        assert len(float_preds) == len(all_float_paths)
        float_dict = {
            k: v for k, v in zip(all_float_paths, float_preds)
        }

        try_generate_garments(None, wholebody_config, 'wholebody', output_dir, invnorm_float=True, float_dict=float_dict)

        all_json_spec_files.append(
            os.path.join(output_dir, 'valid_garment_wholebody', f'valid_garment_wholebody_specification.json')
        )
    
    return all_json_spec_files


def run_garmentcode_parser(all_json_spec_files, json_output, output_dir):
    if 'upperbody_garment' in json_output:
        upper_config = json_output['upperbody_garment']
        lower_config = json_output['lowerbody_garment']

        try_generate_garments(None, upper_config, 'upper', output_dir, invnorm_float=True)
        try_generate_garments(None, lower_config, 'lower', output_dir, invnorm_float=True)

        all_json_spec_files.append(
            os.path.join(output_dir, 'valid_garment_upper', f'valid_garment_upper_specification.json')
        )
        all_json_spec_files.append(
            os.path.join(output_dir, 'valid_garment_lower', f'valid_garment_lower_specification.json')
        )
    else:
        wholebody_config = json_output['wholebody_garment']
        try_generate_garments(None, wholebody_config, 'wholebody', output_dir, invnorm_float=True)

        all_json_spec_files.append(
            os.path.join(output_dir, 'valid_garment_wholebody', f'valid_garment_wholebody_specification.json')
        )
    
    return all_json_spec_files


def copy_results(info_dict, garment_id, output_dir):
    my_info = info_dict[int(garment_id)]
    print(list(my_info.keys()))

    if 'upper_garment' in my_info:
        upper_name = my_info['upper_garment'].split('/')[-1]
        image_path_origin = os.path.join(
            my_info['upper_garment'], f'{upper_name}_pattern.png'
        )
        image_path_tosave = os.path.join(output_dir, 'upper_gt_pattern.png')
        print('image_path_origin', image_path_origin, image_path_tosave)
        shutil.copy(image_path_origin, image_path_tosave)

        json_path = [item for item in os.listdir(my_info['upper_garment']) if item.endswith('_design_params.yaml')][0]
        shutil.copy(os.path.join(my_info['upper_garment'], json_path), os.path.join(output_dir, 'upper_garment_specification.yaml'))

    if 'lower_garment' in my_info:
        lower_name = my_info['lower_garment'].split('/')[-1]
        image_path_origin = os.path.join(
            my_info['lower_garment'], f'{lower_name}_pattern.png'
        )
        image_path_tosave = os.path.join(output_dir, 'lower_gt_pattern.png')
        print('image_path_origin', image_path_origin, image_path_tosave)
        shutil.copy(image_path_origin, image_path_tosave)

        json_path = [item for item in os.listdir(my_info['lower_garment']) if item.endswith('_design_params.yaml')][0]
        shutil.copy(os.path.join(my_info['lower_garment'], json_path), os.path.join(output_dir, 'lower_garment_specification.yaml'))
    
    if 'whole_garment' in my_info:
        whole_name = my_info['whole_garment'].split('/')[-1]
        image_path_origin = os.path.join(
            my_info['whole_garment'], f'{whole_name}_pattern.png'
        )
        image_path_tosave = os.path.join(output_dir, 'whole_gt_pattern.png')
        print('image_path_origin', image_path_origin, image_path_tosave)
        shutil.copy(image_path_origin, image_path_tosave)

        json_path = [item for item in os.listdir(my_info['whole_garment']) if item.endswith('_design_params.yaml')][0]
        shutil.copy(os.path.join(my_info['whole_garment'], json_path), os.path.join(output_dir, 'whole_garment_specification.yaml'))
    
    return






def ordered(d, desired_key_order):
    return OrderedDict([(key, d[key]) for key in desired_key_order])


def recursive_simplify_params(cfg, is_used=True, unused_configs=[], parent_path='design'):
    # change float to 4 decimal places
    if cfg is None:
        print(parent_path)

    cfg_new = {}
    if ('type' not in cfg) or not isinstance(cfg['type'], str):

        if 'enable_asym' in cfg: ############################################
            enable_asym = bool(cfg['enable_asym']['v'])
            if not enable_asym:
                cfg_new['enable_asym'] = cfg['enable_asym']['v']
                return cfg_new

        if parent_path == 'design.sleeve.cuff' and cfg['type']['v'] is None:
            return {'type': None}

        if parent_path == 'design.left.sleeve.cuff' and cfg['type']['v'] is None:
            return {'type': None}
        
        if parent_path == 'design.pants.cuff' and cfg['type']['v'] is None:
            return {'type': None}
        
        for subpattern_n, subpattern_cfg in cfg.items():
            if (subpattern_n in unused_configs) and ('meta' in cfg):
                continue
            else:
                subconfig = recursive_simplify_params(subpattern_cfg, is_used=is_used, parent_path=parent_path + '.' + subpattern_n)
            
            cfg_new[subpattern_n] = subconfig
    
    else:
        type_now = cfg['type']
        if type_now == 'float':
            lower_bd = float(cfg['range'][0])
            upper_bd = float(cfg['range'][1])

            float_val = cfg['v']
            float_val_normed = (float_val - lower_bd) / (upper_bd - lower_bd)
            cfg_new = f'<{float_val_normed:.6f}>'
        
        else:
            cfg_new = cfg['v']

    return cfg_new


toplevel_desired_key_order = ['meta', 'collar', 'flare-skirt', 'godet-skirt', 'left', 'levels-skirt', 'pants', 'pencil-skirt', 'shirt', 'skirt', 'sleeve', 'waistband']
def get_target_str_by_name(new_config):
    # Simplify the JSON configuration to remove unnecessary parameters
    new_config = new_config['design']

    ################ get unused_configs
    unused_configs = []
    ub_garment = new_config['meta']['upper']['v']
    if ub_garment is None:
        unused_configs += ['shirt', 'collar', 'sleeve', 'left']
    
    wb_garment = new_config['meta']['wb']['v']
    if not wb_garment:
        unused_configs.append(wb_config_name)
    
    lower_garment = new_config['meta']['bottom']['v']

    if lower_garment is None:
        unused_configs += all_skirt_configs
    else:
        unused_configs += copy.deepcopy(all_skirt_configs)
        unused_configs.remove(skirt_configs[lower_garment])

        if 'base' in new_config[skirt_configs[lower_garment]]:
            base_garment = new_config[skirt_configs[lower_garment]]['base']['v']
            unused_configs.remove(skirt_configs[base_garment])

    new_config = recursive_simplify_params(new_config, is_used=True, unused_configs=unused_configs)
    toplevel_desired_key_order_new = []
    for key in toplevel_desired_key_order:
        if key in new_config:
            toplevel_desired_key_order_new.append(key)

    new_config = ordered(new_config, toplevel_desired_key_order_new)

    return new_config


def extract_all_floats_wfloats(answerdata):
    #### get all floats
    left_bracket = [m.start() for m in re.finditer("'<", answerdata)]
    right_bracket = [m.start() for m in re.finditer(">'", answerdata)]


    answer_data_processed = ''
    last_end = 0
    for start_idx, end_idx in zip(left_bracket, right_bracket):
        float_str = answerdata[start_idx+2:end_idx-1]
        float_str_new = f'{float(float_str):.3f}'

        answer_data_processed = answer_data_processed + answerdata[last_end:start_idx] + float_str_new
        last_end = end_idx + 2
    
    answer_data_processed = answer_data_processed + answerdata[last_end:]

    return answer_data_processed


def get_simplified_json_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    simplified_config = get_target_str_by_name(config)
    simplified_config_str = extract_all_floats_wfloats(json.dumps(simplified_config).replace('"', "'"))
    
    return simplified_config_str


if __name__ == "__main__":
    text_output = "{'wholebody_garment': {'meta': {'upper': 'Shirt', 'wb': null, 'bottom': 'PencilSkirt'}, 'collar': {'f_collar': 'CircleNeckHalf', 'b_collar': 'CircleNeckHalf', 'width': [SEG], 'fc_depth': [SEG], 'bc_depth': [SEG], 'fc_angle': 89, 'bc_angle': 90, 'f_bezier_x': [SEG], 'f_bezier_y': [SEG], 'b_bezier_x': [SEG], 'b_bezier_y': [SEG], 'f_flip_curve': false, 'b_flip_curve': false, 'component': {'style': 'SimpleLapel', 'depth': 6, 'lapel_standing': true, 'hood_depth': [SEG], 'hood_length': [SEG]}}, 'left': {'enable_asym': false, 'shirt': {'strapless': false, 'width': [SEG], 'flare': [SEG]}, 'collar': {'f_collar': 'CurvyNeckHalf', 'b_collar': 'Bezier2NeckHalf', 'width': [SEG], 'fc_angle': 94, 'bc_angle': 78, 'f_bezier_x': [SEG], 'f_bezier_y': [SEG], 'b_bezier_x': [SEG], 'b_bezier_y': [SEG], 'f_flip_curve': false, 'b_flip_curve': false}, 'sleeve': {'sleeveless': true, 'armhole_shape': 'ArmholeCurve', 'length': [SEG], 'connecting_width': [SEG], 'end_width': [SEG], 'sleeve_angle': 45, 'opening_dir_mix': [SEG], 'standing_shoulder': false, 'standing_shoulder_len': [SEG], 'connect_ruffle': [SEG], 'smoothing_coeff': [SEG], 'cuff': {'type': null, 'top_ruffle': [SEG], 'cuff_len': [SEG], 'skirt_fraction': [SEG], 'skirt_flare': [SEG], 'skirt_ruffle': [SEG]}}}, 'levels-skirt': {'base': 'Skirt2', 'level': 'AsymmSkirtCircle', 'num_levels': 4, 'level_ruffle': [SEG], 'length': [SEG], 'rise': [SEG], 'base_length_frac': [SEG]}, 'pencil-skirt': {'length': [SEG], 'rise': [SEG], 'flare': [SEG], 'low_angle': 6, 'front_slit': [SEG], 'back_slit': [SEG], 'left_slit': [SEG], 'right_slit': [SEG], 'style_side_cut': null, 'style_side_file': 'assets\\img\\Logo_adjusted.svg'}, 'shirt': {'strapless': false, 'length': [SEG], 'width': [SEG], 'flare': [SEG]}, 'sleeve': {'sleeveless': false, 'armhole_shape': 'ArmholeCurve', 'length': [SEG], 'connecting_width': [SEG], 'end_width': [SEG], 'sleeve_angle': 41, 'opening_dir_mix': [SEG], 'standing_shoulder': false, 'standing_shoulder_len': [SEG], 'connect_ruffle': [SEG], 'smoothing_coeff': [SEG], 'cuff': {'type': 'CuffBandSkirt', 'top_ruffle': [SEG], 'cuff_len': [SEG], 'skirt_fraction': [SEG], 'skirt_flare': [SEG], 'skirt_ruffle': [SEG]}}}}"
    float_list = [0.46666, 0.05882, 0.35168, 0.31565, 0.55555, 0.11111, 0.05555, 0.26161, 0.0, 0.0, 0.84385, 0.5, 0.5, 0.45154, 0.5, 0.74222, 0.52844, 0.1, 0.64548, 0.64705, 0.33841, 0.33125, 0.5, 0.94067, 0.05882, 0.18142, 0.90588, 0.0, 0.78553, 0.23471, 0.09625, 0.57287, 0.60905, 1.0, 0.80807, 0.89248, 0.06192, 0.0, 0.0, 0.23333, 0.00239, 0.42758, 0.03335, 0.13036, 0.44444, 0.58823, 0.04042, 0.15044, 0.5, 0.2051, 0.05882, 0.54097, 0.34642, 0.15214]
    
    '''
    text_output = "{'upperbody_garment': {'meta': {'upper': 'FittedShirt', 'wb': 'FittedWB', 'bottom': null}, 'collar': {'f_collar': 'SquareNeckHalf', 'b_collar': 'CircleNeckHalf', 'width': [SEG], 'fc_depth': [SEG], 'bc_depth': [SEG], 'fc_angle': 88, 'bc_angle': 91, 'f_bezier_x': [SEG], 'f_bezier_y': [SEG], 'b_bezier_x': [SEG], 'b_bezier_y': [SEG], 'f_flip_curve': false, 'b_flip_curve': false, 'component': {'style': 'SimpleLapel', 'depth': 7, 'lapel_standing': true, 'hood_depth': [SEG], 'hood_length': [SEG]}}, 'left': {'enable_asym': false, 'shirt': {'strapless': false, 'width': [SEG], 'flare': [SEG]}, 'collar': {'f_collar': 'CircleNeckHalf', 'b_collar': 'CircleNeckHalf', 'width': [SEG], 'fc_angle': 74, 'bc_angle': 76, 'f_bezier_x': [SEG], 'f_bezier_y': [SEG], 'b_bezier_x': [SEG], 'b_bezier_y': [SEG], 'f_flip_curve': false, 'b_flip_curve': false}, 'sleeve': {'sleeveless': false, 'armhole_shape': 'ArmholeSquare', 'length': [SEG], 'connecting_width': [SEG], 'end_width': [SEG], 'sleeve_angle': 16, 'opening_dir_mix': [SEG], 'standing_shoulder': false, 'standing_shoulder_len': [SEG], 'connect_ruffle': [SEG], 'smoothing_coeff': [SEG], 'cuff': {'type': null, 'top_ruffle': [SEG], 'cuff_len': [SEG], 'skirt_fraction': [SEG], 'skirt_flare': [SEG], 'skirt_ruffle': [SEG]}}}, 'levels-skirt': {'base': 'Skirt2', 'level': 'AsymmSkirtCircle', 'num_levels': 3, 'level_ruffle': [SEG], 'length': [SEG], 'rise': [SEG], 'base_length_frac': [SEG]}, 'shirt': {'strapless': true, 'length': [SEG], 'width': [SEG], 'flare': [SEG]}, 'sleeve': {'sleeveless': true, 'armhole_shape': 'ArmholeAngle', 'length': [SEG], 'connecting_width': [SEG], 'end_width': [SEG], 'sleeve_angle': 47, 'opening_dir_mix': [SEG], 'standing_shoulder': false, 'standing_shoulder_len': [SEG], 'connect_ruffle': [SEG], 'smoothing_coeff': [SEG], 'cuff': {'type': 'CuffBandSkirt', 'top_ruffle': [SEG], 'cuff_len': [SEG], 'skirt_fraction': [SEG], 'skirt_flare': [SEG], 'skirt_ruffle': [SEG]}}, 'waistband': {'waist': [SEG], 'width': [SEG]}}, 'lowerbody_garment': {'meta': {'upper': null, 'wb': 'FittedWB', 'bottom': 'PencilSkirt'}, 'levels-skirt': {'base': 'PencilSkirt', 'level': 'Skirt2', 'num_levels': 5, 'level_ruffle': [SEG], 'length': [SEG], 'rise': [SEG], 'base_length_frac': [SEG]}, 'pencil-skirt': {'length': [SEG], 'rise': [SEG], 'flare': [SEG], 'low_angle': -27, 'front_slit': [SEG], 'back_slit': [SEG], 'left_slit': [SEG], 'right_slit': [SEG], 'style_side_cut': null, 'style_side_file': 'assets\\img\\test_shape.svg'}, 'waistband': {'waist': [SEG], 'width': [SEG]}}}"
    float_list = [0.89045, 0.30006, 0.0917, 0.27777, 0.64314, 0.92596, 0.30741, 0.0, 0.0, 0.0, 0.89708, 0.42263, 0.20624, 0.42889, 0.21532, 0.29021, 0.77593, 0.1, 0.03034, 0.64705, 0.75707, 0.82437, 0.5, 0.03369, 0.05882, 0.3254, 0.50061, 0.15035, 0.60268, 0.19597, 0.54243, 0.47119, 0.23333, 0.16666, 0.33333, 0.38813, 0.1, 0.20136, 0.58823, 0.69951, 0.90264, 0.5, 0.32977, 0.05882, 0.21391, 0.07285, 0.50931, 0.53435, 0.20585, 0.6214, 0.87315, 0.36322, 0.83206, 0.82466, 0.5495, 0.18897, 0.0, 0.81245, 0.0, 0.0, 0.0, 0.11111]
    '''
    import torch
    float_list = torch.tensor(float_list)
    garment_id = 1

    text_output = text_float_composite(text_output, float_list)

    print('text_output', text_output)

    json_output = json.loads(text_output.replace("'", '"'), cls=LazyDecoder)

    if 'upperbody_garment' in json_output:
        upper_config = json_output['upperbody_garment']
        lower_config = json_output['lowerbody_garment']
        try_generate_garments(None, upper_config, garment_id, 'runs/try_vis', invnorm_float=True)
        try_generate_garments(None, lower_config, garment_id, 'runs/try_vis', invnorm_float=True)
    
    else:
        whole_config = json_output['wholebody_garment']
        try_generate_garments(None, whole_config, garment_id, 'runs/try_vis', invnorm_float=True)
    

    run_simultion_warp(
        'runs/try_vis/valid_garment_1/valid_garment_1_specification.json',
        'assets/Sim_props/default_sim_props.yaml',
        'runs/try_vis/valid_garment_1'
    )
