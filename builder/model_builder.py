# -*- coding:utf-8 -*-
# author: Xinge
# @file: model_builder.py 

from network.cylinder_spconv_3d import get_model_class
from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv
from network.cylinder_fea_generator import cylinder_fea


def build(model_config):
    output_shape = model_config['output_shape']
    num_class = model_config['num_class']
    num_input_features = model_config['num_input_features']
    use_norm = model_config['use_norm']
    init_size = model_config['init_size']
    fea_dim = model_config['fea_dim']
    out_fea_dim = model_config['out_fea_dim']
    if model_config['model_architecture'] == 'cylinder_asym_clf_test':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_clf_test
        print('using cylinder_asym_clf_test')
        cylinder_3d_spconv_seg = Asymm_3d_spconv_clf_test(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)    
    elif model_config['model_architecture'] == 'cylinder_asym_clf_v1':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_clf_v1
        print('using cylinder_asym_clf_v1')
        cylinder_3d_spconv_seg = Asymm_3d_spconv_clf_v1(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)
    elif model_config['model_architecture'] == 'cylinder_asym_clf_v2':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_clf_v2
        print('using cylinder_asym_clf_v2')
        cylinder_3d_spconv_seg = Asymm_3d_spconv_clf_v2(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)
    elif model_config['model_architecture'] == 'cylinder_asym_clf_v3':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_clf_v3
        print('using cylinder_asym_clf_v3')
        cylinder_3d_spconv_seg = Asymm_3d_spconv_clf_v3(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)     
    elif model_config['model_architecture'] == 'cylinder_asym_clf_v4':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_clf_v4
        print('using cylinder_asym_clf_v4')
        cylinder_3d_spconv_seg = Asymm_3d_spconv_clf_v4(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)     
    elif model_config['model_architecture'] == 'cylinder_asym_clf_v5':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_clf_v5
        print('using cylinder_asym_clf_v5')
        cylinder_3d_spconv_seg = Asymm_3d_spconv_clf_v5(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)         
    elif model_config['model_architecture'] == 'cylinder_asym_clf_v6':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_clf_v6
        print('using cylinder_asym_clf_v6')
        cylinder_3d_spconv_seg = Asymm_3d_spconv_clf_v6(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)   
    elif model_config['model_architecture'] == 'cylinder_asym_clf_v7':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_clf_v7
        print('using cylinder_asym_clf_v7')
        cylinder_3d_spconv_seg = Asymm_3d_spconv_clf_v7(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class) 
    elif model_config['model_architecture'] == 'cylinder_asym_clf_v8':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_clf_v8
        print('using cylinder_asym_clf_v8')
        cylinder_3d_spconv_seg = Asymm_3d_spconv_clf_v8(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)       
    elif model_config['model_architecture'] == 'cylinder_asym_clf_v9':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_clf_v9
        print('using cylinder_asym_clf_v9')
        cylinder_3d_spconv_seg = Asymm_3d_spconv_clf_v9(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)   
    elif model_config['model_architecture'] == 'cylinder_asym_clf_v10':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_clf_v10
        print('using cylinder_asym_clf_v10')
        cylinder_3d_spconv_seg = Asymm_3d_spconv_clf_v10(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class) 
    elif model_config['model_architecture'] == 'cylinder_asym_clf_v11':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_clf_v11
        print('using cylinder_asym_clf_v11')
        cylinder_3d_spconv_seg = Asymm_3d_spconv_clf_v11(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)   
    elif model_config['model_architecture'] == 'cylinder_asym_clf_v12':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_clf_v12
        print('using cylinder_asym_clf_v12')
        cylinder_3d_spconv_seg = Asymm_3d_spconv_clf_v12(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)                     
    elif model_config['model_architecture'] == 'cylinder_asym_clf_v13':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_clf_v13
        print('using cylinder_asym_clf_v13')
        cylinder_3d_spconv_seg = Asymm_3d_spconv_clf_v13(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)           
    elif model_config['model_architecture'] == 'cylinder_asym_clf_v14':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_clf_v14
        print('using cylinder_asym_clf_v14')
        cylinder_3d_spconv_seg = Asymm_3d_spconv_clf_v14(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)           
    elif model_config['model_architecture'] == 'cylinder_asym_clf_v15':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_clf_v15
        print('using cylinder_asym_clf_v15')
        cylinder_3d_spconv_seg = Asymm_3d_spconv_clf_v15(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)             
    elif model_config['model_architecture'] == 'cylinder_asym_clf_v17':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_clf_v17
        print('using cylinder_asym_clf_v17')
        print("MLP Training!!")
        cylinder_3d_spconv_seg = Asymm_3d_spconv_clf_v17(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)                       
    elif model_config['model_architecture'] == 'cylinder_asym_clf_v18':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_clf_v18
        print('using Asymm_3d_spconv_clf_v18')
        print("MLP Training!!")
        cylinder_3d_spconv_seg = Asymm_3d_spconv_clf_v18(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)             
    elif model_config['model_architecture'] == 'cylinder_asym_original':
        from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv_original
        print("using original cylinder_asym original")
        cylinder_3d_spconv_seg = Asymm_3d_spconv_original(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)                     
    else:
        print("using original cylinder_asym")
        cylinder_3d_spconv_seg = Asymm_3d_spconv(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)


    cy_fea_net = cylinder_fea(grid_size=output_shape,
                              fea_dim=fea_dim,
                              out_pt_fea_dim=out_fea_dim,
                              fea_compre=num_input_features)

    model = get_model_class(model_config["model_architecture"])(
        cylin_model=cy_fea_net,
        segmentator_spconv=cylinder_3d_spconv_seg,
        sparse_shape=output_shape
    )

    return model
