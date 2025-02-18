# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import numpy as np
import os
import svs


DATA_PATH = os.environ.get("DATA_PATH")


def dense_build_index(
        index_path: str,
        vec_file: str,
        index_fn,
        dist_type: svs.DistanceType,
        corpus_dtype: svs.DataType = svs.float32,
        index_kwargs: dict = {},
        calib_kwargs: dict = {},
        search_window: int = None,
        logger: logging.Logger = None,
):
    """
    Builds index with SVS assuming doc dataset vectors have been generated
    """
    # Load the vector data
    cfg = index_kwargs.pop('config_path', None)  # Check if previously saved index configuration exists
    if cfg is not None:
        saved_index_data = os.path.join(cfg, 'data')
        assert vec_file == saved_index_data, "You provided a Vamana configuration but no dataset associated " + \
                                             f"with a saved Vamana index. Expected vec_file={saved_index_data}"
    logger.info(f"Loading vectors from {vec_file}")
    try:
        # NOTE: dataloader had an error unless using the exact file path & dtype. File an issue?
        vec_data = svs.VectorDataLoader(vec_file, corpus_dtype)
        logger.info("Vectors loaded with VectorDataLoader")
    except Exception as e:
        logger.info(f"Vector loader didn't work: {e}")
        vec_data = svs.read_vecs(vec_file)
        logger.info(f"But direct loading of vectors did work.")

    # For Approx Nearest Neighbors (ANN) search, initialize the search graph
    if (index_fn == svs.Vamana) and (cfg is None):
        logger.info(f"Building index from {vec_file} with {index_kwargs}")
        # Must build the index, so let's load the build parameters
        build_args = index_kwargs.pop('vamana_build_params', None)
        if build_args:
            assert isinstance(build_args, dict), "vamana_build_params must be a dict"
            build_params = svs.VamanaBuildParameters(**build_args)
        else:
            build_params = svs.VamanaBuildParameters()
        # TODO: file issue about inconsistent input argument naming
        index = svs.Vamana.build(build_parameters=build_params, data_loader=vec_data, distance_type=dist_type,
                                   **index_kwargs)
    else:
        # Load an existing index. Flat indices do not require building, just initializing here.
        if cfg:
            # Load search graph & put the config path back in the parameters
            index_kwargs.update({'graph_loader': svs.GraphLoader(os.path.join(cfg, 'graph')), 'config_path': cfg})
            # Remove the build parameters since we're just loading it
            _ = index_kwargs.pop('vamana_build_params', None)
        logger.info(f"Loading index from {vec_file} with {index_kwargs}")
        index = index_fn(data_loader=vec_data, distance=dist_type, **index_kwargs)

    if calib_kwargs:  # calibrate Vamana graph if requested
        params = svs.VamanaCalibrationParameters()
        params.use_existing_parameter_values = True
        params.search_buffer_optimization = svs.VamanaSearchBufferOptimization.ROIOnly
        params.train_prefetchers = False
        # queries: numpy.ndarray[float16], groundtruth: numpy.ndarray[numpy.uint32], num_neighbors: int
        calib_str = calib_kwargs.pop('calib_prefix')
        logger.info(f"Running search window calibration with {calib_kwargs}")
        calib_path = os.path.join(DATA_PATH, 'vamana_calib_data')
        calib_kwargs.update({'queries': np.load(f'{calib_path}/{calib_str}_queries.npy'),
                             'groundtruth': np.load(f'{calib_path}/{calib_str}_ground_truth.npy')})
        #logger.info(f"DEBUG {calib_kwargs['queries'].shape}")
        search_params = index.experimental_calibrate(calibration_parameters=params, **calib_kwargs)
        logger.info(f"Finished calibration, set search parameters to {search_params}")
    
    if search_window:
        index.search_window_size = search_window
        logger.info(f"Manually set search parameters to {index.search_parameters}")
    
    if (not index_fn == svs.Flat) and (cfg is None):
        try:
            index.save(config_directory=index_path,
                       graph_directory=os.path.join(index_path, 'graph'),
                       data_directory=os.path.join(index_path, 'data'))
            logger.info(f"Saved index to {index_path}")
        except Exception as e:
            logger.info(e)
            logger.warning(f"Unsupported index type {index_fn} could not be saved.")
    else:
        logger.info("Indices of type 'svs.Flat' cannot be saved.")

    return index
