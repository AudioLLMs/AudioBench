
import os
import fire
import json
import logging
from tqdm import tqdm

import torch

from dataset import Dataset
from model import Model

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

def do_model_prediction(input_data, model, batch_size):

    if batch_size not in [1, -1]:
        raise NotImplementedError("Batch size {} not implemented yet".format(batch_size))
    
    if batch_size == -1:
        model_predictions = model.generate(input_data)
    
    else:
        model_predictions = []
        for inputs in tqdm(input_data, leave=False):
            outputs = model.generate(inputs)
            if isinstance(outputs, list):
                model_predictions.extend(outputs)
            else:
                model_predictions.append(outputs)
                
    return model_predictions


def main(
        dataset_name      : str  = None,
        model_name        : str  = None,
        batch_size        : int  = 1,     # it is now a dummy parameter
        overwrite         : bool = False,
        metrics           : str  = None,
        number_of_samples : int  = -1,
        ):

    logger.info("= = "*20)
    logger.info("Dataset name: {}".format(dataset_name))
    logger.info("Model name: {}".format(model_name))
    logger.info("Batch size: {}".format(batch_size))
    logger.info("Overwrite: {}".format(overwrite))
    logger.info("Metrics: {}".format(metrics))
    logger.info("Number of samples: {}".format(number_of_samples))
    logger.info("= = "*20)

    # If the final score log exists, skip the evaluation
    if not overwrite and os.path.exists('log/{}/{}_{}_score.json'.format(model_name, dataset_name, metrics)):
        logger.info("Evaluation has been done before. Skip the evaluation.")
        logger.info("\n\n\n\n\n")
        return

    if model_name == 'WavLLM_fairseq':
        batch_size = -1
        logger.info("Batch size is set to -1 for WavLLM_fairseq model.")

    if batch_size != 1 and model_name != 'WavLLM_fairseq':
        raise NotImplementedError("Batch size {} not implemented yet".format(batch_size))
    elif batch_size != -1 and model_name == 'WavLLM_fairseq':
        raise NotImplementedError("Batch size {} not implemented yet for WavLLM_fairseq".format(batch_size))

    dataset = Dataset(dataset_name, number_of_samples)

    if overwrite or not os.path.exists('log/{}/{}.json'.format(model_name, dataset_name)):
        logger.info("Overwrite is enabled or the results are not found. Try to infer with the model: {}.".format(model_name))
    
        # Load model
        model = Model(model_name)

        # Infer with model
        model_predictions           = do_model_prediction(dataset.input_data, model, batch_size=batch_size)
        data_with_model_predictions = dataset.dataset_processor.format_model_predictions(dataset.input_data, model_predictions)

        # Save the result with predictions
        os.makedirs('log/{}'.format(model_name), exist_ok=True)
        with open('log/{}/{}.json'.format(model_name, dataset_name), 'w') as f:
            json.dump(data_with_model_predictions, f, indent=4, ensure_ascii=False)

    data_with_model_predictions = json.load(open('log/{}/{}.json'.format(model_name, dataset_name)))

    # Metric evaluation
    try:
        # Clear the cache to avoid memory leak
        logger.info("Clear the cache to avoid memory leak")
        del model
        torch.cuda.empty_cache()
    except: 
        pass

    results = dataset.dataset_processor.compute_score(data_with_model_predictions, metrics=metrics)

    # Print the result with metrics
    logger.info('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')
    logger.info('Dataset name: {}'.format(dataset_name.upper()))
    logger.info('Model name: {}'.format(model_name.upper()))
    logger.info(json.dumps({metrics: results[metrics]}, indent=4, ensure_ascii=False))
    logger.info('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')

    # Save the scores
    with open('log/{}/{}_{}_score.json'.format(model_name, dataset_name, metrics), 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
