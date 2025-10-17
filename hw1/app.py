
import os
import sys
import pandas as pd
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

from src.model import ModelWrapper, ColumnSelector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/data/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        self.input_dir = '/data/input'
        self.output_dir = '/data/output'
        logger.info('Loading model weights...')
        self.service_model = ModelWrapper(model_path='/data/model_weights/best_rf_model.pkl')
        logger.info('Model loaded successfully, service initialized')

    def process_single_file(self, file_path):
        try:
            logger.info('Processing file: %s', file_path)
            input_df = pd.read_csv(file_path)

            logger.info('Adding new file to model wrapepr')
            self.service_model.update_scoring_file(input_df)
            
            logger.info('Preprocessing data')
            self.service_model.preprocess()

            logger.info('Making prediction')
            submission = self.service_model.inference(input_df)
            
            logger.info('Prepraring submission file')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"predictions_{timestamp}_{os.path.basename(file_path)}"
            submission.to_csv(os.path.join(self.output_dir, output_filename), index=False)
            logger.info('Predictions saved to: %s', output_filename)

            logger.info('Preparing feature importance json file')
            json_output_filename = f"importance_{timestamp}_{os.path.basename(file_path).replace('.csv', '.json')}"
            importance_data = self.service_model.export_top5_feature_importances()
            pd.Series(importance_data).to_json(os.path.join(self.output_dir, json_output_filename))
            logger.info('Feature importances saved to: %s', json_output_filename)

            logger.info('Scores density plot')
            plot_output_filename = f"scores_density_{timestamp}_{os.path.basename(file_path).replace('.csv', '.png')}"
            self.service_model.save_scores_density_plot(input_df, out_path=os.path.join(self.output_dir, plot_output_filename))
            logger.info('Scores density plot saved to: %s', plot_output_filename)

        except Exception as e:
            logger.error('Error processing file %s: %s', file_path, e, exc_info=True)
            return


class FileHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            logger.debug('New file detected: %s', event.src_path)
            self.service.process_single_file(event.src_path)

if __name__ == "__main__":
    logger.info('Starting ML scoring service...')
    service = ProcessingService()
    observer = Observer()
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info('File observer started')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
        observer.stop()
    observer.join()
