# -*- coding: utf-8 -*-
#BEGIN_HEADER
import logging
import os
import uuid
from biome_classification import load_model, load_inference_data, inference, generate_output_file_list, generate_html_list, waterfall
from installed_clients.KBaseReportClient import KBaseReport
#END_HEADER


class biome_classification:
    '''
    Module Name:
    biome_classification

    Module Description:
    A KBase module: biome_classification
    '''

    ######## WARNING FOR GEVENT USERS ####### noqa
    # Since asynchronous IO can lead to methods - even the same method -
    # interrupting each other, you must be *very* careful when using global
    # state. A method could easily clobber the state set by another while
    # the latter method is running.
    ######################################### noqa
    VERSION = "0.0.1"
    GIT_URL = "https://github.com/kbaseapps/biome_classification.git"
    GIT_COMMIT_HASH = "047f6c523fa7e15921c34aa2abc90361720665fc"

    #BEGIN_CLASS_HEADER
    #END_CLASS_HEADER

    # config contains contents of config file in a hash or None if it couldn't
    # be found
    def __init__(self, config):
        #BEGIN_CONSTRUCTOR
        self.callback_url = os.environ['SDK_CALLBACK_URL']
        self.shared_folder = config['scratch']
        logging.basicConfig(format='%(created)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        #END_CONSTRUCTOR
        pass


    def run_biome_classification(self, ctx, params):
        """
        This example function accepts any number of parameters and returns results in a KBaseReport
        :param params: instance of mapping from String to unspecified object
        :returns: instance of type "ReportResults" -> structure: parameter
           "report_name" of String, parameter "report_ref" of String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN run_biome_classification

        # step 1: load catboost model
        logging.info('Loading the model...')
        model = load_model()
        logging.info('Model successfully loaded!')

        # step 2: load users' data for prediction using catbost model
        logging.info('Loading user input data...')
        sample_id_list, X = load_inference_data()
        logging.info('User data successfully loaded!')

        # step 3: run inference(model.predict) method
        logging.info('User data successfully loaded!')
        print("!!!!!!params:", params)
        n_labels = params['Top_possible_labels']
        n_features = params['Top_important_features']
        output_dir = inference(model, sample_id_list, X, n_labels)
        waterfall(output_dir, model, sample_id_list, X, n_features)

        # step 4: initialize report client
        kbase_report_client = KBaseReport(self.callback_url, service_ver='dev')

        # step 5: generate report
        output_files = generate_output_file_list(output_dir, self.shared_folder)
        html_report = generate_html_list(self.shared_folder)

        report_params = {'message': '',
                         'workspace_name': params.get('workspace_name'),
                         'file_links': output_files,
                         'html_links': html_report,
                         'direct_html_link_index': 0,
                         'html_window_height': 333,
                         'report_object_name': 'biome_classification_report_' + str(uuid.uuid4())}
        report_info = kbase_report_client.create_extended_report(report_params)

        # # STEP 6: contruct the output to send back
        output = {'report_name': report_info['name'],
                  'report_ref': report_info['ref'],
                  'result_directory': output_dir,
                  }
        #END run_biome_classification

        # At some point might do deeper type checking...
        if not isinstance(output, dict):
            raise ValueError('Method run_biome_classification return value ' +
                             'output is not type dict as required.')
        # return the results
        return [output]

    def status(self, ctx):
        #BEGIN_STATUS
        returnVal = {'state': "OK",
                     'message': "",
                     'version': self.VERSION,
                     'git_url': self.GIT_URL,
                     'git_commit_hash': self.GIT_COMMIT_HASH}
        #END_STATUS
        return [returnVal]
