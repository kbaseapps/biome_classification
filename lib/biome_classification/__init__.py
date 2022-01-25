import base64
import csv
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shap
import uuid
import zipfile

from base import Core
from catboost import CatBoostClassifier
from pathlib import Path
from shutil import copytree


MODULE_DIR = "/kb/module"
TEMPLATES_DIR = os.path.join(MODULE_DIR, "lib/templates")
OUTPUT_DIR = '/opt/work/outputdir'


class BiomeClassification(Core):
    def __init__(self, ctx, config, clients_class=None):
        super().__init__(ctx, config, clients_class)
        # Here we adjust the instance attributes for our convenience.
        self.report = self.clients.KBaseReport
        # self.shared_folder is defined in the Core App class.

    def do_analysis(self, params: dict):
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
        n_labels = int(params['Top_possible_labels'])
        n_features = int(params['Top_important_features'])
        inference(model, sample_id_list, X, n_labels)
        waterfall(model, sample_id_list, X, n_features)

        # step 4: generate report
        # output_files = generate_output_file_list(output_dir, self.shared_folder)
        # html_report = generate_html_list(self.shared_folder)

        # report_params = {'message': '',
        #                  'workspace_name': params.get('workspace_name'),
        #                  'file_links': output_files,
        #                  'html_links': html_report,
        #                  'direct_html_link_index': 0,
        #                  'html_window_height': 333,
        #                  'report_object_name': 'biome_classification_report_' + str(uuid.uuid4())}
        # report_info = self.report.create_extended_report(report_params)
        #
        # output = {'report_name': report_info['name'],
        #           'report_ref': report_info['ref'],
        #           'result_directory': output_dir,
        #           }
        # This path is required to properly use the template.
        params["output_value"] = "hello"

        # This is the method that generates the HTML report
        return self.generate_report(params)

    def generate_report(self, params: dict):
        """
        This method is where to define the variables to pass to the report.
        """
        reports_path = os.path.join(self.shared_folder, "reports")
        copytree(OUTPUT_DIR, reports_path)
        # Path to the Jinja template. The template can be adjusted to change
        # the report.
        template_path = os.path.join(TEMPLATES_DIR, "report.html")
        # The keys in this dictionary will be available as variables in the
        # Jinja template. With the current configuration of the template
        # engine, HTML output is allowed.

        # HARD CODED
        # 1. table path
        first_dir = os.path.join(reports_path, 'EB271-05-02')
        first_tsv_file = os.path.join(first_dir, 'EB271-05-02_top_predicted_biomes.tsv')
        with open(first_tsv_file) as f:
            df = pd.read_csv(f, sep="\t")
            df.reset_index(drop=True, inplace=True)

        second_dir = os.path.join(reports_path, 'EB271-02-01')
        second_tsv_file = os.path.join(second_dir, 'EB271-02-01_top_predicted_biomes.tsv')
        with open(second_tsv_file) as f:
            df_2 = pd.read_csv(f, sep="\t")
            df_2.reset_index(drop=True, inplace=True)

        # 2. image path
        img_path = 'EB271-05-02/EB271-05-02_feature_importance.png'
        img_path_2 = 'EB271-02-01/EB271-02-01_feature_importance.png'

        tabs = {'EB271-05-02': dict(df=df.to_html(), image=img_path),
                'EB271-02-01': dict(df=df_2.to_html(), image=img_path_2)}
        template_variables = dict(

            tabs=tabs
        )
        """
        template_variables = dict(
            count_df_html=count_df_html,
            headers=headers,
            scores_df_html=scores_df_html,
            table=table,
            upa=params["upa"],
            output_value=params["output_value"],
        )
        """
        # The KBaseReport configuration dictionary
        config = dict(
            report_name=f"Biome_Classification_Report_{str(uuid.uuid4())}",
            reports_path=reports_path,
            template_variables=template_variables,
            workspace_name=params["workspace_name"],
        )
        return self.create_report_from_template(template_path, config)


def waterfall(model, sample_ids, X, display_features=10):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    num_features = len(shap_values)
    prediction = model.predict(X)
    for i, sample in enumerate(sample_ids):
        pred_cls_idx = np.where(model.classes_ == prediction[i][0])[0][0]
        shap.waterfall_plot(shap.Explanation(values=shap_values[pred_cls_idx][i],
                                             base_values=explainer.expected_value[pred_cls_idx], data=X.iloc[i],
                                             feature_names=X.columns.tolist()),
                            max_display=min(display_features, num_features),
                            show=False)
        fig = plt.gcf()
        plt.title("{}_feature_importance.png".format(sample))
        fig.savefig(os.path.join(OUTPUT_DIR , sample, "{}_feature_importance.png".format(sample)), bbox_inches='tight')
        plt.close()


def load_model():
    model = CatBoostClassifier(
            loss_function='MultiClass',
            custom_metric='Accuracy',
            learning_rate=0.15,
            random_seed=42,
            l2_leaf_reg=3,
            iterations=3)
    model_path = os.path.join('/kb/module/data', 'model_app.json')
    model.load_model(model_path, format='json')
    return model


def load_inference_data():
    # inference data set's first column is data sample id
    inference_data_path = os.path.join('/kb/module/data', 'enigma.tsv')
    with open(inference_data_path, 'r') as f:
        df = pd.read_csv(f, sep="\t")
    X = df.iloc[:, 1:]
    sample_id = df[df.columns[0]]
    return sample_id, X


def inference(model, sample_ids, inference_data, n_labels=10):
    prediction_label = model.predict(inference_data)
    prediction_prob = model.predict(inference_data, prediction_type="Probability")
    class_list = model.classes_
    output_dir = '/opt/work/outputdir'
    for i, sample in enumerate(sample_ids):
        # Step 1. set path for tsv table and plot
        sample_dir = os.path.join(output_dir, sample)
        figure_path = os.path.join(sample_dir, "{}_top_predicted_biomes.png".format(sample))
        tsv_path = os.path.join(sample_dir, "{}_top_predicted_biomes.tsv".format(sample))
        Path(sample_dir).mkdir(parents=True, exist_ok=True)

        # Step 2. extract top predicted biomes
        sample_prob = prediction_prob[i]
        sorted_idx = np.argsort(sample_prob)[::-1]  # from highest to lowest
        biome_prob_pair = [(class_list[idx], sample_prob[idx]) for idx in sorted_idx[:n_labels]]

        # Step 3. save top predicted biome bar plot
        biome, prob = zip(*biome_prob_pair)
        plt.barh(biome, prob)
        plt.title('{}'.format(sample))
        plt.gca().invert_yaxis()
        plt.xlabel('probability')
        plt.savefig(figure_path, bbox_inches='tight')
        plt.close()

        # Step 4. save top predicted biome tsv table
        with open(tsv_path, 'w') as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(['Predicted Biome', 'Probabality'])
            for biome, prob in biome_prob_pair:
                writer.writerow([biome, prob])


def generate_output_file_list(result_directory, shared_folder):
    output_files = list()
    output_directory = os.path.join(shared_folder, str(uuid.uuid4()))
    os.mkdir(output_directory)
    result_file = os.path.join(output_directory, 'biome_classification_result.zip')

    with zipfile.ZipFile(result_file, 'w',
                         zipfile.ZIP_DEFLATED,
                         allowZip64=True) as zip_file:
        for root, dirs, files in os.walk(result_directory):
            for file in files:
                if file.endswith('tsv') or file.endswith('png'):
                    zip_file.write(os.path.join(root, file),
                                   os.path.join(os.path.basename(root), file))

    output_files.append({'path': result_file,
                         'name': os.path.basename(result_file),
                         'label': os.path.basename(result_file),
                         'description': 'File(s) generated by DESeq2 App'})
    return output_files


def tsv2html(tsv_file_name, html_file_name):
    df = pd.read_csv(tsv_file_name, sep='\t', header=0)
    old_width = pd.get_option('display.max_colwidth')
    pd.set_option('display.max_colwidth', -1)
    with open(html_file_name, 'w') as html_file:
        html_file.write(df.to_html(index=False))
    pd.set_option('display.max_colwidth', old_width)


def generate_html_list(shared_folder):
    output_report_directory = os.path.join(shared_folder, "output_report_directory")
    tsv_file_path = '/opt/work/outputdir/EB271-02-01/EB271-02-01_top_predicted_biomes.tsv'
    html_file_path = '/opt/work/outputdir/EB271-02-01/EB271-02-01_top_predicted_biomes.html'
    tsv2html(tsv_file_path, html_file_path)

    copytree("/opt/work/outputdir", output_report_directory)
    html_links = [
        {
            "path": os.path.join(output_report_directory, 'EB271-02-01'),
            "name": 'EB271-02-01_top_predicted_biomes.html',
            "description": 'Biome prediction report',
        }
    ]
    return html_links




