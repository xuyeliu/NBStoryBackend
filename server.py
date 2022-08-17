from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import os
from min_example import interface, load_model
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import pytest
import backend_api
import re
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline


app = Flask(__name__, static_folder="../client/dist", template_folder="../client")
port = int(os.getenv('PORT', 8080))
CORS(app)
config, model, asttok, tdatstok, comstok = load_model()
def re_0002(i):
    # split camel case and remove special characters
    tmp = i.group(0)
    if len(tmp) > 1:
        if tmp.startswith(' '):
            return tmp
        else:
            return '{} {}'.format(tmp[0], tmp[1])
    else:
        return ' '.format(tmp)
re_0001_ = re.compile(r'([^a-zA-Z0-9 ])|([a-z0-9_][A-Z])')

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/test")
def test():
	return "Test Successful!"

@app.route("/handshake", methods=['GET'])
def handshake():
	return "success"

@app.route('/submit_payload_bullet_point', methods=['POST'])
def submit_payload_bullet_point():
    # get the parameters from the POST request(these would be required as parameters for your main()/Interface function)
    
    bullet_points = []
    print("access1")
    pipeline = SummarizationPipeline(
    model=AutoModelWithLMHead.from_pretrained("SEBIS/code_trans_t5_base_code_documentation_generation_python"),
    tokenizer=AutoTokenizer.from_pretrained("SEBIS/code_trans_t5_base_code_documentation_generation_python", skip_special_tokens=True),
    device=-1)
    payload_json = request.get_json(force=True)
    print("access")
    for code in payload_json:
        if code['cellType'] == "code":
            print(code['source'])
            i = 0
            cell_id = int(code['cellID'].split('-')[-1])
            if code['source'].find("#") != -1:
                cell_list = code['source'].split("\n")
                for tmp in cell_list:
                    if tmp.find("#") != -1:
                        bp = dict()
                        bp['cellID'] = code['cellID']
                        bp['bullet_ID'] = 'b-' + str(cell_id) + '-' + str(i)
                        bp['bullet'] = re_0001_.sub(re_0002, tmp).lower().strip()
                        bp['type'] = "bullet_point"
                        bp['model'] = "notebook"
                        bp['weight'] = 10
                        bp['isChosen'] = True
                        bullet_points.append(bp)
                        i += 1
                        break
            bp = dict()
            bp['cellID'] = code['cellID']
            bp['bullet_ID'] = 'b-' + str(cell_id) + '-' + str(i)
            bp['bullet'] = pipeline(code['source'])[0]['summary_text'].strip()
            bp['type'] = "bullet_point"
            bp['model'] = "code_trans_t5_base"
            bp['weight'] = 9
            bp['isChosen'] = True
            bullet_points.append(bp)
    print(bullet_points)
    # parse the payload_json to get your parameters:
    return jsonify(bullet_points)

@app.route('/submit_payload_title', methods=['POST'])
def submit_payload_title():
    payload_json = request.get_json(force=True)
    code_list = []
    title_list = []
    title_haconvgnn = {}
    title_t5 = {}
    for cell in payload_json:
        flag = False
        if cell['cellType'] == "markdown":
            cell_list = cell['source'].split("\n")
            for tmp in cell_list:
                if tmp.find('#') != -1:
                    title_markdown = {}
                    title_markdown['title'] = re_0001_.sub(re_0002, tmp).lower().strip()
                    title_markdown['type'] = "markdown"
                    title_markdown['model'] = "notebook"
                    title_markdown['weight'] = 9
                    title_markdown['isChosen'] = True
                    title_list.append(title_markdown)
                    flag = True
                    break
            if flag == True:
                break
    for code in payload_json:
        if code['cellType'] == "code":
            code_list.append(code['source'])
	# call your main()/Interface function
    res = interface(code_list, config, model, asttok, tdatstok, comstok)
    res = res.replace("<NULL>", "")
    res = res.replace("<s>", "")
    res = res.replace("</s>", "")
    title_haconvgnn['title'] = res.strip()
    title_haconvgnn['type'] = "markdown"
    title_haconvgnn['model'] = "HAConvGNN"
    title_haconvgnn['weight'] = 9
    title_haconvgnn['isChosen'] = True
    title_list.append(title_haconvgnn)
    pipeline = SummarizationPipeline(
        tokenizer = AutoTokenizer.from_pretrained(
            "t5/",
            cache_dir="output/",
            return_dict=True
        ),
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "t5/"),
            device=-1
    )
    t5_title = pipeline("\n".join(code_list))
    title_t5['title'] = t5_title[0]['summary_text']
    title_t5['type'] = "markdown"
    title_t5['model'] = "T5-Base"
    title_t5['weight'] = 8
    title_t5['isChosen'] = True
    title_list.append(title_t5)
    # make sure that result is in dictionary format
    result = jsonify(title_list)
    return result

@app.route('/submit_payload_relevance', methods=['POST'])
def submit_payload_relevance():
    payload_json = request.get_json(force=True)
    response = backend_api.get_relevent_cells(payload_json)
    return jsonify(response)

@app.route('/submit_payload_layout', methods=['POST'])
def submit_payload_layout():
    payload_json = request.get_json(force=True)
    res = backend_api.get_layout(payload_json)
    return jsonify(res)

if __name__ == "__main__":

	app.run(host='0.0.0.0', port=int(port))
