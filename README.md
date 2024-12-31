- [Introduction](#introduction)
- [Usage example: Command lines](#command)
- [Usage example: Python API](#api)
- [Pre-trained PhoNLP model](#pretrained)

<p align="center">	
<img width="100" alt="logo" src="https://user-images.githubusercontent.com/2412555/106093264-85897700-6162-11eb-9777-e2068d4442f2.png">
</p>


# <a name="introduction"></a> PhoNLP: A BERT-based multi-task learning model for part-of-speech tagging, named entity recognition and dependency parsing

PhoNLP is a multi-task learning model for joint part-of-speech (POS) tagging, named entity recognition (NER) and dependency parsing. Experiments on Vietnamese benchmark datasets show that PhoNLP produces state-of-the-art results, outperforming a single-task learning approach that fine-tunes the pre-trained Vietnamese language model [PhoBERT](https://github.com/VinAIResearch/PhoBERT) for each task independently.

_Although we evaluate PhoNLP on Vietnamese, our usage examples below can directly work for other languages that have gold annotated corpora available for the three tasks of POS tagging, NER and dependency parsing, and a pre-trained BERT-based language model available from [transformers](https://huggingface.co/models) (e.g. BERT, mBERT, RoBERTa, XLM-RoBERTa)._

<p align="center">	
<img width="600" alt="logo" src="https://user-images.githubusercontent.com/2412555/106093259-83271d00-6162-11eb-8fd6-93dbf4569aea.png">
</p>

Details of the PhoNLP model architecture and experimental results can be found in our [following paper](https://aclanthology.org/2021.naacl-demos.1/):

    @inproceedings{phonlp,
    title     = {{PhoNLP: A joint multi-task learning model for Vietnamese part-of-speech tagging, named entity recognition and dependency parsing}},
    author    = {Linh The Nguyen and Dat Quoc Nguyen},
    booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Demonstrations},
    pages     = {1--7},
    year      = {2021}
    }


**Please CITE** our paper when PhoNLP is used to help produce published results or incorporated into other software.

## Installation

- Python version >= 3.6; PyTorch version >= 1.4.0
- PhoNLP can be installed using `pip` as follows: `pip3 install phonlp`
- Or PhoNLP can also be installed from source with the following commands: 
	```
	git clone https://github.com/VinAIResearch/PhoNLP
	cd PhoNLP
	pip3 install -e .
	```

## <a name="command"></a> Usage example: Command lines

To play with the examples using command lines, please install `phonlp` from the source:	
```
git clone https://github.com/VinAIResearch/PhoNLP
cd PhoNLP
pip3 install -e . 
```

### Training

```
cd phonlp/models
python3 run_phonlp.py --mode train --save_dir <model_folder_path> \
	--pretrained_lm <transformers_pretrained_model> \
	--lr <float_value> --batch_size <int_value> --num_epoch <int_value> \
	--lambda_pos <float_value> --lambda_ner <float_value> --lambda_dep <float_value> \
	--train_file_pos <path_to_training_file_pos> --eval_file_pos <path_to_validation_file_pos> \
	--train_file_ner <path_to_training_file_ner> --eval_file_ner <path_to_validation_file_ner> \
	--train_file_dep <path_to_training_file_dep> --eval_file_dep <path_to_validation_file_dep>
```

`--lambda_pos`, `--lambda_ner` and  `--lambda_dep` represent mixture weights associated with POS tagging, NER and dependency parsing losses, respectively, and `lambda_pos + lambda_ner + lambda_dep = 1`.

Example:

```
cd phonlp/models
python3 run_phonlp.py --mode train --save_dir ./phonlp_tmp \
	--pretrained_lm "vinai/phobert-base" \
	--lr 1e-5 --batch_size 32 --num_epoch 40 \
	--lambda_pos 0.4 --lambda_ner 0.2 --lambda_dep 0.4 \
	--train_file_pos ../sample_data/pos_train.txt --eval_file_pos ../sample_data/pos_valid.txt \
	--train_file_ner ../sample_data/ner_train.txt --eval_file_ner ../sample_data/ner_valid.txt \
	--train_file_dep ../sample_data/dep_train.conll --eval_file_dep ../sample_data/dep_valid.conll
```

### Evaluation

```
cd phonlp/models
python3 run_phonlp.py --mode eval --save_dir <model_folder_path> \
	--batch_size <int_value> \
	--eval_file_pos <path_to_test_file_pos> \
	--eval_file_ner <path_to_test_file_ner> \
	--eval_file_dep <path_to_test_file_dep> 
```

Example:

```
cd phonlp/models
python3 run_phonlp.py --mode eval --save_dir ./phonlp_tmp \
	--batch_size 8 \
	--eval_file_pos ../sample_data/pos_test.txt \
	--eval_file_ner ../sample_data/ner_test.txt \
	--eval_file_dep ../sample_data/dep_test.conll 
```


### Annotate a corpus

```
cd phonlp/models
python3 run_phonlp.py --mode annotate --save_dir <model_folder_path> \
	--batch_size <int_value> \
	--input_file <path_to_input_file> \
	--output_file <path_to_output_file> 
```

Example:

```
cd phonlp/models
python3 run_phonlp.py --mode annotate --save_dir ./phonlp_tmp \
	--batch_size 8 \
	--input_file ../sample_data/input.txt \
	--output_file ../sample_data/output.txt 
```


## <a name="api"></a> Usage example: Python API

```python
import phonlp

# Load the trained PhoNLP model
model = phonlp.load(save_dir='/absolute/path/to/phonlp_tmp')

# Annotate a corpus where each line represents a word-segmented sentence
model.annotate(input_file='/absolute/path/to/input.txt', output_file='/absolute/path/to/output.txt')

# Annotate a word-segmented sentence
model.print_out(model.annotate(text="Tôi đang làm_việc tại VinAI ."))
```

By default, the output for each input sentence is formatted with 6 columns representing word index, word form, POS tag, NER label, head index of the current word and its dependency relation type:

```
1	Tôi	P	O	3	sub	
2	đang	R	O	3	adv
3	làm_việc	V	O	0	root
4	tại	E	O	3	loc
5	VinAI	Np 	B-ORG	4	prob
6	.	CH	O	3	punct
```

The output can be formatted following the 10-column CoNLL format where the last column is used to represent NER predictions. This can be done by adding `output_type='conll'` into the `model.annotate()` function. 

Also, in the `model.annotate()` function, the value of the parameter `batch_size` can be adjusted to fit your computer's memory instead of using the default one at 1  (`batch_size=1`). Here, a larger `batch_size` would lead to a faster performance speed.

## <a name="pretrained"></a> Pre-trained PhoNLP model for Vietnamese

- The pre-trained PhoNLP model for Vietnamese can be manually downloaded from [HERE](https://drive.google.com/file/d/1wfVGVsKkg7sH8R_3ECRoQ46DVOAmsNkQ/view?usp=sharing).
- Or it can be downloaded as follows:

```python
import phonlp

# Automatically download the pre-trained PhoNLP model for Vietnamese
# and save it in a local machine folder
phonlp.download(save_dir='/absolute/path/to/pretrained_phonlp')

# Load the pre-trained PhoNLP model for Vietnamese
model = phonlp.load(save_dir='/absolute/path/to/pretrained_phonlp')

# Annotate a corpus where each line represents a word-segmented sentence
model.annotate(input_file='/absolute/path/to/input.txt', output_file='/absolute/path/to/output.txt')

# Annotate a word-segmented sentence
model.print_out(model.annotate(text="Tôi đang làm_việc tại VinAI ."))
```

### Using VnCoreNLP to perform word and sentence segmentation on raw Vietnamese texts 

In case the input Vietnamese texts are `raw`, i.e. without word and sentence segmentation, a word segmenter must be applied to produce word-segmented sentences before feeding to the pre-trained PhoNLP model for Vietnamese. Users should use [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) to perform word and sentence segmentation (as it produces the same Vietnamese tone normalization that was applied to the data of the POS tagging, NER and dependency parsing tasks). 

#### Installation

	pip3 install py_vncorenlp

#### Example usage

```python
import py_vncorenlp

# Automatically download VnCoreNLP components from the original repository
# and save them in some local machine folder
py_vncorenlp.download_model(save_dir='/absolute/path/to/vncorenlp')

# Load VnCoreNLP for word and sentence segmentation
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/absolute/path/to/vncorenlp')

# Perform word and sentence segmentation 
print(rdrsegmenter.word_segment("Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."))
# ['Ông Nguyễn_Khắc_Chúc đang làm_việc tại Đại_học Quốc_gia Hà_Nội .', 'Bà Lan , vợ ông Chúc , cũng làm_việc tại đây .']
```
