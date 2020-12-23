# PhoNLP: A joint multi-task learning model for Vietnamese part-of-speech tagging, named entity recognition and dependency parsing

[comment]: <> (> Short blurb about what your product does.)

[comment]: <> ([![NPM Version][npm-image]][npm-url])

[comment]: <> ([![Build Status][travis-image]][travis-url])

[comment]: <> ([![Downloads Stats][npm-downloads]][npm-url])

[comment]: <> (One to two paragraph statement about your product and what it does.)

![](header.png)

## Installation

PhoNLP supports Python 3.6 or later. You can install PhoNLP via pip by using the command line: 
```sh
pip install phonlp
```
You can also install from source by using the below command lines:

```sh
git clone https://github.com/VinAIResearch/PhoNLP
cd PhoNLP
pip install -e .
```

## Usage example

```python
import phonlp
# Download the pre-trained PhoNLP model
# and save it in a local machine folder
phonlp.download(save_dir='./phonlp')
# Load the pre-trained PhoNLP model
model = phonlp.load(save_dir='./phonlp')
# Annotate a corpus
model.annotate(input_file='input.txt', output_file='output.txt')
# Annotate a sentence
model.print_out(model.annotate(text="Tôi đang làm_việc tại VinAI ."))
```
This command will print out the results for input sentence follow by Universal Dependencies parse of that sentence. The output should look like:
```sh
1	Tôi	P	O	3	sub	

2	đang	R	O	3	adv

3	làm_việc	V	O	0	root

4	tại	E	O	3	loc

5	VinAI	Np 	B-ORG	4	prob

6	.	CH	O	3	punct
```
Input file includes word segmented text and follows by the below format:
```sh
" Bệnh bệnh 623 " , nữ , 83 tuổi , trú ở phường Điện_Nam_Trung , thị_xã Điện_Bàn . 
" Bệnh_nhân 1.000 " , tại Khánh_Hoà , nam , 33 tuổi , là chuyên_gia , quốc_tịch Philippines . 
```
Output file format will similar the above output.

Additionally, you can also obtain the output following by CoNLL format. To obtain the CoNLL format output, you just need add parameter `output_type='conll'` in model.annotate() function. Also, depending on your computer's memory, you can also set parameter `batch_size=batch_size` you want when you annotate corpus to increase the speed of annotation.

[comment]: <> (_For more examples and usage, please refer to the [Wiki][wiki]._)

## Training and Evaluating Model

You can use the below commands to train model:

```sh
git clone https://github.com/VinAIResearch/PhoNLP
cd PhoNLP/phonlp/models
python3 run_phonlp.py --save_dir model_folder_path --train_file_dep path_to_dep_training_file --eval_file_dep path_to_dep_validation_file --train_file_pos path_to_pos_training_file --eval_file_pos path_to_pos_validation_file --train_file_ner path_to_ner_training_file --eval_file_ner path_to_ner_validation_file
```

And testing model:

```sh
python3 run_phonlp.py --save_dir model_folder_path --mode predict --eval_file_dep path_to_dep_test_file --eval_file_pos path_to_pos_test_file --eval_file_ner path_to_ner_test_file
```
Data format for dependency parsing task follows by CoNLL format.

Data format for NER and POS follow by the below format :

```sh
Tôi	P

đang	R

làm_việc	V

tại	E

VinAI	Np

.	CH
```

You can also annotate corpus from raw text by using the command:

```sh
python3 run_phonlp.py --save_dir model_folder_path --mode annotate --input_file path_to_input_file --output_file path_to_output_file
```


## License

	MIT License

	Copyright (c) 2020 VinAI Research

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.
