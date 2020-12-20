# PhoNLP: A joint multi-task learning model for Vietnamese part-of-speech tagging, named entity recognition and dependency parsing

[comment]: <> (> Short blurb about what your product does.)

[comment]: <> ([![NPM Version][npm-image]][npm-url])

[comment]: <> ([![Build Status][travis-image]][travis-url])

[comment]: <> ([![Downloads Stats][npm-downloads]][npm-url])

[comment]: <> (One to two paragraph statement about your product and what it does.)

![](header.png)

## Installation

OS X & Linux:

```sh
pip install phonlp
```

## Usage example

```python
import phonlp
# Download phonlp model
phonlp.download(path_save_model='./')

# Load model
model = phonlp.load_model(path_save_model='./')

# Input text must be already word-segmented!
line = "Tôi là sinh_viên trường đại_học Bách_khoa ."

# Annotate sentence
phonlp.annotate(model, text=line, type='sentence')

# 
phonlp.annotate(model, input_file='input.txt', output_file='output.txt', type='corpus')
```
This command will print out the results for input sentence follow by Universal Dependencies parse of that sentence. The output should look like:
```sh
1	Tôi	_	_	P	_	2	sub	_	O

2	là	_	_	V	_	0	root	_	O

3	sinh_viên	_	_	N	_	2	vmod	_	O

4	trường	_	_	N	_	3	nmod	_	B-ORG

5	đại_học	_	_	N	_	4	nmod	_	I-ORG

6	Bách_Khoa	_	_	Np	_	4	nmod	_	I-ORG

7	.	_	_	CH	_	2	punct	_	O
```
You can also use phonlp tool to annotate an input raw text corpus by using following command:
```python
phonlp.annotate(model, input_file='input.txt', output_file='output.txt', type='corpus')
```
Input file includes word segmented text and follows by the below format:
```sh
" Bệnh bệnh 623 " , nữ , 83 tuổi , trú ở phường Điện_Nam_Trung , thị_xã Điện_Bàn . 
" Bệnh_nhân 1.000 " , tại Khánh_Hoà , nam , 33 tuổi , là chuyên_gia , quốc_tịch Philippines . 
```
Output file format will similar the above output.


[comment]: <> (_For more examples and usage, please refer to the [Wiki][wiki]._)

## Training and Evaluating Model

You can use the below commands to train model:

```sh
python3 train_jointmodel_3task.py --save_dir path_save_model --train_file_dep path_to_dep_training_file --eval_file_dep path_to_dep_validation_file --train_file_pos path_to_pos_training_file --eval_file_pos path_to_pos_validation_file --train_file_ner path_to_ner_training_file --eval_file_ner path_to_ner_validation_file
```

And testing model:

```sh
python3 train_jointmodel_3task.py --save_dir path_save_model --mode predict --eval_file_dep path_to_dep_test_file --eval_file_pos path_to_pos_test_file --eval_file_ner path_to_ner_test_file
```
Data format for dependency parsing task follows by CoNLL-U format. You can see at https://universaldependencies.org/format.html .

Data format for NER and POS follow by the below format :

```sh
Tôi	P

là	V

sinh_viên	N

trường	N	

đại_học	N	

Bách_Khoa	Np

.	CH
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
