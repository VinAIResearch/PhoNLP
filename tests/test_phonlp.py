# -*- coding: utf-8 -*-
import phonlp

def test_phonlp():
    text = "bầu_trời hôm_nay thật trong_xanh , bát_ngát và nhiều chim"
    phonlp.download('./')
    model = phonlp.load("./")
    out = model.annotate(text)
    model.print_out(out)
    model.annotate(input_file='/home/vinai/Documents/test_pho/input.txt',
                   output_file='/home/vinai/Documents/test_pho/output.txt')


if __name__ == '__main__':
    test_phonlp()