import sys, os

from source.utils.utils import clean_text
import pytest


@pytest.mark.parametrize('txt, exp_result', [ 
	('\r vahid \r \t\t\t\\t    sanei was an intern at \n Google\t\t', 'vahid sanei was an intern at Google'),
	('\t\t\t Thanks YY and Xiang for helping \r                  me during this         internship! \t \n \r', 'Thanks YY and Xiang for helping me during this internship!')
])
def test_clean_text(txt, exp_result):
    assert clean_text(txt) == exp_result
    
   

