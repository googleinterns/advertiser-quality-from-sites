from source.recursive_model.tree_lib import html_to_encoded_tree
import pytest


@pytest.mark.parametrize('html_content, max_depth, label, exp_result', [ 
	('<html> <title>A resturant</title> <h> <p>Menu</p> <h>Location</h> </html>', 4, 'food', '((A resturant)((Menu)(Location)))->food\n'),
	('<html> <title>A resturant</title> <h> <p>Menu</p> <h>Location</h> </h> </html>', 1, 'food', '(A resturant Menu Location)->food\n'),
	('<html> <title>Auto Zone</title> <h> <p>Menu</p> <h>Location</h> </h> </html>', 2, 'car', '((Auto Zone)(Menu Location))->car\n'),
])
def test_html_to_encoded_tree(html_content, max_depth, label, exp_result):
    assert html_to_encoded_tree(html_content, max_depth, label) == exp_result
    
   
