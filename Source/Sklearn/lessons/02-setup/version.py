from platform import python_version

print python_version( )

# To install these libraries:
# $: sudo easy_install pip

# $: sudo pip install scipy
# $: sudo pip install numpy
# $: sudo pip install pandas
# $: sudo pip install scikit-learn


# scipy
import scipy

print('scipy: {}'.format( scipy.__version__ ))

# numpy
import numpy

print('numpy: {}'.format( numpy.__version__ ))

# matplotlib
import matplotlib

print('matplotlib: {}'.format( matplotlib.__version__ ))

# pandas
import pandas

print('pandas: {}'.format( pandas.__version__ ))

# scikit-learn
import sklearn

print('sklearn: {}'.format( sklearn.__version__ ))
