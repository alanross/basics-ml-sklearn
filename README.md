# basics-ml-sklearn

Collection of examples getting started with ML utilizing SkLearn  

#### Python3
https://docs.python-guide.org/starting/install3/osx/

#### VirtualEnv

Setup
```
$: mkdir venv
$: cd venv
$: virtualenv -p python3 sklearn
$: source sklearn/bin/activate
$: pip install scipy
$: pip install numpy
$: pip install pandas
$: pip install scikit-learn
$: pip install matplotlib
$: pip install urllib3
```

Activate / Deactivate
```
$: cd to path sub parent folder of sklearn virtualenv
$: source sklearn/bin/activate
$: deactivate
```

#### Brew
If error running, make sure libusb is installed and a multi threaded openCV version
```
$: brew install libusb
$: brew install opencv --with-tbb --with-python --with-ffpmeg
```

#### Pip
Install xcode command line tools: $ xcode-select --install
Install brew			: $ ruby -e "$(curl -fsSL https://raw.github.com/Homebrew/homebrew/go/install)"
Install python in brew		: $ brew install python
Install pip			: $ sudo easy_install pip
Install virtual env		: $ pip install --upgrade virtualenv 
Create a new virtual envirnment : $ virtualenv --system-site-packages newVitualEnvDirectory # for Python 2.7
Go to virtual environment	: $ cd newVitualEnvDirectory
Activate this virtual env	: $ source ./bin/activate (this will activate newVitualEnvDirectory )
Install modules/packages	: (newVitualEnvDirectory)$ easy_install -U pip
Install modules/packages	: (newVitualEnvDirectory)$ pip install --upgrade tensorflow 
Install modules/packages	: (newVitualEnvDirectory)$ pip install --upgrade scipy pandas numpy scikit-learn

Note, There are issues in High Sierra using pip directly:
https://stackoverflow.com/questions/20877106/using-intellijidea-within-an-existing-virtualenv


#### PyCharm setup:
1. In the Settings/Preferences dialog (⌘,), select Project: <project name> | Project Interpreter.
2. Open "Project Interpreter" drop down, select "Show All..."
3. Click '+' in dialog box to add new virtual environment
4. In dialog box select "existing environment" with path to /pathTo/venv/sklearn/bin/python3.6