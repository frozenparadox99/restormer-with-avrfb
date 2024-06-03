
# Project Setup Instructions

Follow the steps below to set up the project environment and run the necessary files.

## Step 1: Create a Conda Environment

Create a new Conda environment with Python 3.10.12 using the conda-forge channel:

```sh
conda create --name py310 python=3.10.12 -c conda-forge
```

## Step 2: Activate the Conda Environment

Activate the newly created environment:

```sh
conda activate py310
```

## Step 3: Install Required Packages

Install the required packages from the \`requirements.txt\` file:

```sh
pip install -r requirements.txt
```

## Step 4: Extract Files

Run the script to download and extract the necessary files:

```sh
python extract_files.py
```

If the script doesn't work, follow these manual steps:
1. Go to [this Google Drive link](https://drive.google.com/file/d/1Y35DS9atsaE_ZH1mVnSApVEXVMQQS6jg/view?usp=sharing).
2. Download the zip file.
3. Place the zip file in the root directory of this repository.
4. Extract the contents into a folder called \`Restormer\` in the root directory.

## Step 5: Install PyTorch

Go to [pytorch.org](https://pytorch.org) to get the appropriate installation command for your system based on the CUDA version and your preference. Here are some common options:

### Option A: Install with pip (CUDA 11.8)

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Option B: Install with pip (CUDA 12.1)

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Option C: Install with conda (CUDA 11.8)

```sh
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Option D: Install with conda (CUDA 12.1)

```sh
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Step 6: Run the Main Script

Finally, run the main script:

```sh
python main.py
```


