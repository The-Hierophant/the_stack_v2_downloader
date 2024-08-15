
# Download Script for the-stack-v2 Dataset
## Introduction
the-stack-v2 is the training data of starcoder v2. Whereas, the starcoder merely provides the metadata of its [training dataset](https://huggingface.co/datasets/bigcode/the-stack-v2).

This repository implements concurrent downloading and packaging of the downloaded files into Parquet datasets, based on [huangyangyu/starcoder_data](https://github.com/huangyangyu/starcoder_data).

## Usage
You can use the following command line to download the dataset. Set your Hugging Face access token through the --hug_access_token parameter. Ensure the token has read permissions for the the-stack-v2 dataset. Fine-tune the max_workers parameter according to your environment.

```bash
python -m venv venv
source venv/bin/activate # activate th env based on your system
pip install boto3 botocore smart_open datasets tqdm pandas pyarrow

python download_the_stack_v2.py \
  --hug_access_token {your_huggingface_access_token} \
  --language Python \
  --download_folder {output_parquet_dir} \
  --max_workers 256
```

## Note
This script is still in testing; it can download the Python subset (~1TB, 47272886 files) in approximately 10 hours.
