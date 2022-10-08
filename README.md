### SETUP
### When running some scripts for training, transformation, etc. with argparse, create scripts which
### will run all needed combination of these scripts
1. INSTALL requirements.txt file
2. RUN conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
3. RUN run.sh script
4. RUN testing/test_data_loading.py
5. RUN src/preprocess/sample_df.py
6. RUN src/preprocess/data_cleaning.py
7. RUN src/embeddings/custom_embedding.py

8. RUN src/utils/embedding_downloader.py