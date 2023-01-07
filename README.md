# Speaker_change_detection

- To use same conda environment.
``` sh
    conda create --name bilstm_change_detection --file spec-file.txt
    source activate bilstm_change_detection
```

 ``` sh
    mkdir feature_storage
 ```
- Now, we can extract features. You can use "pyannote_based.txt" as a featureplan.
    ``` sh
        python feature_extraction.py root_dir featureplan

        python feature_extraction.py "./amicorpus/*/audio/" "pyannote_based.txt" {example usage}
    ```
    
- Now, we need to create the ground truth text files. For that, we need to download .mdtm files for AMI corpus. 
``` sh
wget https://raw.githubusercontent.com/pyannote/pyannote-db-odessa-ami/master/AMI/data/speaker_diarization/dev.mdtm

wget https://raw.githubusercontent.com/pyannote/pyannote-db-odessa-ami/master/AMI/data/speaker_diarization/trn.mdtm

wget https://raw.githubusercontent.com/pyannote/pyannote-db-odessa-ami/master/AMI/data/speaker_diarization/tst.mdtm
```

And create a empty folder to stores these .txt files.
``` sh
mkdir txt_files
``` 

 After that, we will run these command.
``` sh
python ground_truth_txt.py dev.mdtm
python ground_truth_txt.py trn.mdtm
python ground_truth_txt.py tst.mdtm
```

We need to delete some feature numpy files which has not corresponding txt files.
``` sh
python txt_checker.py ./feature_storage/ ./txt_files/
```

Now, we can train the system. 

``` sh
python train_model.py root_dir featureplan how_many_repeat how_many_step boost how_many_boost fuzzy epoch 

python train_model.py "./feature_storage/" "pyannote_based.txt" 10 30 True 120 True 2 {example usage}
```

Now, we can create the prediction.

``` sh
python create_prediction.py EN2001b "pyannote_based.txt" 0.7
```

Lastly, to convert from txt to internal metadata format.

``` sh
python metadata_converter.py input_directory output_directory --outputType=mpeg7 --inputType=txt_file

python metadata_converter.py testdata/ami_pred out/ami_mpeg7_pred --outputType=mpeg7 --inputType=txt_file {example usage}

```
