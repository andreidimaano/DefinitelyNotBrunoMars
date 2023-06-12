# DefinitelyNotBrunoMars
Voice DeepFake and Pose Transfer using GANs UCR CS228 Deep Learning Final Project

MaskGAN-VC
1. install pip3 install on requirements.txt
2. Dataset:
    - Create a folder in root directory called "training-data"
    - Upload these [4 folders](https://drive.google.com/drive/folders/1HuwZcXkjUETTXvENUrUyhUk0Z8KqlMR0?usp=sharing) into "training-data"
    - Create a folder in the root directory called "og-training-data"
    - Upload these [2 folders](https://drive.google.com/drive/folders/1AfjN0yvVaywEGZV0_qO2ltKP-4lDpeZc?usp=sharing) into "og-training-data"
    - run the "create_norm_spec_submission.ipynb"
3. Training:
    - Run "train_submission.ipynb". The notebook will output loss graphs and MCDs. It will also output wavs into "/code/outputs/generated_audio

For all Augmentations:
1. Run "visualizations_(just_plots)_submission.ipynb" to produce all augmentation data visualizations (blue audio signal graphs and log mel spectrogram data).

Dance GAN:
visualization file and dance code :
[drive files for visualization file](https://drive.google.com/drive/folders/125cfU69dP0IaqL6Xvl77wSKu4WtCz92j?usp=drive_link)

1. Run the notebook "run_Dance.ipynb"
    - Dataset: Need to have "DL Project"(linked above) in personal drive.
    - **Important**: you must stop the notebook where it tell you to stop (the notebook will instruct you). Then follow the instruction in the notebook
2. For the DanceGAN output figure, it will output the target to this directory: "DLproject/EverybodyDanceNow_reproduce_pytorch/data/source/images/[frame number].png" and the source to this directory: "DLproject/EverybodyDanceNow_reproduce_pytorch/results/target/test_latest/images/images/"
