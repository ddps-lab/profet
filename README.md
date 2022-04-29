# PROFET
PROFET: PROFiling-based CNN Training Latency prohET for GPU Cloud Instances

[Demo Page](https://ddps-lab.github.io/profet/demo/index.html)

### Run Docker
```
sudo snap install docker
sudo docker pull tensorflow/tensorflow:2.5.0
sudo docker run -it tensorflow/tensorflow:2.5.0 bash
```

### Setting
```
apt-get update
apt-get install git -y
cd home
git clone https://github.com/anonymous-profet/profet.git
cd profet
pip install -r requirements.txt
```

### Data Preprocessing
```
cd data
python anchor_preprocessing.py
```

### Profet Inference
```
cd ../profet
python train_anchor_model.py --anchor_instance g3s.xlarge
python train_scaler_model.py
python anchor_prediction.py --filename 'vgg16_224ds_16bs_test.json' --anchor_instance 'g3s.xlarge' --anchor_latency 323
python scaler_prediction.py --target_instance g3s.xlarge --latency_min 10 --latency_max 100 --size_pred 128 --batch_or_dataset batchsize
```

### Profet Validation
```
cd ../validation
python anchor_validaiton.py --anchor_instance g3s.xlarge
python train_scaler.py
python scaler_validation.py --target_instance g3s.xlarge
```
