# eye_closure
疲劳闭眼检测


## dlib install
```bash
git clone https://github.com/davisking/dlib.git
```
```bash
cd dlib; mkdir build; cd build; cmake .. ; cmake --build .
```
## opencv install
```bash
sudo apt-get install libopencv-dev 
```
## model
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```
## runing
```bash
mkdir build && cd build

cmake .. && make
```
