# handsome
Human's appearance related.

By zhouming402@163.com

### eye-rec

#### Dependecy

- OpenCV

#### How to Run

1. c++11 surpported compilier is required.

2. I make a successful compile on **ubuntu 16.04** and **mac os-X 10.10**. Windows should be OK.

3. under model directory, there is a face landmarker, a eye landmarker and a eye-status classfier.

	```
	git clone --recursive https://github.com/GragonShit/handsome.git
	cd eye-rec/cpp
	mkdir build && cd build
	cmake ..
	make
	./demo
	```

4. when the demo is running, 68 face landmarks with green and 12 eye pupil landmarks with red will show.
when u close ur eyes, corresponding pupil landmarks will disappear.

5. recognize() function output std::pair<double,int>, the first is probability and the second is label. You can make a threshold to control the final result.

### tongue-rec

#### Dependecy

- OpenCV

#### How to Run

1. c++11 surpported compilier is required.

2. I make a successful compile on **ubuntu 16.04** and **mac os-X 10.10**. Windows should be OK.

3. under model directory, there is a face landmarker and a tongue-status classfier.

	```
	git clone --recursive https://github.com/GragonShit/handsome.git
	cd tongue-rec/cpp
	mkdir build && cd build
	cmake ..
	make
	./demo ../../../model/shape_predictor_68_face_landmarks.dat ../../../model/Net-weights-tongue
	```

4. There will be 3 status, 0 for mouth close, 1 for tongue out, 2 for mouth open. 
The output of recognize function is std::pair<double,int>, where the first is probability and the second is label.

5. when the demo is running, 68 face landmarks with green will show.
when u put out ur tongue, corresponding lower lip landmarks will disappear.

6. This version is not stable, iterations will go on.


