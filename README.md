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
	mkdir build
	cmake ..
	make
	./demo
	```

4. when the demo is running, 68 face landmarks with green and 12 eye pupil landmarks with red will show.
when u close ur eyes, corresponding pupil landmarks will disappear.

