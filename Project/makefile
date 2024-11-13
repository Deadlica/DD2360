cpu: CPU/main.cpp
	g++ -O3 -o raytracer CPU/main.cpp

run-cpu: cpu
	./raytracer > image.ppm

clean-cpu:
	rm raytracer

.PHONY: cpu run-cpu clean-cpu