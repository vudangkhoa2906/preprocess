C_preprocess: C_preprocess.cpp
	g++ -g `pkg-config --cflags opencv` $< `pkg-config --libs opencv` -o $@
clean:
	rm C_preprocess
