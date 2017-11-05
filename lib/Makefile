all:
	cd nms/src/cuda; \
	pwd;\
	echo "Compiling nms kernels by nvcc..."; \
	nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52; \
	cd ../..;\
	python build.py;\
	cd ..;\

	cd layer_utils/roi_pooling/src/cuda; \
	pwd; \
	echo "Compiling roi_pooling kernels by nvcc..."; \
	nvcc -c -o roi_pooling_kernel.cu.o roi_pooling_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52; \
	cd ../..;\
	python build.py;\
	cd ../../..

# clean:
# 	rm -rf */*.pyc
# 	rm -rf */*.so
