__kernel void backsub(	__global float *mask, __global float *background, __constant float *new_img,
						__constant int *width, __constant int *height, __constant float *weight,
						__constant float *threshold){

	int2 pos=(int2) (get_global_id(1), get_global_id(0));
	int2 size=(int2) (*width, *height);
	if(any(pos>size))
		return;

	int i= (pos.y*size.x + pos.x)*3;
	int3 c= (int3) (i, i+1, i+2);

	float w=*weight;
	float3 c_mask=(float3) (mask[c.x], mask[c.y], mask[c.z]);
	float3 c_new=(float3) (new_img[c.x], new_img[c.y], new_img[c.z]);
	
	// merge new image into the background if difference <= threshold
	int r= all(fabs(c_mask-c_new) <= *threshold);
	if(r){
		background[c.x]=(background[c.x]+c_new.x)/2.0f;
		background[c.y]=(background[c.y]+c_new.y)/2.0f;
		background[c.z]=(background[c.z]+c_new.z)/2.0f;
	}
	// merge new image into our mask using weighted addition
	mask[c.x]=((w-1.0f)*c_mask.x+c_new.x)/w;
	mask[c.y]=((w-1.0f)*c_mask.y+c_new.y)/w;
	mask[c.z]=((w-1.0f)*c_mask.z+c_new.z)/w;
}


__kernel void join_histogram(__global int *hist1, __constant int *hist2, __constant int *weight){
	int offset=get_global_id(0);
	for(int i=0;i<256;++i)
		hist1[i*3+offset]=((*weight-1)*hist1[i*3+offset] + hist2[i*3+offset])/(*weight);
}


__kernel void cal_histogram(	__global int *histogram, __constant float *img, 
								__constant int *width, __constant int *height){
	int2 pos=(int2) (get_global_id(1), get_global_id(0));
	int2 size=(int2) (*width, *height);
	
	if(any(pos>size))
		return;

	int i=(pos.y*size.x+pos.x)*3;
	int3 color=(int3) (img[i], img[i+1], img[i+2]);
	atomic_inc(histogram+color.x*3);
	atomic_inc(histogram+color.y*3+1);
	atomic_inc(histogram+color.z*3+2);
}


__kernel void fin_histogram( __global int *histogram ){
	int offset=get_global_id(0);
	for(int i=1;i<256;++i)
		histogram[i*3+offset]+=histogram[i*3-3+offset];
}


__kernel void cal_lut(	__constant int *histogram, __constant int *img_histogram, __global int *lut){
	int offset=get_global_id(0);
	float ratio=(float) (img_histogram[255*3+offset])/(float)(histogram[255*3+offset]);
	for(int c=0;c<256;++c){
		int rs=0;
		while((int) ((float) (histogram[rs*3+offset]*ratio)) < img_histogram[c*3+offset])
			++rs;

		lut[c*3+offset]=rs;
	}	
}


__kernel void deflicker( 	__constant int *lut, __global float *img, 
							__constant int *width, __constant int *height){
	int2 pos=(int2) (get_global_id(1), get_global_id(0));
	int2 size=(int2) (*width, *height);
	
	if(any(pos>size))
		return;
	
	int i=(pos.y*size.x+pos.x)*3;
	int3 color=(int3) (img[i], img[i+1], img[i+2]);
	
	img[i]=lut[color.x*3];
	img[i+1]=lut[color.y*3+1];
	img[i+2]=lut[color.z*3+2];
}
