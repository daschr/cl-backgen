#!/usr/bin/python3
import argparse
import numpy as np
import pyopencl as cl
import cv2 as cv
import sys

class Deflicker:
	def __init__(self, weight=20, platform=0, kernel_source="kernel.cl"):
		self.input=input
		self.weight=weight
		self.src=open(kernel_source, "r").read()
		
		platforms=cl.get_platforms()
		if len(platforms)<=platform:
			raise IndexError(f"Could not find platform {platform}!")
		
		self.device=platforms[platform].get_devices()
		
	def run(self, input, silent=True, sidebyside=False, output_img=None, output_vid=None):
		mf=cl.mem_flags
		ctx=cl.Context(self.device)
		cmd_queue=cl.CommandQueue(ctx)

		prg=cl.Program(ctx, self.src).build()
		
		capture=cv.VideoCapture(input)
		

		if capture.isOpened():
			_,nimg=capture.read()
			nimg=nimg.astype(np.float32)

			fps=capture.get(cv.CAP_PROP_FPS)
			vidout=None
			if output_vid is not None:
				vidout=cv.VideoWriter(output_vid,
									cv.VideoWriter_fourcc(*"mp4v"),
									fps, nimg.shape[:2][::-1])

			width=cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR|mf.HOST_NO_ACCESS, hostbuf=np.int32(nimg.shape[1]))
			height=cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR|mf.HOST_NO_ACCESS, hostbuf=np.int32(nimg.shape[0]))
			
			jweight=cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR|mf.HOST_NO_ACCESS, hostbuf=np.int32(self.weight))
	
			res=np.empty_like(nimg).astype(np.float32)
			
			img=cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=nimg)

			histogram=cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR|mf.HOST_NO_ACCESS, hostbuf=np.zeros(256*3).astype(np.int32))
			img_histogram=cl.Buffer(ctx, mf.WRITE_ONLY|mf.COPY_HOST_PTR, hostbuf=np.zeros(256*3).astype(np.int32))
			lut=cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=np.zeros(256*3).astype(np.int32))
		
			prg.cal_histogram(cmd_queue, nimg.shape[:2], None, histogram, img, width, height)
			prg.fin_histogram(cmd_queue, (3, ), None, histogram) 
			
			new_img=cl.Buffer(ctx, mf.WRITE_ONLY|mf.COPY_HOST_PTR, hostbuf=nimg.astype(np.float32))

			try:
				while True:
					cl.enqueue_fill_buffer(cmd_queue, img_histogram, np.int32(0), 0, 3*4*256) 
					prg.cal_histogram(cmd_queue, nimg.shape[:2], None, img_histogram, new_img, width, height)
					prg.fin_histogram(cmd_queue, (3, ), None, img_histogram) 
					
					prg.cal_lut(cmd_queue, (3, ), None, histogram, img_histogram, lut)
					prg.deflicker(cmd_queue, nimg.shape[:2], None, lut, new_img, width, height)
					
					prg.join_histogram(cmd_queue, (3, ), None, histogram, img_histogram, jweight)
				
					if (not silent) or (vidout is not None):
						cl.enqueue_copy(cmd_queue, res, new_img)
					
					if vidout is not None:
						vidout.write(res.astype(np.uint8))
					
					if not silent:	   
						cv.imshow('deflickered', res.astype(np.uint8))
						if sidebyside: cv.imshow('real', nimg)
						if cv.waitKey(1) == 27: break
						
					_,nimg=capture.read()
					if nimg is None: break
					cl.enqueue_copy(cmd_queue, new_img, nimg.astype(np.float32))
			except IndexError:
				pass
			
			cl.enqueue_copy(cmd_queue, res, new_img)
			if output_img is not None:
				cv.imwrite(output_img, res.astype(np.uint8))		
			
			if vidout is not None:
				vidout.write(res.astype(np.uint8))
			
			if not silent: 
				cv.imshow('deflickered', res.astype(np.uint8))
				cv.waitKey(0)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='This program deflcikers videos using histogram matching per color channel')
	parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default=None)
	parser.add_argument('--output', type=str, help='Save the last frame to this path', default=None)
	parser.add_argument('--vidoutput', type=str, help='Write video to path', default=None)
	parser.add_argument('--weight', type=float, help='the weight which a new image histogram gets merged into', default=20)
	parser.add_argument('--silent', nargs='?', const=True, help='silent mode, do not show images', default=False)
	parser.add_argument('--sidebyside', nargs='?', const=True, help='show the generated background and the newest frame side by side', default=False)
	args = parser.parse_args()

	deflicker=Deflicker(weight=args.weight)

	if args.input is None:
		parser.print_help()
		sys.exit(1)

	deflicker.run(args.input, silent=args.silent, sidebyside=args.sidebyside, output_img=args.output, output_vid=args.vidoutput)
