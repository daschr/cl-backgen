#!/usr/bin/python3
import argparse
import numpy as np
import pyopencl as cl
import cv2 as cv
import sys

class BackgroundSubtractor:
    def __init__(self, weight=0.5, threshold=1.0, join_weight=30, platform=0, kernel_source="kernel.cl"):
        self.input=input
        self.weight=weight
        self.join_weight=join_weight
        self.threshold=threshold
        self.src=open(kernel_source, "r").read()
        
        platforms=cl.get_platforms()
        if len(platforms)<=platform:
            raise IndexError(f"Could not find platform {platform}!")
        
        self.device=platforms[platform].get_devices()
        
    def run(self, input, silent=True, deflicker=False, sidebyside=False, 
            output_img=None, output_vid=None, nframes=10):
        mf=cl.mem_flags
        ctx=cl.Context(self.device)
        cmd_queue=cl.CommandQueue(ctx)

        prg=cl.Program(ctx, self.src).build()
        
        capture=cv.VideoCapture(input)
        

        if capture.isOpened():
            fimg=capture.read()[1]
            fimg=fimg.astype(np.float32)

            fps=capture.get(cv.CAP_PROP_FPS)
            vidout=None
            if output_vid is not None:
                vidout=cv.VideoWriter(output_vid,
                                    cv.VideoWriter_fourcc(*"mp4v"),
                                    fps, fimg.shape[:2][::-1])

            weight=cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR|mf.HOST_NO_ACCESS, hostbuf=np.float32(self.weight*fps))
            threshold=cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR|mf.HOST_NO_ACCESS, hostbuf=np.float32(self.threshold))
            jweight=cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR|mf.HOST_NO_ACCESS, hostbuf=np.int32(self.join_weight))

            width=cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR|mf.HOST_NO_ACCESS, hostbuf=np.int32(fimg.shape[1]))
            height=cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR|mf.HOST_NO_ACCESS, hostbuf=np.int32(fimg.shape[0]))
    
            histogram,img_histogram, lut=(None, None, None)
            if deflicker:
                histogram=cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR|mf.HOST_NO_ACCESS, hostbuf=np.zeros(256*3).astype(np.int32))
                img_histogram=cl.Buffer(ctx, mf.WRITE_ONLY|mf.COPY_HOST_PTR, hostbuf=np.zeros(256*3).astype(np.int32))
                lut=cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR|mf.HOST_NO_ACCESS, hostbuf=np.zeros(256*3).astype(np.int32))

            img=cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=fimg)
            background=cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=fimg)

            _,nimg=capture.read()
            new_img=cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=nimg.astype(np.float32))
            
            if deflicker:
                prg.cal_histogram(cmd_queue, fimg.shape[:2], None, histogram, img, width, height)
                prg.fin_histogram(cmd_queue, (3, ), None, histogram) 
             
            res=np.empty_like(fimg).astype(np.float32)
   
            try:
                while True:
                    if deflicker:
                        cl.enqueue_fill_buffer(cmd_queue, img_histogram, np.int32(0), 0, 3*4*256) 
                        prg.cal_histogram(cmd_queue, fimg.shape[:2], None, img_histogram, new_img, width, height)
                        prg.fin_histogram(cmd_queue, (3, ), None, img_histogram)
  
                        prg.cal_lut(cmd_queue, (3, ), None, histogram, img_histogram, lut)
                        prg.deflicker(cmd_queue, fimg.shape[:2], None, lut, new_img, width, height)
                        
                        prg.join_histogram(cmd_queue, (3, ), None, histogram, img_histogram, jweight)


                    prg.backsub(cmd_queue, fimg.shape[:2], None, img, background, new_img, width, height, weight, threshold)
            
                    if (not silent) or (vidout is not None):
                        cl.enqueue_copy(cmd_queue, res, background)
                    
                    if vidout is not None:
                        vidout.write(res.astype(np.uint8))
                    
                    if not silent:     
                        cv.imshow('background', res.astype(np.uint8))
                        if sidebyside: cv.imshow('real', nimg)
                        if cv.waitKey(1) == 27: break
                 
                    _,nimg=capture.read()
                    if nimg is None: break

                    cl.enqueue_copy(cmd_queue, new_img, nimg.astype(np.float32))
            except IndexError:
                pass
            
            cl.enqueue_copy(cmd_queue, res, background)
            if output_img is not None:
                cv.imwrite(output_img, res.astype(np.uint8))        
            
            if vidout is not None:
                vidout.write(res.astype(np.uint8))
            
            if not silent: 
                cv.imshow('background', res.astype(np.uint8))
                cv.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This program generates a static background image from videos using '+
                                                "an opencl kernel which calculates pixel values based on averages.")
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default=None)
    parser.add_argument('--output', type=str, help='Save the last frame to this path', default=None)
    parser.add_argument('--vidoutput', type=str, help='Write video to path', default=None)
    parser.add_argument('--weight', type=float, help='the weight which a new image gets merged into (0;inf]', default=0.5)
    parser.add_argument('--threshold', type=int, help='threshold which an pixel is seen as changed [1;255]', default=1)
    parser.add_argument('--silent', nargs='?', const=True, help='silent mode, do not show images', default=False)
    parser.add_argument('--sidebyside', nargs='?', const=True, help='show the generated background and the newest frame side by side', default=False)
    parser.add_argument('--deflicker', nargs='?', const=False, help='deflickers frames before processing them with backsub', default=False)
    args = parser.parse_args()

    subtractor=BackgroundSubtractor(weight=args.weight, threshold=args.threshold)

    if args.input is None:
        parser.print_help()
        sys.exit(1)

    subtractor.run(args.input, silent=args.silent, deflicker=args.deflicker, sidebyside=args.sidebyside, output_img=args.output, output_vid=args.vidoutput)
