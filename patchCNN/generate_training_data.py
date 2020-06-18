import cv2
import numpy as np
import os

#size - in pixels, size of linear motion blur (1, 3, .. 25)
#angle - in degrees, direction of motion blur (0, 30, ..150)
def generate_kernel(size, angle):
	k = np.zeros((size, size), dtype=np.float32)
	k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
	centre = (size // 2  , size // 2 )
	rotate_mat = cv2.getRotationMatrix2D(centre, angle, 1.0) # 2 x 3: modified transformation matrix for rotation around centre
	k = cv2.warpAffine(k, rotate_mat, (size, size) )  
	k = k * ( 1.0 / np.sum(k) ) 
	return k

def apply_blurkernel(imgsharp, kernelsize, angle):
	kernel = generate_kernel(kernelsize, angle)
	blurred = cv2.filter2D(imgsharp, -1, kernel)
	return blurred

def generate_kernel_labels():
	sizes = []
	angles = []
	for i in range(13):
		sizes.append(2*i+1)
	for i in range(6):
		angles.append(i*30)

	allkernel_labels = []
	for i in range(len(angles)):
		for j in range(len(sizes)):
			if (i!=0 and j==0):
				pass
			else:
				label = (angles[i], sizes[j])
				allkernel_labels.append(label)

	labels = np.asarray(allkernel_labels)
	np.savetxt('allkernel_labels.csv', labels, delimiter=',')
	
	return allkernel_labels


if __name__ == '__main__':
	use_edges = True
	sharp_dir = 'training_data/pascal_voc2010/'
	patch_dir = 'training_data/patches/'
	sharp_img_list = os.listdir(sharp_dir)
	patchsize = 30
	kernel_labels = generate_kernel_labels()
	
	for k in range(73):
		print('Kernel ' + str(k))
		(angle, kernelsize) = kernel_labels[k]

		folder = patch_dir + str(k)
		if not os.path.exists(folder):
			os.makedirs(folder)
		for img in sharp_img_list:
			input_img = cv2.imread(sharp_dir + img)
			r, c = input_img.shape[:2]

			blurred = apply_blurkernel(input_img, kernelsize, angle)
			
			numpatch_horiz = c//patchsize
			numpatch_vert = r//patchsize

			if (use_edges):
				edges = cv2.Canny(input_img, 100, 200)

				for h in range(numpatch_horiz):
					for v in range(numpatch_vert):
						patch_filename = folder + '/' + img[:-4] + '_' + str(v) + '_' + str(h) + '.png'
						patch_edge = edges[v*patchsize: (v+1)*patchsize, h*patchsize: (h+1)*patchsize]/255
						if (np.sum(patch_edge) >= 10):
							patch = input_img[v*patchsize: (v+1)*patchsize, h*patchsize: (h+1)*patchsize, :]
							cv2.imwrite(patch_filename, patch) 
			
			else:
				for h in range(numpatch_horiz):
					for v in range(numpatch_vert):
						patch_filename = folder + '/' + img[:-4] + '_' + str(v) + '_' + str(h) + '.png'
						patch = blurred[v*patchsize: (v+1)*patchsize, h*patchsize: (h+1)*patchsize, :]
						cv2.imwrite(patch_filename, patch) 