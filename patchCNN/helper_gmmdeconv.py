import numpy as np 
import scipy.io
import cv2
from itertools import product

#define params
patchSize = 8


def loggausspdf(X, sigma):
	'''
	log pdf of Gaussian with zero mean
	Inputs:
	   X - patches (in columns)
	   sigma - 64x64 cov matrix (associated with a mixture weight)
	
	Outputs:
	   y - vector containing log probabilities of each patch belonging to GMM mixture component assoc with input covariance matrix sigma
	'''

	d = X.shape[0] #64, dimension of each 8x8 patch
	R = np.linalg.cholesky(sigma).T #upper triangular matrix returned by cholesky decomposition of positive definite matrix sigma, sigma = R*R.H

	lstsqsol = np.linalg.lstsq(R.T, X, rcond=None)[0]
	q = np.sum(lstsqsol**2, 0) #quadratic term, distance of each patch from GMM mixture component

	c = d*np.log(2*np.pi) + 2*np.sum(np.log(np.diagonal(R))) 

	#compute log pdf
	y = -(c+q)/2 

	return y



def aprxMAPGMM(Y,patchSize, noiseSD, gmmfilename = "GSModel_8x8_200_2M_noDC_zeromean.mat"):
	'''
	approximate GMM MAP estimation - a single iteration of the "hard version"
	EM MAP procedure (see paper for a reference)

	Inputs:
	   Y - the noisy patches (in columns)
	   noiseSD - noise standard deviation
	   gmmfilename - the mat file containing params of trained GMM model
	
	Outputs:
	   Xhat - the restored patches
'''

	#load GMM params from mat file
	mat = scipy.io.loadmat(gmmfilename) #params of trained GMM model on 8x8 patches, zero mean
	nmodels = int(mat['GS']['nmodels']) #200
	covs = mat['GS']['covs'][0][0] # (64, 64, 200) - each mixture component associated with 64x64 covariance matrix
	mixweights = mat['GS']['mixweights'][0][0] # (200, 1) - each mixture component has a probability weight
	means = mat['GS']['means'][0][0] # (64, 200) - means are all zeros

	#general noise covariance matrix
	SigmaNoise = noiseSD**2 * np.identity(patchSize**2)

	#remove DC component
	meanY = np.mean(Y, axis=0) #mean for each column
	Y = Y - meanY

	#calculate assignment probabilities for each mixture component for all patches
	PYZ = np.zeros((nmodels, Y.shape[1]))
	for i in range(nmodels):
		covs[:,:,i] = covs[:,:,i] + SigmaNoise
		PYZ[i,:] = np.log(mixweights[i]) + loggausspdf(Y, covs[:,:,i])

	#find most likely GMM mixture component for each patch
	ks = np.argmax(PYZ, axis=0)

	#perform wiener filtering
	Xhat = np.zeros(Y.shape)
	for i in range(nmodels):
		inds = np.array(np.where(ks==i))
		if (inds.size !=0):
			A = covs[:,:,i] + SigmaNoise
			b = np.matmul(covs[:,:,i], Y[:,inds[0]]) + np.matmul(SigmaNoise, np.tile(means[:,i].reshape(-1,1), (1, len(inds[0])))) #second term is zero because means=0
			Xhat[:,inds[0]] = np.linalg.lstsq(A, b, rcond=None)[0]

	Xhat = Xhat + meanY

	return Xhat


def generate_kernel(size, angle):
	'''
	size - in pixels, size of linear motion blur (1, 3, .. 25)
	angle - in degrees, direction of motion blur (0, 30, ..150)
	'''
	k = np.zeros((size, size), dtype=np.float32)
	k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
	centre = (size // 2  , size // 2 )
	rotate_mat = cv2.getRotationMatrix2D(centre, angle, 1.0) # 2 x 3: modified transformation matrix for rotation around centre
	k = cv2.warpAffine(k, rotate_mat, (size, size) )  
	k = k * ( 1.0 / np.sum(k) ) 
	return k

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
	
	return allkernel_labels

def nonunif_convolution(kernel_labels, kernelInit, img_in): #quite slow... about 10s for 375x500
	'''
	Input Parameters:
	
	 kernel_labels: vector with index corresponding to kernel id and each element a tuple (ori, mag)
	 kernelInit:  2D matrix, convolution kernel ids for each pixel
	 img_in: grayscale image to be convolved with kernels


	Outputs:
	 img_out: convolved output
	'''
	if (len(img_in.shape)==3):
		row, col, dim = img_in.shape
	else:
		row, col = img_in.shape
		dim = 1
		img_in = img_in.reshape(row, col, dim)

	#init convolved output
	img_out = img_in

	for r in range(row):
		for c in range(col):
			#obtain kernel matrix for convolution
			ker_id = int(kernelInit[r,c])
			angle, size = kernel_labels[ker_id]
			ker = generate_kernel(size, angle)

			#get relevant surrounding pixels for convolution
			#rmin rmax cmin cmax are indices corresponding to full image, all inclusive
			#r_pix, c_pix are indices of central pixel corresponding to extracted patch
			half = size//2
			#row
			if (r<half):
				rmin = 0
				rmax = r + half
				r_pix = r
				ker = ker[half-r:, :]
			elif (r>row-1-half):
				rmin = r - half
				rmax = row-1
				r_pix = half
				ker = ker[:half+row-r, :]
			else:
				rmin = r - half
				rmax = r + half
				r_pix = half

			#col
			if (c<half):
				cmin = 0
				cmax = c + half
				c_pix = c
				ker = ker[:, half-c:]
			elif (c>col-1-half):
				cmin = c - half
				cmax = col-1
				c_pix = half
				ker = ker[:, :half+col-c]
			else:
				cmin = c - half
				cmax = c + half
				c_pix = half


			patch_surr_pix = img_in[rmin:rmax+1, cmin:cmax+1,:]

			#convolution
			r_k, c_k = ker.shape #patch_surr_pix should have same shape
			img_out[r,c,:] = np.tensordot(ker,patch_surr_pix, axes=((0,1),(0,1)))


	return img_out



def im2col(im, patchsize, stepsize=1):
	'''
	Inputs:
		im - grayscale img, 2D array
		patchsize - tuple containing patch dimensions (height, width)
		stepsize - extent of overlap of patches

	Outputs:
		col - patches in columns
	'''
	# Parameters
	M,N = im.shape
	col_extent = N - patchsize[1] + 1
	row_extent = M - patchsize[0] + 1

	# Get Starting block indices
	start_idx = np.arange(patchsize[0])[:,None]*N + np.arange(patchsize[1])

	# Get offsetted indices across the height and width of input array
	offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)

	# Get all actual indices & index into input array for final output
	col =  np.take (im,start_idx.ravel()[:,None] + offset_idx.ravel()[::stepsize])

	return col


if __name__ == "__main__":
	
########################################
	#test aprxMAPGMM
	patchSize = 8
	Y = np.random.rand(64,3)
	
	noiseSD = 2
	Xhat = aprxMAPGMM(Y,patchSize, noiseSD)
	Y = Y*255
	Xhat = Xhat*255
	I = Y.reshape((8,8,3))
	I2 = Xhat.reshape((8,8,3))
	
	cv2.imwrite('mapgmm_y.jpg', I)
	cv2.imwrite('mapgmm_x.jpg', I2)
    
#test non-unif convolution
	np.random.seed(0)
	im = cv2.imread('2008_002047.jpg')
	r,c,d = im.shape
	kernel_labels = generate_kernel_labels()
	kernels = np.ones((r,c))*9
	kernels[:r//2][:c//2] = 45
	kernels[r//2:][:c//2] = 33
	kernels[:r//2][c//2:] = 0

	im_blur = nonunif_convolution(kernel_labels, kernels, im)
	cv2.imwrite('nonunifblur.jpg', im_blur) 