from PIL import Image
import numpy as np  
import matplotlib.pyplot as plt
import math

def swap(s1, s2):
    return s2, s1

def mag(x):
	return np.absolute(x[0])

# sort for complex numbers
def bubble_sort(x):
	sz = len(x)
	for i in range(sz):
		for j in range(0,sz-i-1):
			if mag(x[j]) > mag(x[j+1]): x[j], x[j+1] = swap(x[j],x[j+1])

	return x

def evd_helper(ldiagA,pairm,eigA,eigAin,A,k):
	t = 0
	r = 256-k
	for (x,y) in pairm:
		t+=1
		if t <= r : ldiagA[y] = 0			# Equating the values to 0 -> eigenvalues other than the top k

	diagA = np.diag(ldiagA)
	tfinal = np.dot(eigA,diagA)
	A_rec = np.dot(tfinal,eigAin)			# A_rec = V Sigma V-1 where V = eigA, V-1 = eigAin, Sigma = diagA
	A_rec = A_rec.real						# Since there should not be any imaginary part(occurs in negligible quatity in few cells)

	diff = np.subtract(A,A_rec)
	diff_square = np.square(diff)
	fnorm = math.sqrt(np.sum(diff_square)) 	# Finding the frobeius norm

	# # Saving the image
	# data = Image.fromarray((A_rec * 255).astype(np.uint32))
	# data.save("image.png")
	return fnorm

# Performs the eigen value decomposition of the matrix and returns an array of norm values for k value from (0 .. 256)
def evd():
	arr = []
	img = Image.open("31.jpg")			# Storing image input
	A = np.array(img)
	ldiagA,eigA = np.linalg.eig(A)		# Using eig function in linalg library to obtain the eigenvalues and eigenvectors
	eigAin = np.linalg.inv(eigA)

	pairm = []							# Stores the position of eigenvalues (used after sorting to make 0's)
	for i in range(len(ldiagA)):
		pairm.append((ldiagA[i],i))

	pairm = bubble_sort(pairm)			# Sorting eigenvalues
	temp = ldiagA.copy()				# Storing the ldiagA (being changed in the helper function - making few of them 0's)

	for i in range(257):				# Helper function gives the norm value for each k value
		arr.append(evd_helper(ldiagA,pairm,eigA,eigAin,A,i))
		ldiagA = temp.copy()

	return arr

def svd_helper(ldiagA,pairm,U,Vt,A,k):
	temp_ldiagA = ldiagA
	t = 0
	r = 256-k
	for (x,y) in pairm:
		t+=1
		if t <= r : temp_ldiagA[y] = 0		# Equating the values to 0 -> eigenvalues other than the top k

	diagA = np.diag(temp_ldiagA)
	tfinal = np.dot(U,diagA)
	A_rec = np.dot(tfinal,Vt)				# A_rec = U Sigma Vt

	diff = np.subtract(A,A_rec)
	diff_square = np.square(diff)
	fnorm = math.sqrt(np.sum(diff_square)) 	# Finding the frobeius norm

	# # Saving the image
	# data = Image.fromarray((A_rec * 255).astype(np.uint32))
	# data.save("imag.png")

	return fnorm

# Performs the singular value decomposition of the matrix and returns an array of norm values for k value from (0 .. 256)
def svd():
	img = Image.open("31.jpg")
	A = np.array(img)					# storing image input
	arr = []
	A = A.astype('float64')
	At = (A.transpose()).conjugate()
	dd,V = np.linalg.eig(np.dot(At,A)) 	# performing eigen value decomposition on (A_transpose A)
	ldiagA = np.sqrt(dd)				# square root of the eigenvalues obtained for (A_t A) -> Sigma 
	Vt = (V.transpose()).conjugate()
	tt = np.diag(ldiagA)
	diagin = np.linalg.inv(tt)
	U = np.dot(np.dot(A,V),diagin)      # U = A (Vt)-1 (Sigma)-1  -> (Using matrix operations on A = U Sigma Vt)

	pairm = []							# Stores the position of eigenvalues (used after sorting to make 0's)
	for i in range(len(ldiagA)):
		pairm.append((ldiagA[i],i))

	pairm = bubble_sort(pairm)			# Sorting eigenvalues

	for i in range(257):				# Helper function gives the norm value for each k value
		arr.append(evd_helper(ldiagA,pairm,U,Vt,A,i))
		ldiagA = np.sqrt(dd)

	return arr


# Getting the norm values from the respective functions and plotting the graph
evd_norm = evd()
svd_norm = svd()
x = []

for i in range(257):
	x.append(i)

plt.plot(x,evd_norm,label = "evd")
plt.plot(x,svd_norm,label = "svd")

plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()

plt.show()
plt.savefig('graph.png')