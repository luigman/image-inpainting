import os
import glob
import cv2
from scipy.ndimage import gaussian_filter1d
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from skimage import io, color

imgDir = 'images/'
saveDir = 'output/'
maskDir = 'masks/'

def computeNormals(contours):
    #input: contours shape: (numContours, numPts, 1, 2)
    #output: normals shape: (numContours, numPts, 1, 2)
    vis = False
    normals = []
    for contour in contours:
        normal_i = []
        for i in range(0,np.array(contour).shape[0],1):
            contour = np.array(contour)
            #contours = gaussian_filter1d(contours, 1, axis = 1)
            pt1 = contour[i,0]
            pt1 = np.array([pt1[0], pt1[1], 0])
            pt2 = contour[(i+1)%len(contour),0]
            pt2 = np.array([pt2[0], pt2[1], 0])
            normal = np.cross(pt1-pt2, np.array([0, 0, 1]))
            normal = normal/np.linalg.norm(normal)
            normal_i.append(normal[0:2].reshape(1,2))
            
            if vis and i % 5 == 0:
                cv2.line(img, tuple(pt1[0:2].astype(int)), tuple((pt1[0:2]+10*normal[0:2]).astype(int)), (0, 0, 255), thickness=2)
        normals.append(normal_i)
        assert np.array(normal_i).shape == np.array(contour).shape
    if vis:
        cv2.drawContours(img, contours, -1, (0,255,0), 3)

    assert len(normals) == len(contours)
    return normals

def computeD(contours, data, img, size):
    normals = computeNormals(contours)
    for contour, normal in zip(contours, normals):
        for i in range(len(contour)):
            p = contour[i,0]
            p = (p[1], p[0]) #openCV indexes column-first
            neighbors = getNeighbor(img, p, size)
            data[p[0], p[1]] = np.abs(np.dot(computeIsophote(neighbors), np.array(normal)[i,0]))/255
    return data

def computeC(contours, confidence, maskThresh_0, size):
    for contour in contours:
        for i in range(len(contour)):
            p = contour[i,0]
            p = (p[1], p[0]) #openCV indexes column-first
            neighbors = getNeighbor(confidence, p, size)*getNeighbor(maskThresh_0, p, size)/255
            confidence[p[0], p[1]] = np.sum(neighbors)/size**2
    return confidence

def getNeighbor(img, p, size):
    neighbors = []
    for i in range(size):
        neighborRow = []
        indI = p[0]-size//2+i
        if indI >=0 and indI < len(img):
            for j in range(size):
                indJ = p[1]-size//2+j
                if indJ >=0 and indJ < len(img[0]):
                    neighborRow.append(img[indI, indJ])
            neighbors.append(neighborRow)
    return np.array(neighbors)

def computeIsophote(neighbors):
    grad = np.gradient(neighbors)
    gradx, grady = grad[0], grad[1]
    mag = gradx**2 + grady**2
    ind = np.unravel_index(np.argmax(mag, axis=None), mag.shape)
    maxGrad = np.array([gradx[ind], grady[ind], 0])
    isophote = np.cross(maxGrad, np.array([0,0,1]))
    isophote = isophote/np.linalg.norm(isophote)

    #Visualize
    # cv2.line(neighbors, (4,4), tuple(((4,4)+10*isophote[0:2]).astype(int)), (0, 255, 0), thickness=1)
    # plt.imshow(neighbors)
    # plt.show()
    return isophote[0:2]

def findBestPoint(contours, priority):
    bestPriority = np.NINF
    bestP = -1
    for contour in contours:
        for i in range(len(contour)):
            p = contour[i,0]
            p = (p[1], p[0]) #openCV indexes column-first
            priority_i = priority[p[0], p[1]]

            if priority_i > bestPriority:
                bestPriority = priority_i
                bestP = p

    return bestP

def findBestExemplar(p, img_lab, mask, maskEdgeCase, size):
    r = size//2
    neighborMask = getNeighbor(np.repeat(maskEdgeCase[:,:,np.newaxis], 3, axis=2)/255, p, size)
    neighbors = getNeighbor(img_lab, p, size)*neighborMask
    bestD = np.Inf
    bestExemplar = -1
    bestInd = -1

    stride = max(size//9, 1) #speed up large images
    for i in range(r+1,img_lab.shape[0]-r-2, stride):
        for j in range(r+1,img_lab.shape[1]-r-2, stride):
            if np.count_nonzero(maskEdgeCase[i-r:i+r+1, j-r:j+r+1] == 0) > 0:
                continue #skip regions that include target
            neighbors_i = img_lab[i-r:i+r+1, j-r:j+r+1]
            if not neighbors.shape == neighbors_i.shape:
                continue
            d = np.linalg.norm((neighbors-neighbors_i*neighborMask).flatten())
            if d<bestD:
                bestD = d
                bestExemplar = neighbors_i
                bestInd = np.array([i,j])
    return bestExemplar, bestInd

def buildExemplarImage(img_lab, mask, size):
    r = size//2
    exemplarImage = np.zeros((img_lab.shape[0], img_lab.shape[1], img_lab.shape[2], size**2))
    stride = 1
    for i in range(r,img_lab.shape[0]-r-1, stride):
        for j in range(r,img_lab.shape[1]-r-1, stride):
            if np.count_nonzero(mask[i-r:i+r+1, j-r:j+r+1] == 0) > 0:
                continue #skip regions that include target
            exemplarImage[i,j] = img_lab[i-r:i+r+1, j-r:j+r+1].reshape(size**2, 3).T
    return exemplarImage

def findBestExemplarVect(p, exemplar, img_lab, mask, size):
    #Vectorizes the opperation in findBestExemplar
    r = size//2
    neighborMask = getNeighbor(np.repeat(mask[:,:,np.newaxis], 3, axis=2)/255, p, size)
    neighbors = (getNeighbor(img_lab, p, size)*neighborMask)
    print(neighbors.dtype)
    neighborShape = neighbors.shape
    if neighbors.shape[0] != size or neighbors.shape[1] != size:
        print("Warning: Size mismatch. Using alternate method (slower).")
        return findBestExemplar(p, img_lab, mask, size)
    else:
        neighbors = neighbors.reshape(size**2, 3).T
        #neighbors = neighbors.reshape(size**2, 3).T.reshape(1, 1, -1)
    pImg = np.tile(neighbors, (img_lab.shape[0], img_lab.shape[1], 1, 1))

    pImg = pImg.reshape((img_lab.shape[0], img_lab.shape[1], -1))
    #pImg = neighbors
    print(pImg.dtype)
    exemplar = exemplar.reshape((img_lab.shape[0], img_lab.shape[1], -1))
    print(exemplar.dtype)
    
    d = np.linalg.norm(pImg-exemplar*(pImg!=0), axis=2)
    #d = np.linalg.norm((pImg-exemplar)[:,:,pImg[0,0]!=0], axis=2)
    dimg = (d/np.max(d)*255).astype(np.uint8)

    while True:
        i,j = np.unravel_index(np.argmin(d, axis=None), d.shape)
        neighbors_i = img_lab[i-r:i+r+1, j-r:j+r+1]
        if not neighborShape == neighbors_i.shape:
            print("Error: mismatched shapes")

        d[i,j] = np.inf

        if np.count_nonzero(mask[i-r:i+r+1, j-r:j+r+1] == 0) == 0:
            break #check if region excludes target
        else:
            print("best in target region")

    
    # cv2.rectangle(dimg, (p[1]-r, p[0]-r), (p[1]+r+1, p[0]+r+1), (255,0,0), 1)
    # cv2.rectangle(dimg, (j-r, i-r), (j+r+1, i+r+1), (0,0,255), 1)
    # plt.imshow(dimg)
    # plt.show()

    return neighbors_i, (i,j)


if __name__ == "__main__":
    

    for filename in os.listdir(imgDir):
        if not glob.glob(maskDir+filename.split('.')[0]+'_mask.*'):
            print("Image " + filename + " has no mask. Skipping...")
            continue
        print("Reading "+filename)
        img = cv2.imread(imgDir+filename)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_lab = color.rgb2lab(img_rgb)
        mask = cv2.imread(maskDir+filename.split('.')[0]+'_mask.jpg')
        maskGray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, maskThresh = cv2.threshold(maskGray, 50, 255, 0)
        size = int(min(img.shape[:-1])/30//2*2+1) #The size of the fill region (rounded to nearest odd number)
        print("Texel size: ", size)
        #exemplarImg = buildExemplarImage(img_lab, maskThresh, size)

        #Exclude edges to prevent odd boundary effects
        r = size//2
        maskEdgeCase = np.copy(maskThresh) #store unmodified mask
        maskThresh[:(r+1),:] = 255
        maskThresh[-(r+1):,:] = 255
        maskThresh[:,:(r+1)] = 255
        maskThresh[:,-(r+1):] = 255

        confidence = maskThresh / 255
        data = np.zeros(confidence.shape)
        maskThresh_0 = maskThresh
        iteration = 0
        

        while True:
            print("Filling region "+str(iteration))
            #1a Identify the fill front
            contours, hierarchy = cv2.findContours(maskThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours)==1:
                break #exit condition if full region has been filled
            
            contours = contours[1:] #ignore edge of image

            #1b Compute Priorities
            confidence = computeC(contours, confidence, maskThresh_0, size)
            data = computeD(contours, data, img, size)
            priority = confidence*data
            # plt.imshow(data)
            # plt.show()

            #2a find best point
            p = findBestPoint(contours, priority)

            #2b Find exemplar that minimizes d(phi_p, phi_q)
            exemplar, exemplar_p = findBestExemplar(p, img_lab, maskThresh_0, maskEdgeCase, size) #Slower
            #exemplar, exemplar_p = findBestExemplarVect(p, exemplarImg, img_lab, maskThresh_0, size) #Faster, large memory usage (20GB on large images)

            #2c Copy image data
            img_lab[p[0]-r:p[0]+r+1, p[1]-r:p[1]+r+1] = exemplar
            img = (color.lab2rgb(img_lab)*255).astype(np.uint8)

            #3 Update C
            confidence[p[0]-r:p[0]+r+1, p[1]-r:p[1]+r+1] = confidence[p[0], p[1]]
            confidence[:(r+2),:] = 0
            confidence[-(r+2):,:] = 0
            confidence[:,:(r+2)] = 0
            confidence[:,-(r+2):] = 0

            #Other updates
            img_lab = color.rgb2lab(img) #check here
            maskThresh[p[0]-r:p[0]+r+1, p[1]-r:p[1]+r+1] = 255

            if iteration % 20 ==0: #or p == (92,0):
                # fig, axs = plt.subplots(1, 3)
                img_vis = np.array(img)
                cv2.rectangle(img_vis, (p[1]-r, p[0]-r), (p[1]+r+1, p[0]+r+1), (255,0,0), 1)
                try:
                    cv2.rectangle(img_vis, (exemplar_p[1]-r, exemplar_p[0]-r), (exemplar_p[1]+r+1, exemplar_p[0]+r+1), (0,0,255), 1)
                    cv2.drawContours(img_vis, contours, -1, (0,255,0), 3)
                    # axs[0].imshow(img_vis)
                    # axs[2].imshow(confidence)
                    # axs[3].imshow(data)
                    # fig.savefig(saveDir+'tmp_'+filename, dpi=1000)
                    cv2.imwrite(saveDir+filename.split('.')[0]+'_confidence.png', (confidence/np.max(confidence)*255).astype(np.uint8))

                    im_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(saveDir+filename, np.array(im_bgr))
                except:
                    print(getNeighbor(img_lab, p, size).shape)
                    break

            iteration += 1
        
        im_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(saveDir+filename+"_final", np.array(im_bgr))
