#!/usr/bin/evn python

"""
@file    AutoPano.py
@author  rohithjayarajan
@date 02/17/2019

Licensed under the
GNU General Public License v3.0
"""

import numpy as np
import cv2
import argparse
from PIL import Image
import glob
from skimage.feature import peak_local_max
import math
import random
import matplotlib.pyplot as plt
from HelperFunctions import HelperFunctions

debug = False


class Stitcher:

    """
    Read a set of images for Panorama stitching
    """

    def __init__(self, BasePath, NumFeatures):
        self.BasePath = BasePath
        InputImageList = []
        for filename in sorted(glob.glob(self.BasePath + '/*.jpg')):
            ImageTemp = cv2.imread(filename)
            InputImageList.append(ImageTemp)
        self.NumFeatures = NumFeatures
        self.Images = np.array(InputImageList)
        self.NumImages = len(InputImageList)
        self.FirstHalf = []
        self.LastHalf = []
        self.MiddleImage = None
        self.HelperFunctions = HelperFunctions()

    def ImageContainers(self):
        self.CenterImgId = self.NumImages/2
        self.MiddleImage = self.Images[self.CenterImgId]
        for IdX in range(self.NumImages):
            if(IdX <= self.CenterImgId):
                self.FirstHalf.append(self.Images[IdX])
            else:
                self.LastHalf.append(self.Images[IdX])

    """
	Corner Detection
	Save Corner detection output as corners.png
	"""

    def DetectCornersShiTomasi(self, Image):
        grayImage = np.float32(cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY))
        ShiTomasiCorners = cv2.goodFeaturesToTrack(
            grayImage, self.NumFeatures, 0.01, 10)
        ShiTomasiCorners = np.int0(ShiTomasiCorners)
        ShiTomasiCorners = np.reshape(
            ShiTomasiCorners, (ShiTomasiCorners.shape[0], 2))
        return ShiTomasiCorners

    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

    def DetectCornersHarris(self, InputImage, Threshold=0.005):
        grayImage = np.float32(cv2.cvtColor(InputImage, cv2.COLOR_BGR2GRAY))
        CornerImage = cv2.cornerHarris(grayImage, 2, 3, 0.04)
        tempImage = grayImage.copy()
        tempImage[CornerImage < Threshold * CornerImage.max()] = 0
        return tempImage

    def ANMS(self, CornerScoreImage, Nbest=100):

        LocalMaxima = peak_local_max(CornerScoreImage, min_distance=15)
        r = np.zeros((LocalMaxima.shape[0], 3), dtype=np.float32)
        r[:, 2] = float("inf")
        IdX = 0

        for i in LocalMaxima:
            r[IdX][0] = i[1]
            r[IdX][1] = i[0]
            ED = float("inf")
            for j in LocalMaxima:
                if (CornerScoreImage[j[0], j[1]] > CornerScoreImage[i[0], i[1]]):
                    ED = (j[1] - i[1])**2 + (j[0] - i[0])**2
                if (ED < r[IdX][2]):
                    r[IdX][2] = ED
            IdX += 1
        ind = np.argsort(r[:, 2])
        r = r[ind]

        if debug:
            print("Features after ANMS: " + str(len(r[0:Nbest, 0:2])))
        return r[0:Nbest, 0:2]

    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

    def FeatureDescriptor(self, Features, Image):
        FeatureDescriptors = []
        PatchSize = 40
        Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

        for feature in Features:
            PatchY = int(feature[0] - PatchSize/2)
            PatchX = int(feature[1] - PatchSize/2)
            Patch = Image[PatchX:PatchX+PatchSize, PatchY:PatchY+PatchSize]

            if [Patch.shape[0], Patch.shape[1]] == [40, 40]:
                FeatureSet = []
                FeatureSet.append(feature)
                Patch = cv2.GaussianBlur(Patch, (5, 5), 1.2, 1.4)
                Patch = Patch[0:40:5, 0:40:5]
                # Patch = cv2.resize(Patch, dsize=(8, 8))
                Descriptor = Patch.ravel()
                mu = np.mean(Descriptor)
                sigma = np.std(Descriptor)
                Descriptor = (Descriptor - mu)/sigma
                FeatureSet.append(Descriptor)
                FeatureDescriptors.append(FeatureSet)

        return FeatureDescriptors

    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""

    def FeatureMatching(self, FeatureDescriptor1, FeatureDescriptor2):

        MatchingFeatures = []
        des1 = []
        des2 = []
        for features1 in FeatureDescriptor1:
            best1 = float("inf")
            best2 = float("inf")
            Match = []
            for features2 in FeatureDescriptor2:
                SSD = self.HelperFunctions.ComputeSSD(
                    features1[1], features2[1])
                if (SSD < best1 and SSD < best2):
                    best2 = best1
                    best1 = SSD
                    tempBest1 = features1[0]
                    tempBest2 = features2[0]
                    tempdes1 = features1[1]
                    tempdes2 = features2[1]
                if (SSD > best1 and SSD < best2):
                    best2 = SSD
            ratio = float(best1)/float(best2)
            if (ratio < 0.5):
                Match.append(tempBest1)
                Match.append(tempBest2)
                MatchNp = np.array(Match)
                MatchingFeatures.append(MatchNp)
                des1.append(tempdes1)
                des2.append(tempdes2)

        size = 1
        angle = 1
        response = 1
        octave = 1
        class_id = 1
        kp1 = []
        kp2 = []

        for points in MatchingFeatures:
            kp1.append(cv2.KeyPoint(x=points[0][0], y=points[0][1], _size=size, _angle=angle,
                                    _response=response, _octave=octave, _class_id=class_id))
            kp2.append(cv2.KeyPoint(x=points[1][0], y=points[1][1], _size=size, _angle=angle,
                                    _response=response, _octave=octave, _class_id=class_id))
        return np.array(MatchingFeatures), np.array(kp1), np.array(kp2), np.array(des1), np.array(des2)

    """
	Refine: RANSAC, Estimate Homography
	"""

    def RANSACHomography(self, MatchingFeatures, Nmax, tolerance=0.5):

        InlierPercent = 0.0
        RANSACiter = 0
        maxHits = 0

        while InlierPercent < 0.95 and RANSACiter < Nmax:
            p = random.sample(MatchingFeatures, 4)
            p1 = p[0][0]
            p2 = p[1][0]
            p3 = p[2][0]
            p4 = p[3][0]

            p1d = p[0][1]
            p2d = p[1][1]
            p3d = p[2][1]
            p4d = p[3][1]

            pts1 = np.float32([[p1[1], p1[0]], [p2[1], p2[0]],
                               [p3[1], p3[0]], [p4[1], p4[0]]])

            pts2 = np.float32([[p1d[1], p1d[0]], [p2d[1], p2d[0]],
                               [p3d[1], p3d[0]], [p4d[1], p4d[0]]])

            H = cv2.getPerspectiveTransform(pts1, pts2)

            SSDVal = self.HelperFunctions.SSDRansac(MatchingFeatures, H)
            hits = (SSDVal < tolerance).sum()
            if(hits > maxHits):
                maxHits = hits
                IndicesBest = np.argwhere(SSDVal < tolerance)
            RANSACiter += 1
            InlierPercent = float(hits)/float(len(MatchingFeatures))

        Inliers = [MatchingFeatures[i] for i in IndicesBest]
        Inliers = np.array(Inliers)
        Inliers = np.reshape(Inliers, (Inliers.shape[0], 2, 2))
        # print("Inlier percent")
        # print(InlierPercent)
        # print(Inliers.shape)

        A = np.zeros([2*len(Inliers), 9])
        idX = 0
        for inliers in Inliers:
            y = inliers[0][1]
            x = inliers[0][0]
            ydash = inliers[1][1]
            xdash = inliers[1][0]
            A[idX][0] = x
            A[idX][1] = y
            A[idX][2] = 1
            A[idX][6] = -xdash*x
            A[idX][7] = -xdash*y
            A[idX][8] = -xdash
            idX += 1
            A[idX][3] = x
            A[idX][4] = y
            A[idX][5] = 1
            A[idX][6] = -ydash*x
            A[idX][7] = -ydash*y
            A[idX][8] = -ydash
            idX += 1

        U, S, V = np.linalg.svd(A, full_matrices=True)
        Hpred = V[8, :].reshape((3, 3))

        return Hpred

    def EstimateHomography(self, Image1, Image2, RansacIter=1000):
        CornersImage1 = self.DetectCornersShiTomasi(Image1)
        CornersImage2 = self.DetectCornersShiTomasi(Image2)
        FeatureDescriptorImage1 = self.FeatureDescriptor(CornersImage1, Image1)
        FeatureDescriptorImage2 = self.FeatureDescriptor(CornersImage2, Image2)
        Matches, Kp1, Kp2, Des1, Des2 = self.FeatureMatching(
            FeatureDescriptorImage1, FeatureDescriptorImage2)
        self.HelperFunctions.DrawMatches(Image1, Image2, Kp1, Kp2)
        H = self.RANSACHomography(Matches, RansacIter)
        Hinv = np.linalg.inv(H)

        return H, Hinv

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

    def RemoveBlackBoundary(self, ImageIn):
        gray = cv2.cvtColor(ImageIn, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        ImageOut = ImageIn[y:y+h, x:x+w]
        return ImageOut

    def Warping(self, Img, Homography, NextShape):
        nH, nW, _ = Img.shape
        Borders = np.array([[0, nW, nW, 0], [0, 0, nH, nH], [1, 1, 1, 1]])
        BordersNew = np.dot(Homography, Borders)
        Ymin = min(BordersNew[1]/BordersNew[2])
        Xmin = min(BordersNew[0]/BordersNew[2])
        Ymax = max(BordersNew[1]/BordersNew[2])
        Xmax = max(BordersNew[0]/BordersNew[2])
        if Ymin < 0:
            MatChange = np.array(
                [[1, 0, -1 * Xmin], [0, 1, -1 * Ymin], [0, 0, 1]])
            Hnew = np.dot(MatChange, Homography)
            h = int(round(Ymax - Ymin)) + NextShape[0]
        else:
            MatChange = np.array(
                [[1, 0, -1 * Xmin], [0, 1, Ymin], [0, 0, 1]])
            Hnew = np.dot(MatChange, Homography)
            h = int(round(Ymax + Ymin)) + NextShape[0]
        w = int(round(Xmax - Xmin)) + NextShape[1]
        sz = (w, h)
        PanoHolder = cv2.warpPerspective(Img, Hnew, dsize=sz)
        return PanoHolder, int(Xmin), int(Ymin)

    def Blender(self):
        Pano = self.Images[0]
        for NextImage in self.Images[1:]:
            H, Hinv = self.EstimateHomography(Pano, NextImage)
            PanoHolder, oX, oY = self.Warping(Pano, H, NextImage.shape)
            oX = abs(oX)
            oY = abs(oY)
            for IdY in range(oY, NextImage.shape[0]+oY):
                for IdX in range(oX, NextImage.shape[1]+oX):
                    y = IdY - oY
                    x = IdX - oX
                    PanoHolder[IdY, IdX, :] = NextImage[y, x, :]
            # Pano = self.RemoveBlackBoundary(PanoHolder)
            Pano = PanoHolder
        PanoResize = cv2.resize(Pano, (1280, 1024))
        self.HelperFunctions.ShowImage(PanoResize, 'PanoResize')
        PanoResize = cv2.GaussianBlur(PanoResize, (5, 5), 1.2)
        return PanoResize


def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/rohith/CMSC733/git/Auto-Pano/Data/Test/TestSet22',
                        help='Base path of images, Default:/home/rohith/CMSC733/git/Auto-Pano/Data/Train/Set1')
    Parser.add_argument('--NumFeatures', default='700',
                        help='Number of best features to extract from each image, Default:100')

    Args = Parser.parse_args()
    NumFeatures = int(Args.NumFeatures)
    BasePath = Args.BasePath

    myStitcher = Stitcher(BasePath, NumFeatures)
    Pano = myStitcher.Blender()


if __name__ == '__main__':
    main()
