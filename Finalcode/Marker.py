import cv2
import cv2.aruco as aruco
import freenect
import numpy as np
import copy




class marker(object):
  """docstring for ClassName"""

  distCoeffs = np.array([2.6451622333009589e-01, -8.3990749424620825e-01,
         -1.9922302173693159e-03, 1.4371995932897616e-03,
         9.1192465078713847e-01])

  cameraMatrix = np.array([5.2921508098293293e+02, 0., 3.2894272028759258e+02, 0.,
         5.2556393630057437e+02, 2.6748068171871557e+02, 0., 0., 1.])



  def __init__(self):
    # super(ClassName, self).__init__()
    # self.arg = arg
    self.ARUCO_PARAMETERS = aruco.DetectorParameters_create()
    self.ARUCO_PARAMETERS.doCornerRefinement = True
    self.ARUCO_PARAMETERS.adaptiveThreshWinSizeStep = 4
    self.ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_100)
    self.T = np.zeros(3)
    self.cameraMatrix = self.cameraMatrix.reshape((3,3))
    self.distCoeffs = np.expand_dims(self.distCoeffs, axis = 0)
    self.Flag = False

  def get_video(self):
  	video, _ = freenect.sync_get_video()
  	video = cv2.cvtColor(video,cv2.COLOR_RGB2BGR)
  	return video

  def StartDetecting(self):
    # while 1:
      frame = self.get_video()
      img = copy.deepcopy(frame)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.ARUCO_DICT, parameters=self.ARUCO_PARAMETERS)

      # img = aruco.drawDetectedMarkers(img, corners, ids)

      if ids is not None and np.sum(np.array(ids)) == 160:
        rvecs, tvecs = aruco.estimatePoseSingleMarkers(corners, 4.8, self.cameraMatrix, self.distCoeffs)

        for i in range(len(ids)):
          # img = aruco.drawAxis(rgb, cameraMatrix, distCoeffs, rvecs[0][i,:,:], tvecs[0][i,:,:], 3)
          if ids[i] == 70:
            # cv2.Rodrigues(rvecs[0][i].reshape((1,3)), R1)
            T1 = tvecs[0][i].reshape(3,)
          elif ids[i] == 90:
            # cv2.Rodrigues(rvecs[0][i].reshape((1,3)), R2)
            T2 = tvecs[0][i].reshape(3,)
            
        self.T = T1 - T2
        self.Flag = False

        # return self.T

      if ids is None:
        self.Flag = True

      # cv2.waitKey(1)




