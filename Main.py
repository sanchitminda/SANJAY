import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

option_tiny = {'model':'cfg\\tiny-yolo.cfg',
          'load': 'bin\\tiny-yolo.weights',
          'gpu': 0.7,
          'threshold':0.3,
          }

option = {'model':'cfg\\yolo.cfg',
          'load': 'bin\\yolov2.weights',
          'gpu': 0.7,
          'threshold':0.3,
          }

tfnet = TFNet(option)


capture = cv2.VideoCapture(0)
# capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
# capture.set(cv2.CAP_PROP_FRAME_WIDTH,720)
colors = [ tuple(255 * np.random.rand(3)) for i in range(50)]


while capture.isOpened():
    stime = time.time()
    ret, fram = capture.read()

    if ret:
        results = tfnet.return_predict(fram)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = '{} {:.0f}'.format(result['label'],result['confidence']*100)
            fram = cv2.rectangle(fram,tl,br,color,3)
            fram = cv2.putText(fram,label,tl,cv2.FONT_HERSHEY_COMPLEX,1,color,2)
        cv2.imshow('frame',fram)
        print('fps {:.1f}'.format(1/(time.time() - stime)))
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break
